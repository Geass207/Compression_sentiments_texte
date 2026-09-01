import copy
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.classifier import SentimentHead, evaluate_classifier, train_classifier
from src.pruning import (
    iterative_prune_classifier_weights,
    obd_classifier_weight_scores,
)
from src.experiment_config import (
    CLASSIFIER_BATCH_SIZE,
    CLASSIFIER_EPOCHS,
    CLASSIFIER_LEARNING_RATE,
    CLASSIFIER_PATIENCE,
    HEAD_ARCHITECTURES,
    OBD_HESSIAN_VECTORS,
    OBD_PRUNING_STEP,
    OBD_RETRAINING_EPOCHS,
    SEED,
)


WORD2VEC_CODE = "W1"
PRUNING_STEP = OBD_PRUNING_STEP
BATCH_SIZE = CLASSIFIER_BATCH_SIZE
TRAINING_EPOCHS = CLASSIFIER_EPOCHS
RETRAINING_EPOCHS = OBD_RETRAINING_EPOCHS
LEARNING_RATE = CLASSIFIER_LEARNING_RATE
PATIENCE = CLASSIFIER_PATIENCE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train = pd.read_pickle("artifacts/train_encoded.pkl")
validation = pd.read_pickle("artifacts/validation_encoded.pkl")
test = pd.read_pickle("artifacts/test_encoded.pkl")
train_labels = train["sentiment"].to_numpy()
validation_labels = validation["sentiment"].to_numpy()
test_labels = test["sentiment"].to_numpy()

results = []
pruning_rows = []
training_rows = []


def record_training_history(
    word2vec_code,
    head_code,
    model_code,
    phase,
    history,
    pruning_iteration=None,
):
    for epoch in range(len(history["train_loss"])):
        training_rows.append(
            {
                "word2vec": word2vec_code,
                "head": head_code,
                "model": model_code,
                "phase": phase,
                "pruning_iteration": pruning_iteration,
                "epoch": epoch + 1,
                "train_loss": history["train_loss"][epoch],
                "validation_loss": history["validation_loss"][epoch],
                "validation_accuracy": history["validation_accuracy"][epoch],
                "validation_f1": history["validation_f1"][epoch],
            }
        )


for word2vec_code in (WORD2VEC_CODE,):
    train_vectors = torch.load(
        f"artifacts/train_vectors_{word2vec_code}.pt", weights_only=True
    )
    validation_vectors = torch.load(
        f"artifacts/validation_vectors_{word2vec_code}.pt", weights_only=True
    )
    test_vectors = torch.load(
        f"artifacts/test_vectors_{word2vec_code}.pt", weights_only=True
    )
    word2vec_checkpoint = torch.load(
        f"artifacts/word2vec_{word2vec_code}.pt",
        map_location="cpu",
        weights_only=True,
    )
    word2vec_training_parameters = sum(
        tensor.numel()
        for tensor in word2vec_checkpoint["model_state"].values()
    )
    word2vec_inference_parameters = word2vec_checkpoint["model_state"][
        "input_embeddings.weight"
    ].numel()
    input_dim = train_vectors.shape[1]

    for head_code, dimensions in HEAD_ARCHITECTURES.items():
        full_dims = dimensions["full"]
        direct_dims = dimensions["direct"]

        torch.manual_seed(SEED)
        m1 = SentimentHead(input_dim, full_dims).to(device)
        m1_history = train_classifier(
            m1,
            train_vectors,
            train_labels,
            validation_vectors,
            validation_labels,
            epochs=TRAINING_EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
            patience=PATIENCE,
        )
        record_training_history(
            word2vec_code, head_code, "M1", "initial_training", m1_history
        )

        full_weight_count = sum(
            parameter.numel()
            for name, parameter in m1.named_parameters()
            if name.endswith("weight")
        )
        direct_reference = SentimentHead(input_dim, direct_dims)
        direct_weight_count = sum(
            parameter.numel()
            for name, parameter in direct_reference.named_parameters()
            if name.endswith("weight")
        )
        target_sparsity = 1 - direct_weight_count / full_weight_count

        m2, m3, sparse_masks, pruning_history = (
            iterative_prune_classifier_weights(
                copy.deepcopy(m1),
                train_vectors,
                train_labels,
                validation_vectors,
                validation_labels,
                target_sparsity=target_sparsity,
                score_function=obd_classifier_weight_scores,
                pruning_step=PRUNING_STEP,
                retraining_epochs=RETRAINING_EPOCHS,
                batch_size=BATCH_SIZE,
                hessian_vectors=OBD_HESSIAN_VECTORS,
                lr=LEARNING_RATE,
                patience=None,
            )
        )
        torch.save(
            pruning_history,
            f"artifacts/classifier_pruning_history_{word2vec_code}_{head_code}.pt",
        )

        for iteration, step in enumerate(pruning_history, start=1):
            for stage in ("before_retraining", "after_retraining"):
                metrics = step[f"validation_metrics_{stage}"]
                pruning_rows.append(
                    {
                        "word2vec": word2vec_code,
                        "head": head_code,
                        "iteration": iteration,
                        "sparsity": step["sparsity"],
                        "stage": stage,
                        "mean_pruned_score": step[
                            "pruned_scores"
                        ].mean().item(),
                        "scoring_seconds": step["scoring_seconds"],
                        **metrics,
                    }
                )
            record_training_history(
                word2vec_code,
                head_code,
                "M3",
                "pruning_retraining",
                step["retraining_history"],
                pruning_iteration=iteration,
            )

        torch.manual_seed(SEED)
        m4 = SentimentHead(input_dim, direct_dims).to(device)
        m4_history = train_classifier(
            m4,
            train_vectors,
            train_labels,
            validation_vectors,
            validation_labels,
            epochs=TRAINING_EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
            patience=PATIENCE,
        )
        record_training_history(
            word2vec_code, head_code, "M4", "direct_training", m4_history
        )

        models = {
            "M1": (m1, {}, "complet"),
            "M2": (m2, sparse_masks, "prune"),
            "M3": (m3, sparse_masks, "prune_reentraine"),
            "M4": (m4, {}, "direct_taille_finale"),
        }

        for model_code, (model, masks, description) in models.items():
            validation_metrics, _ = evaluate_classifier(
                model,
                validation_vectors,
                validation_labels,
                batch_size=BATCH_SIZE,
            )
            test_metrics, _ = evaluate_classifier(
                model,
                test_vectors,
                test_labels,
                batch_size=BATCH_SIZE,
            )
            weight_count = sum(
                parameter.numel()
                for name, parameter in model.named_parameters()
                if name.endswith("weight")
            )
            active_weights = (
                sum(mask.sum().item() for mask in masks.values())
                if masks
                else weight_count
            )
            bias_count = sum(
                parameter.numel()
                for name, parameter in model.named_parameters()
                if name.endswith("bias")
            )
            head_parameters = sum(
                parameter.numel() for parameter in model.parameters()
            )

            row = {
                "word2vec": word2vec_code,
                "head": head_code,
                "model": model_code,
                "description": description,
                "embedding_dim": input_dim,
                "hidden_dims": "-".join(map(str, model.hidden_dims)),
                "target_sparsity": target_sparsity,
                "head_sparsity": 1 - active_weights / weight_count,
                "word2vec_training_parameters": word2vec_training_parameters,
                "word2vec_inference_parameters": word2vec_inference_parameters,
                "head_parameters": head_parameters,
                "head_total_weights": weight_count,
                "head_active_weights": active_weights,
                "head_bias_parameters": bias_count,
                "head_effective_parameters": active_weights + bias_count,
                "head_dense_macs": weight_count,
                "head_effective_macs": active_weights,
                "stored_inference_parameters": (
                    word2vec_inference_parameters + head_parameters
                ),
                "effective_parameters": (
                    word2vec_inference_parameters + active_weights + bias_count
                ),
            }
            row.update(
                {
                    f"validation_{name}": value
                    for name, value in validation_metrics.items()
                }
            )
            row.update(
                {
                    f"test_{name}": value
                    for name, value in test_metrics.items()
                }
            )
            results.append(row)

            torch.save(
                {
                    "word2vec": word2vec_code,
                    "head": head_code,
                    "model": model_code,
                    "description": description,
                    "model_state": model.state_dict(),
                    "input_dim": input_dim,
                    "hidden_dims": model.hidden_dims,
                    "masks": masks,
                    "target_sparsity": target_sparsity,
                    "validation_metrics": validation_metrics,
                    "test_metrics": test_metrics,
                },
                f"artifacts/classifier_{word2vec_code}_{head_code}_{model_code}.pt",
            )

        pd.DataFrame(results).to_csv(
            "artifacts/classification_results.csv", index=False
        )
        pd.DataFrame(pruning_rows).to_csv(
            "artifacts/classifier_pruning_metrics.csv", index=False
        )
        pd.DataFrame(training_rows).to_csv(
            "artifacts/classifier_training_history.csv", index=False
        )

print(f"{len(results)} combinaisons évaluées et sauvegardées.")
