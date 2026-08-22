import sys
from pathlib import Path

import pandas as pd
import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pruning import (
    iterative_prune_cbow_dimensions,
    obd_cbow_dimension_scores,
)
from src.word2vec import (
    CBOW,
    evaluate_cbow,
    make_cbow_dataset,
    negative_sampling_distribution,
    text_vectors,
    train_cbow,
)


FULL_DIMENSION = 200
PRUNED_DIMENSION = 56
PRUNING_STEP = 16
WINDOW_SIZE = 2
NEGATIVE_SAMPLES = 5
BATCH_SIZE = 512
TRAINING_EPOCHS = 5
RETRAINING_EPOCHS = 1
LEARNING_RATE = 1e-3
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)

train = pd.read_pickle("artifacts/train_encoded.pkl")
validation = pd.read_pickle("artifacts/validation_encoded.pkl")
test = pd.read_pickle("artifacts/test_encoded.pkl")
tokenizer = Tokenizer.from_file("artifacts/tokenizer.json")

vocab_size = tokenizer.get_vocab_size()
probabilities = negative_sampling_distribution(train, vocab_size)
train_dataset = make_cbow_dataset(train, WINDOW_SIZE)
validation_dataset = make_cbow_dataset(validation, WINDOW_SIZE)

# W1 : Word2Vec complet, dimension 200.
w1 = CBOW(vocab_size, FULL_DIMENSION).to(device)
train_cbow(
    w1,
    train_dataset,
    probabilities,
    k=NEGATIVE_SAMPLES,
    batch_size=BATCH_SIZE,
    epochs=TRAINING_EPOCHS,
    lr=LEARNING_RATE,
)

# W2 : dernier état juste après pruning. W3 : même état après réentraînement.
w2, w3, kept_dimensions, pruning_history = iterative_prune_cbow_dimensions(
    w1,
    train_dataset,
    probabilities,
    target_dimension=PRUNED_DIMENSION,
    validation_dataset=validation_dataset,
    score_function=obd_cbow_dimension_scores,
    pruning_step=PRUNING_STEP,
    retraining_epochs=RETRAINING_EPOCHS,
    k=NEGATIVE_SAMPLES,
    batch_size=BATCH_SIZE,
    lr=LEARNING_RATE,
)

# W4 : modèle directement entraîné avec la dimension finale 56.
torch.manual_seed(SEED)
w4 = CBOW(vocab_size, PRUNED_DIMENSION).to(device)
train_cbow(
    w4,
    train_dataset,
    probabilities,
    k=NEGATIVE_SAMPLES,
    batch_size=BATCH_SIZE,
    epochs=TRAINING_EPOCHS,
    lr=LEARNING_RATE,
)

variants = {
    "W1": (w1, "complet_d200"),
    "W2": (w2, "prune_d56"),
    "W3": (w3, "prune_reentraine_d56"),
    "W4": (w4, "direct_d56"),
}
splits = {"train": train, "validation": validation, "test": test}
metrics = []

for code, (model, description) in variants.items():
    validation_loss = evaluate_cbow(
        model,
        validation_dataset,
        probabilities,
        k=NEGATIVE_SAMPLES,
        batch_size=BATCH_SIZE,
        seed=SEED,
    )
    training_parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    inference_parameter_count = model.input_embeddings.weight.numel()
    metrics.append(
        {
            "word2vec": code,
            "description": description,
            "embedding_dim": model.embedding_dim,
            "training_parameters": training_parameter_count,
            "inference_parameters": inference_parameter_count,
            "validation_loss": validation_loss,
        }
    )
    torch.save(
        {
            "variant": code,
            "description": description,
            "model_state": model.state_dict(),
            "vocab_size": vocab_size,
            "embedding_dim": model.embedding_dim,
            "window_size": WINDOW_SIZE,
            "negative_samples": NEGATIVE_SAMPLES,
            "kept_original_dimensions": (
                kept_dimensions
                if code in {"W2", "W3"}
                else torch.arange(model.embedding_dim)
            ),
        },
        f"artifacts/word2vec_{code}.pt",
    )

    for split_name, data in splits.items():
        torch.save(
            text_vectors(model, data),
            f"artifacts/{split_name}_vectors_{code}.pt",
        )

pd.DataFrame(metrics).to_csv("artifacts/word2vec_metrics.csv", index=False)
torch.save(pruning_history, "artifacts/word2vec_pruning_history.pt")

pruning_rows = []
for iteration, step in enumerate(pruning_history, start=1):
    pruning_rows.append(
        {
            "iteration": iteration,
            "dimension": step["dimension"],
            "number_pruned": len(step["pruned_dimensions"]),
            "mean_pruned_score": step["pruned_scores"].mean().item(),
            "validation_loss_before_retraining": step[
                "validation_loss_before_retraining"
            ],
            "validation_loss_after_retraining": step[
                "validation_loss_after_retraining"
            ],
        }
    )

pd.DataFrame(pruning_rows).to_csv(
    "artifacts/word2vec_pruning_metrics.csv", index=False
)
