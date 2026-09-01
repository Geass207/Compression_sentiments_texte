"""Pruning MINT one-shot de la tête H2 entraînée avec Word2Vec W1 complet."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from time import perf_counter

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.classifier import SentimentHead, evaluate_classifier, train_classifier
from src.mint import (
    balanced_calibration_indices,
    collect_sentiment_head_activations,
    mint_group_scores,
    mint_weight_mask,
)
from src.experiment_config import (
    CLASSIFIER_BATCH_SIZE,
    CLASSIFIER_LEARNING_RATE,
    CLASSIFIER_PATIENCE,
    EMBEDDING_DIM,
    HEAD_ARCHITECTURES,
    MINT_GROUPS_PER_LAYER,
    MINT_MAXIMUM_PRUNING_FRACTION,
    MINT_PARTITION_SIZE,
    MINT_RETRAINING_EPOCHS,
    MINT_SAMPLES_PER_CLASS,
    SEED,
)


WORD2VEC_CODE = "W1"
HEAD_CODE = "H2"
EXPECTED_INPUT_DIM = EMBEDDING_DIM
EXPECTED_HIDDEN_DIMS = HEAD_ARCHITECTURES[HEAD_CODE]["full"]
DIRECT_HIDDEN_DIMS = HEAD_ARCHITECTURES[HEAD_CODE]["direct"]

DEFAULT_GROUPS_PER_LAYER = MINT_GROUPS_PER_LAYER
DEFAULT_SAMPLES_PER_CLASS = MINT_SAMPLES_PER_CLASS
DEFAULT_PARTITION_SIZE = MINT_PARTITION_SIZE
DEFAULT_MAXIMUM_PRUNING_FRACTION = MINT_MAXIMUM_PRUNING_FRACTION
DEFAULT_RETRAINING_EPOCHS = MINT_RETRAINING_EPOCHS

BATCH_SIZE = CLASSIFIER_BATCH_SIZE
LEARNING_RATE = CLASSIFIER_LEARNING_RATE
PATIENCE = CLASSIFIER_PATIENCE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calcule les scores MINT de W1/H2, masque les blocs les moins "
            "informatifs, puis réentraîne une fois la tête masquée."
        )
    )
    parser.add_argument(
        "--groups-per-layer",
        type=int,
        nargs=3,
        default=DEFAULT_GROUPS_PER_LAYER,
        metavar=("G0", "G1", "G2"),
        help=(
            "Nombre de groupes amont pour les trois transitions de H2 : "
            f"{EXPECTED_INPUT_DIM}→{EXPECTED_HIDDEN_DIMS[0]}, "
            f"{EXPECTED_HIDDEN_DIMS[0]}→{EXPECTED_HIDDEN_DIMS[1]} et "
            f"{EXPECTED_HIDDEN_DIMS[1]}→1."
        ),
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=DEFAULT_SAMPLES_PER_CLASS,
        help="Nombre d'observations de calibration tirées dans chaque classe.",
    )
    parser.add_argument(
        "--partition-size",
        type=int,
        default=DEFAULT_PARTITION_SIZE,
        help="Taille m de P, des sources et des donneurs pour chaque score.",
    )
    parser.add_argument(
        "--maximum-pruning-fraction",
        type=float,
        default=DEFAULT_MAXIMUM_PRUNING_FRACTION,
        help="Protection gamma appliquée séparément à chaque couche.",
    )
    parser.add_argument(
        "--target-sparsity",
        type=float,
        default=None,
        help=(
            "Sparsité de blocs par couche. Par défaut, reprend la sparsité "
            "cible du checkpoint H2/M1 calculée face à l'architecture directe."
        ),
    )
    parser.add_argument(
        "--retraining-epochs",
        type=int,
        default=DEFAULT_RETRAINING_EPOCHS,
        help="Nombre maximal d'époques du réentraînement MINT one-shot.",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if any(number <= 0 for number in args.groups_per_layer):
        raise ValueError("Tous les nombres de groupes doivent être positifs.")
    if args.samples_per_class <= 0 or args.partition_size <= 0:
        raise ValueError("Les tailles de calibration et de partition doivent être positives.")
    if args.samples_per_class * 2 < 3 * args.partition_size:
        raise ValueError(
            "La calibration binaire contient 2 * samples_per_class observations, "
            "mais MINT en exige au moins 3 * partition_size."
        )
    if not 0.0 <= args.maximum_pruning_fraction <= 1.0:
        raise ValueError("maximum-pruning-fraction doit appartenir à [0, 1].")
    if args.target_sparsity is not None and not 0.0 <= args.target_sparsity <= 1.0:
        raise ValueError("target-sparsity doit appartenir à [0, 1].")
    if args.retraining_epochs <= 0:
        raise ValueError("retraining-epochs doit être strictement positif.")


def validate_full_word2vec(checkpoint: dict, vector_dimension: int) -> None:
    """Vérifie que W1 est le Word2Vec complet à la dimension attendue."""
    if checkpoint.get("variant") != WORD2VEC_CODE:
        raise ValueError("Le checkpoint Word2Vec chargé n'est pas W1.")
    if checkpoint.get("embedding_dim") != EXPECTED_INPUT_DIM:
        raise ValueError(
            f"W1 n'a pas la dimension complète attendue de {EXPECTED_INPUT_DIM}."
        )
    if vector_dimension != EXPECTED_INPUT_DIM:
        raise ValueError(
            "Les vecteurs W1 fournis à la tête n'ont pas "
            f"{EXPECTED_INPUT_DIM} dimensions."
        )

    kept_dimensions = torch.as_tensor(checkpoint.get("kept_original_dimensions"))
    expected_dimensions = torch.arange(EXPECTED_INPUT_DIM)
    if not torch.equal(kept_dimensions.cpu(), expected_dimensions):
        raise ValueError("W1 contient une sélection de dimensions et semble avoir été pruné.")

    model_state = checkpoint["model_state"]
    for embedding_name in ("input_embeddings.weight", "output_embeddings.weight"):
        if model_state[embedding_name].shape[1] != EXPECTED_INPUT_DIM:
            raise ValueError(
                f"{embedding_name} n'a pas {EXPECTED_INPUT_DIM} colonnes."
            )


def classifier_weight_names(model: SentimentHead) -> list[str]:
    return [
        *(f"hidden_layers.{index}.weight" for index in range(len(model.hidden_layers))),
        "output_layer.weight",
    ]


def cpu_state_dict(model: SentimentHead) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
    }


def apply_masks(
    model: SentimentHead,
    masks: dict[str, torch.Tensor],
) -> None:
    device = next(model.parameters()).device
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in masks:
                parameter.mul_(masks[name].to(device))


def evaluate_model(
    model: SentimentHead,
    validation_vectors: torch.Tensor,
    validation_labels,
    test_vectors: torch.Tensor,
    test_labels,
) -> tuple[dict[str, float], dict[str, float]]:
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
    return validation_metrics, test_metrics


def result_row(
    model_code: str,
    description: str,
    model: SentimentHead,
    validation_metrics: dict[str, float],
    test_metrics: dict[str, float],
    word2vec_inference_parameters: int,
    target_sparsity: float,
    masks: dict[str, torch.Tensor] | None,
) -> dict:
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
    head_parameters = sum(parameter.numel() for parameter in model.parameters())

    row = {
        "word2vec": WORD2VEC_CODE,
        "head": HEAD_CODE,
        "model": model_code,
        "description": description,
        "embedding_dim": EXPECTED_INPUT_DIM,
        "hidden_dims": "-".join(map(str, model.hidden_dims)),
        "target_sparsity": target_sparsity,
        "head_sparsity": 1.0 - active_weights / weight_count,
        "word2vec_training_parameters": 2 * word2vec_inference_parameters,
        "word2vec_inference_parameters": word2vec_inference_parameters,
        "head_parameters": head_parameters,
        "head_total_weights": weight_count,
        "head_active_weights": active_weights,
        "head_bias_parameters": bias_count,
        "head_effective_parameters": active_weights + bias_count,
        "head_dense_macs": weight_count,
        "head_effective_macs": active_weights,
        "stored_inference_parameters": word2vec_inference_parameters + head_parameters,
        "effective_parameters": word2vec_inference_parameters + active_weights + bias_count,
    }
    row.update(
        {f"validation_{name}": value for name, value in validation_metrics.items()}
    )
    row.update({f"test_{name}": value for name, value in test_metrics.items()})
    return row


def save_classifier_checkpoint(
    path: Path,
    model_code: str,
    description: str,
    model: SentimentHead,
    masks: dict[str, torch.Tensor],
    target_sparsity: float,
    validation_metrics: dict[str, float],
    test_metrics: dict[str, float],
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "word2vec": WORD2VEC_CODE,
            "head": HEAD_CODE,
            "model": model_code,
            "method": "MINT",
            "description": description,
            "model_state": cpu_state_dict(model),
            "input_dim": EXPECTED_INPUT_DIM,
            "hidden_dims": model.hidden_dims,
            "masks": {name: mask.cpu() for name, mask in masks.items()},
            "target_sparsity": target_sparsity,
            "groups_per_layer": tuple(args.groups_per_layer),
            "samples_per_class": args.samples_per_class,
            "partition_size": args.partition_size,
            "maximum_pruning_fraction": args.maximum_pruning_fraction,
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True)

    word2vec_path = artifacts / "word2vec_W1.pt"
    word2vec_file_before = (
        word2vec_path.stat().st_size,
        word2vec_path.stat().st_mtime_ns,
    )

    train_vectors = torch.load(
        artifacts / "train_vectors_W1.pt",
        map_location="cpu",
        weights_only=True,
    )
    validation_vectors = torch.load(
        artifacts / "validation_vectors_W1.pt",
        map_location="cpu",
        weights_only=True,
    )
    test_vectors = torch.load(
        artifacts / "test_vectors_W1.pt",
        map_location="cpu",
        weights_only=True,
    )
    train = pd.read_pickle(artifacts / "train_encoded.pkl")
    validation = pd.read_pickle(artifacts / "validation_encoded.pkl")
    test = pd.read_pickle(artifacts / "test_encoded.pkl")
    train_labels = train["sentiment"].to_numpy()
    validation_labels = validation["sentiment"].to_numpy()
    test_labels = test["sentiment"].to_numpy()

    word2vec_checkpoint = torch.load(
        word2vec_path,
        map_location="cpu",
        weights_only=True,
    )
    validate_full_word2vec(word2vec_checkpoint, train_vectors.shape[1])
    word2vec_inference_parameters = word2vec_checkpoint["model_state"][
        "input_embeddings.weight"
    ].numel()
    print(
        "W1 vérifié : Word2Vec complet, non pruné, "
        f"dimension {EXPECTED_INPUT_DIM}."
    )

    classifier_checkpoint = torch.load(
        artifacts / "classifier_W1_H2_M1.pt",
        map_location="cpu",
        weights_only=True,
    )
    if tuple(classifier_checkpoint["hidden_dims"]) != EXPECTED_HIDDEN_DIMS:
        expected_architecture = "→".join(
            map(str, (EXPECTED_INPUT_DIM, *EXPECTED_HIDDEN_DIMS, 1))
        )
        raise ValueError(
            "Le checkpoint W1/H2/M1 n'a pas l'architecture "
            f"{expected_architecture}."
        )
    if classifier_checkpoint["input_dim"] != EXPECTED_INPUT_DIM:
        raise ValueError(
            "Le checkpoint W1/H2/M1 n'accepte pas des entrées de dimension "
            f"{EXPECTED_INPUT_DIM}."
        )

    target_sparsity = args.target_sparsity
    if target_sparsity is None:
        target_sparsity = float(classifier_checkpoint["target_sparsity"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline_model = SentimentHead(EXPECTED_INPUT_DIM, EXPECTED_HIDDEN_DIMS).to(device)
    baseline_model.load_state_dict(classifier_checkpoint["model_state"])

    generator = torch.Generator().manual_seed(args.seed)
    calibration_indices = balanced_calibration_indices(
        train_labels,
        samples_per_class=args.samples_per_class,
        generator=generator,
    )
    calibration_vectors = train_vectors[calibration_indices]
    if len(calibration_vectors) < 3 * args.partition_size:
        raise ValueError("L'ensemble de calibration est trop petit pour les trois partitions MINT.")

    print(
        f"Calibration équilibrée : {len(calibration_vectors)} observations, "
        f"m={args.partition_size}, groupes={tuple(args.groups_per_layer)}."
    )
    print(f"Sparsité cible des blocs : {target_sparsity:.2%}.")

    activations = collect_sentiment_head_activations(
        baseline_model,
        calibration_vectors,
    )
    weight_names = classifier_weight_names(baseline_model)
    named_parameters = dict(baseline_model.named_parameters())
    if len(activations) != len(args.groups_per_layer):
        raise ValueError("Il faut un nombre de groupes pour chacune des trois couches H2.")

    scores_by_layer: dict[str, torch.Tensor] = {}
    groups_by_layer: dict[str, list[torch.Tensor]] = {}
    masks: dict[str, torch.Tensor] = {}
    score_rows: list[dict] = []
    layer_rows: list[dict] = []

    scoring_started = perf_counter()
    for layer_index, ((inputs, outputs), number_of_groups, weight_name) in enumerate(
        zip(activations, args.groups_per_layer, weight_names)
    ):
        weight = named_parameters[weight_name]
        print(
            f"MINT couche {layer_index + 1}/3 — {weight_name}, "
            f"poids={tuple(weight.shape)}, groupes={number_of_groups}...",
            flush=True,
        )
        layer_started = perf_counter()
        layer_generator = torch.Generator().manual_seed(args.seed + layer_index)
        scores, groups = mint_group_scores(
            inputs,
            outputs,
            number_of_groups=number_of_groups,
            partition_size=args.partition_size,
            generator=layer_generator,
        )
        mask = mint_weight_mask(
            scores,
            groups,
            weight_shape=tuple(weight.shape),
            target_sparsity=target_sparsity,
            maximum_pruning_fraction=args.maximum_pruning_fraction,
        )
        scores_by_layer[weight_name] = scores
        groups_by_layer[weight_name] = groups
        masks[weight_name] = mask

        for output_index in range(scores.shape[0]):
            for group_index, group in enumerate(groups):
                block_is_pruned = not bool(mask[output_index, group].all())
                score_rows.append(
                    {
                        "word2vec": WORD2VEC_CODE,
                        "head": HEAD_CODE,
                        "layer": weight_name,
                        "output_index": output_index,
                        "group_index": group_index,
                        "group_start": int(group[0]),
                        "group_stop": int(group[-1]) + 1,
                        "group_size": len(group),
                        "score": float(scores[output_index, group_index]),
                        "pruned": block_is_pruned,
                    }
                )

        pruned_scores = scores[
            torch.tensor(
                [
                    not bool(mask[output_index, group].all())
                    for output_index in range(scores.shape[0])
                    for group in groups
                ],
                dtype=torch.bool,
            ).reshape_as(scores)
        ]
        layer_rows.append(
            {
                "layer": weight_name,
                "output_units": weight.shape[0],
                "input_units": weight.shape[1],
                "groups": len(groups),
                "blocks": scores.numel(),
                "mean_score": float(scores.mean()),
                "minimum_score": float(scores.min()),
                "maximum_score": float(scores.max()),
                "mean_pruned_score": float(pruned_scores.mean()),
                "active_weights": int(mask.sum()),
                "total_weights": mask.numel(),
                "weight_sparsity": 1.0 - int(mask.sum()) / mask.numel(),
                "scoring_seconds": perf_counter() - layer_started,
            }
        )
        print(
            f"  terminé en {layer_rows[-1]['scoring_seconds']:.1f}s — "
            f"score moyen={scores.mean():.4f}, "
            f"poids prunés={layer_rows[-1]['weight_sparsity']:.2%}."
        )

    total_scoring_seconds = perf_counter() - scoring_started
    print(f"Calcul de tous les scores terminé en {total_scoring_seconds:.1f}s.")

    pruned_model = copy.deepcopy(baseline_model)
    apply_masks(pruned_model, masks)
    baseline_validation, baseline_test = evaluate_model(
        baseline_model,
        validation_vectors,
        validation_labels,
        test_vectors,
        test_labels,
    )
    pruned_validation, pruned_test = evaluate_model(
        pruned_model,
        validation_vectors,
        validation_labels,
        test_vectors,
        test_labels,
    )

    retrained_model = copy.deepcopy(pruned_model)
    torch.manual_seed(args.seed)
    retraining_history = train_classifier(
        retrained_model,
        train_vectors,
        train_labels,
        validation_vectors,
        validation_labels,
        epochs=args.retraining_epochs,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        patience=PATIENCE,
        masks=masks,
    )
    apply_masks(retrained_model, masks)
    retrained_validation, retrained_test = evaluate_model(
        retrained_model,
        validation_vectors,
        validation_labels,
        test_vectors,
        test_labels,
    )

    mint_rows = [
        result_row(
            "M1",
            "complet_reference_mint",
            baseline_model,
            baseline_validation,
            baseline_test,
            word2vec_inference_parameters,
            target_sparsity,
            None,
        ),
        result_row(
            "MINT-M2",
            "mint_prune_sans_reentrainement",
            pruned_model,
            pruned_validation,
            pruned_test,
            word2vec_inference_parameters,
            target_sparsity,
            masks,
        ),
        result_row(
            "MINT-M3",
            "mint_prune_reentraine",
            retrained_model,
            retrained_validation,
            retrained_test,
            word2vec_inference_parameters,
            target_sparsity,
            masks,
        ),
    ]

    pd.DataFrame(mint_rows).to_csv(artifacts / "mint_results.csv", index=False)
    pd.DataFrame(score_rows).to_csv(artifacts / "mint_scores.csv", index=False)
    pd.DataFrame(layer_rows).to_csv(
        artifacts / "mint_layer_metrics.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "epoch": range(1, len(retraining_history["train_loss"]) + 1),
            **retraining_history,
        }
    ).to_csv(artifacts / "mint_training_history.csv", index=False)

    torch.save(
        {
            "word2vec": WORD2VEC_CODE,
            "head": HEAD_CODE,
            "method": "MINT",
            "groups_per_layer": tuple(args.groups_per_layer),
            "samples_per_class": args.samples_per_class,
            "partition_size": args.partition_size,
            "maximum_pruning_fraction": args.maximum_pruning_fraction,
            "target_sparsity": target_sparsity,
            "calibration_indices": calibration_indices,
            "scores_by_layer": scores_by_layer,
            "groups_by_layer": groups_by_layer,
            "masks": masks,
            "scoring_seconds": total_scoring_seconds,
        },
        artifacts / "mint_scores_W1_H2.pt",
    )
    save_classifier_checkpoint(
        artifacts / "classifier_W1_H2_MINT_M2.pt",
        "MINT-M2",
        "mint_prune_sans_reentrainement",
        pruned_model,
        masks,
        target_sparsity,
        pruned_validation,
        pruned_test,
        args,
    )
    save_classifier_checkpoint(
        artifacts / "classifier_W1_H2_MINT_M3.pt",
        "MINT-M3",
        "mint_prune_reentraine",
        retrained_model,
        masks,
        target_sparsity,
        retrained_validation,
        retrained_test,
        args,
    )

    classification_results_path = artifacts / "classification_results.csv"
    classification_results = pd.read_csv(classification_results_path)
    mint_model_codes = {"MINT-M2", "MINT-M3"}
    previous_mint_rows = (
        (classification_results["word2vec"] == WORD2VEC_CODE)
        & (classification_results["head"] == HEAD_CODE)
        & classification_results["model"].isin(mint_model_codes)
    )
    classification_results = classification_results.loc[~previous_mint_rows]
    classification_results = pd.concat(
        [classification_results, pd.DataFrame(mint_rows[1:])],
        ignore_index=True,
    )
    classification_results.to_csv(classification_results_path, index=False)

    word2vec_file_after = (
        word2vec_path.stat().st_size,
        word2vec_path.stat().st_mtime_ns,
    )
    if word2vec_file_after != word2vec_file_before:
        raise RuntimeError("Le checkpoint W1 a été modifié pendant le pruning MINT.")

    print(
        "MINT W1/H2 terminé — "
        f"F1 test avant={baseline_test['f1']:.4f}, "
        f"pruné={pruned_test['f1']:.4f}, "
        f"réentraîné={retrained_test['f1']:.4f}, "
        f"sparsité={mint_rows[-1]['head_sparsity']:.2%}."
    )
    print("Le checkpoint Word2Vec W1 est resté inchangé.")


if __name__ == "__main__":
    main()
