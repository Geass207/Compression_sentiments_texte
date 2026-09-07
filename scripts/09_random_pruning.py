"""Témoin de pruning aléatoire des têtes de classification.

Le masque final est un sous-ensemble uniforme, sans remise, des poids des
couches linéaires. Il atteint exactement la sparsité utilisée par OBD et est
appliqué avec le même calendrier itératif afin d'isoler l'effet du critère de
sélection. Les biais et le backbone Word2Vec ne sont jamais prunés.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from scipy.stats import t as student_t

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.classifier import SentimentHead, evaluate_classifier, train_classifier
from src.experiment_config import (
    CLASSIFIER_BATCH_SIZE,
    CLASSIFIER_EPOCHS,
    CLASSIFIER_LEARNING_RATE,
    CLASSIFIER_PATIENCE,
    HEAD_ARCHITECTURES,
    OBD_PRUNING_STEP,
    OBD_RETRAINING_EPOCHS,
)


DEFAULT_SEEDS = (42, 43, 44, 45, 46)
WORD2VEC_CODE = "W1"
METRIC_NAMES = (
    "loss",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "macro_f1",
    "roc_auc",
)
SUMMARY_METRICS = tuple(
    f"{split}_{metric}"
    for split in ("validation", "test")
    for metric in METRIC_NAMES
)
PAIRED_METRICS = ("test_loss", "test_accuracy", "test_f1", "test_roc_auc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Répète un pruning aléatoire global à la sparsité d'OBD, avec le "
            "même calendrier de réentraînement, puis agrège les métriques."
        )
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help="Graines à exécuter (défaut : 42 43 44 45 46).",
    )
    parser.add_argument(
        "--heads",
        choices=tuple(HEAD_ARCHITECTURES),
        nargs="+",
        default=tuple(HEAD_ARCHITECTURES),
        help="Têtes à élaguer (défaut : H1 H2).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/random_pruning"),
        help="Répertoire isolé recevant les résultats.",
    )
    parser.add_argument(
        "--rerun-completed",
        action="store_true",
        help="Recalcule aussi les graines déjà marquées comme terminées.",
    )
    parser.add_argument(
        "--initial-epochs",
        type=int,
        default=CLASSIFIER_EPOCHS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--retraining-epochs",
        type=int,
        default=OBD_RETRAINING_EPOCHS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--pruning-step",
        type=float,
        default=OBD_PRUNING_STEP,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("La liste des graines contient des doublons.")
    if any(seed < 0 for seed in args.seeds):
        raise ValueError("Les graines doivent être positives ou nulles.")
    if args.initial_epochs <= 0 or args.retraining_epochs <= 0:
        raise ValueError("Les nombres d'époques doivent être strictement positifs.")
    if not 0.0 < args.pruning_step <= 1.0:
        raise ValueError("pruning-step doit appartenir à ]0, 1].")


def seed_everything(seed: int) -> None:
    """Initialise toutes les sources d'aléa utilisées par l'expérience."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def cpu_state_dict(model: SentimentHead) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
    }


def cpu_masks(masks: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: mask.detach().cpu() for name, mask in masks.items()}


def apply_masks(
    model: SentimentHead,
    masks: dict[str, torch.Tensor],
) -> None:
    device = next(model.parameters()).device
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in masks:
                parameter.mul_(masks[name].to(device))


def target_sparsity(
    input_dim: int,
    full_dims: tuple[int, ...],
    direct_dims: tuple[int, ...],
) -> float:
    """Reprend la cible définie par le rapport M1/M4 des seuls poids."""
    full = SentimentHead(input_dim, full_dims)
    direct = SentimentHead(input_dim, direct_dims)
    full_weights = sum(
        parameter.numel()
        for name, parameter in full.named_parameters()
        if name.endswith("weight")
    )
    direct_weights = sum(
        parameter.numel()
        for name, parameter in direct.named_parameters()
        if name.endswith("weight")
    )
    return 1.0 - direct_weights / full_weights


def make_weight_masks(model: SentimentHead) -> dict[str, torch.Tensor]:
    return {
        name: torch.ones_like(parameter, dtype=torch.bool, device="cpu")
        for name, parameter in model.named_parameters()
        if name.endswith("weight")
    }


def update_masks_from_flat(
    masks: dict[str, torch.Tensor],
    flat_mask: torch.Tensor,
) -> None:
    start = 0
    for name, mask in masks.items():
        stop = start + mask.numel()
        mask.copy_(flat_mask[start:stop].reshape_as(mask))
        start = stop


def iterative_random_pruning(
    baseline: SentimentHead,
    *,
    target: float,
    pruning_step: float,
    retraining_epochs: int,
    mask_seed: int,
    retraining_seed: int,
    train_vectors: torch.Tensor,
    train_labels: np.ndarray,
    validation_vectors: torch.Tensor,
    validation_labels: np.ndarray,
) -> tuple[
    SentimentHead,
    SentimentHead,
    dict[str, torch.Tensor],
    list[dict],
    list[dict],
]:
    """Applique un masque aléatoire avec le calendrier itératif d'OBD.

    Une permutation globale est tirée une seule fois. Son préfixe définit donc
    un ensemble uniforme de poids à supprimer, indépendamment des valeurs, des
    gradients, de la Hessienne, des activations et des labels.
    """
    current = copy.deepcopy(baseline)
    masks = make_weight_masks(current)
    total_weights = sum(mask.numel() for mask in masks.values())
    target_pruned = round(total_weights * target)
    regular_step = max(1, round(total_weights * pruning_step))

    generator = torch.Generator(device="cpu").manual_seed(mask_seed)
    pruning_order = torch.randperm(total_weights, generator=generator)[:target_pruned]
    flat_mask = torch.ones(total_weights, dtype=torch.bool)
    trajectory: list[dict] = []
    training_rows: list[dict] = []
    pruned_count = 0
    model_just_pruned = copy.deepcopy(current)

    while pruned_count < target_pruned:
        iteration = len(trajectory) + 1
        number_to_prune = min(regular_step, target_pruned - pruned_count)
        selected = pruning_order[pruned_count : pruned_count + number_to_prune]
        flat_mask[selected] = False
        update_masks_from_flat(masks, flat_mask)
        apply_masks(current, masks)
        model_just_pruned = copy.deepcopy(current)

        metrics_before, _ = evaluate_classifier(
            current,
            validation_vectors,
            validation_labels,
            batch_size=CLASSIFIER_BATCH_SIZE,
        )
        seed_everything(retraining_seed + iteration)
        history = train_classifier(
            current,
            train_vectors,
            train_labels,
            validation_vectors,
            validation_labels,
            epochs=retraining_epochs,
            batch_size=CLASSIFIER_BATCH_SIZE,
            lr=CLASSIFIER_LEARNING_RATE,
            patience=None,
            masks=masks,
        )
        apply_masks(current, masks)
        metrics_after, _ = evaluate_classifier(
            current,
            validation_vectors,
            validation_labels,
            batch_size=CLASSIFIER_BATCH_SIZE,
        )
        pruned_count += number_to_prune
        sparsity = pruned_count / total_weights
        trajectory.append(
            {
                "iteration": iteration,
                "newly_pruned_weights": number_to_prune,
                "pruned_weights": pruned_count,
                "total_weights": total_weights,
                "sparsity": sparsity,
                **{f"before_{name}": value for name, value in metrics_before.items()},
                **{f"after_{name}": value for name, value in metrics_after.items()},
            }
        )
        for epoch in range(len(history["train_loss"])):
            training_rows.append(
                {
                    "iteration": iteration,
                    "epoch": epoch + 1,
                    "sparsity": sparsity,
                    **{name: values[epoch] for name, values in history.items()},
                }
            )
        print(
            f"  itération {iteration}: sparsité={sparsity:.2%}, "
            f"val_f1 avant={metrics_before['f1']:.4f}, "
            f"après={metrics_after['f1']:.4f}",
            flush=True,
        )

    return model_just_pruned, current, cpu_masks(masks), trajectory, training_rows


def evaluate_row(
    *,
    seed: int,
    head: str,
    method: str,
    model_code: str,
    description: str,
    model: SentimentHead,
    masks: dict[str, torch.Tensor] | None,
    target: float,
    validation_vectors: torch.Tensor,
    validation_labels: np.ndarray,
    test_vectors: torch.Tensor,
    test_labels: np.ndarray,
    word2vec_parameters: int,
    additional_epoch_budget: int,
    procedure_seconds: float,
    mask_seed: int | None,
) -> dict:
    validation_metrics, _ = evaluate_classifier(
        model,
        validation_vectors,
        validation_labels,
        batch_size=CLASSIFIER_BATCH_SIZE,
    )
    test_metrics, _ = evaluate_classifier(
        model,
        test_vectors,
        test_labels,
        batch_size=CLASSIFIER_BATCH_SIZE,
    )
    total_weights = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.endswith("weight")
    )
    active_weights = (
        sum(int(mask.sum().item()) for mask in masks.values())
        if masks
        else total_weights
    )
    biases = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.endswith("bias")
    )
    head_parameters = sum(parameter.numel() for parameter in model.parameters())
    row = {
        "seed": seed,
        "mask_seed": mask_seed,
        "word2vec": WORD2VEC_CODE,
        "head": head,
        "method": method,
        "model": model_code,
        "description": description,
        "hidden_dims": "-".join(map(str, model.hidden_dims)),
        "target_sparsity": target,
        "head_sparsity": 1.0 - active_weights / total_weights,
        "head_parameters": head_parameters,
        "head_total_weights": total_weights,
        "head_active_weights": active_weights,
        "head_bias_parameters": biases,
        "head_effective_parameters": active_weights + biases,
        "head_effective_macs": active_weights,
        "pipeline_effective_parameters": word2vec_parameters + active_weights + biases,
        "additional_epoch_budget": additional_epoch_budget,
        "procedure_seconds": procedure_seconds,
    }
    row.update(
        {
            f"validation_{name}": value
            for name, value in validation_metrics.items()
        }
    )
    row.update({f"test_{name}": value for name, value in test_metrics.items()})
    return row


def descriptive_statistics(values: np.ndarray) -> dict[str, float | int]:
    count = len(values)
    mean = float(values.mean())
    variance = float(values.var(ddof=1)) if count > 1 else math.nan
    standard_deviation = math.sqrt(variance) if count > 1 else math.nan
    if count > 1:
        critical = float(student_t.ppf(0.975, df=count - 1))
        half_width = critical * standard_deviation / math.sqrt(count)
        ci_low, ci_high = mean - half_width, mean + half_width
    else:
        ci_low = ci_high = math.nan
    return {
        "n": count,
        "mean": mean,
        "std": standard_deviation,
        "variance": variance,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
    }


def save_aggregates(output_dir: Path) -> None:
    metric_paths = sorted(output_dir.glob("seed_*/metrics.csv"))
    if not metric_paths:
        return
    rows = pd.concat((pd.read_csv(path) for path in metric_paths), ignore_index=True)
    rows = rows.sort_values(["seed", "head", "method", "model"])
    rows.to_csv(output_dir / "metrics_by_seed.csv", index=False)

    summary_rows = []
    for keys, group in rows.groupby(
        ["head", "method", "model", "description"], sort=False
    ):
        base = dict(zip(("head", "method", "model", "description"), keys))
        for metric in SUMMARY_METRICS:
            summary_rows.append(
                {
                    **base,
                    "metric": metric,
                    **descriptive_statistics(group[metric].astype(float).to_numpy()),
                }
            )
    pd.DataFrame(summary_rows).to_csv(output_dir / "metrics_summary.csv", index=False)

    comparison_parts = [rows]
    repeated_path = Path("artifacts/repeated_runs/metrics_by_seed.csv")
    if repeated_path.exists():
        repeated = pd.read_csv(repeated_path)
        repeated = repeated[
            repeated["model"].isin(("OBD-M2", "OBD-M3", "MINT-M2", "MINT-M3"))
        ]
        comparison_parts.append(repeated)
    comparison_rows = pd.concat(comparison_parts, ignore_index=True, sort=False)
    requested = (
        ("RAND-M2", "M1"),
        ("RAND-M3", "M1"),
        ("OBD-M2", "RAND-M2"),
        ("OBD-M3", "RAND-M3"),
        ("MINT-M2", "RAND-M2"),
        ("MINT-M3", "RAND-M3"),
    )
    paired_rows = []
    for head, head_rows in comparison_rows.groupby("head", sort=False):
        models = set(head_rows["model"])
        indexed = head_rows.set_index(["seed", "model"])
        for candidate, reference in requested:
            if candidate not in models or reference not in models:
                continue
            candidate_seeds = set(
                head_rows.loc[head_rows["model"] == candidate, "seed"].astype(int)
            )
            reference_seeds = set(
                head_rows.loc[head_rows["model"] == reference, "seed"].astype(int)
            )
            common_seeds = sorted(candidate_seeds & reference_seeds)
            for metric in PAIRED_METRICS:
                differences = np.asarray(
                    [
                        float(indexed.loc[(seed, candidate), metric])
                        - float(indexed.loc[(seed, reference), metric])
                        for seed in common_seeds
                    ],
                    dtype=float,
                )
                paired_rows.append(
                    {
                        "head": head,
                        "candidate": candidate,
                        "reference": reference,
                        "metric": metric,
                        **descriptive_statistics(differences),
                    }
                )
    pd.DataFrame(paired_rows).to_csv(
        output_dir / "paired_differences_summary.csv", index=False
    )


def save_checkpoint(
    path: Path,
    *,
    seed: int,
    mask_seed: int,
    head: str,
    model_code: str,
    model: SentimentHead,
    masks: dict[str, torch.Tensor],
    target: float,
    metrics: dict,
) -> None:
    torch.save(
        {
            "seed": seed,
            "mask_seed": mask_seed,
            "word2vec": WORD2VEC_CODE,
            "head": head,
            "method": "random",
            "model": model_code,
            "model_state": cpu_state_dict(model),
            "input_dim": model.input_dim,
            "hidden_dims": model.hidden_dims,
            "masks": masks,
            "target_sparsity": target,
            "validation_metrics": {
                name.removeprefix("validation_"): value
                for name, value in metrics.items()
                if name.startswith("validation_")
            },
            "test_metrics": {
                name.removeprefix("test_"): value
                for name, value in metrics.items()
                if name.startswith("test_")
            },
        },
        path,
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = Path("artifacts")

    train_vectors = torch.load(
        artifacts / "train_vectors_W1.pt", map_location="cpu", weights_only=True
    )
    validation_vectors = torch.load(
        artifacts / "validation_vectors_W1.pt", map_location="cpu", weights_only=True
    )
    test_vectors = torch.load(
        artifacts / "test_vectors_W1.pt", map_location="cpu", weights_only=True
    )
    train_labels = pd.read_pickle(artifacts / "train_encoded.pkl")[
        "sentiment"
    ].to_numpy()
    validation_labels = pd.read_pickle(artifacts / "validation_encoded.pkl")[
        "sentiment"
    ].to_numpy()
    test_labels = pd.read_pickle(artifacts / "test_encoded.pkl")[
        "sentiment"
    ].to_numpy()
    word2vec = torch.load(
        artifacts / "word2vec_W1.pt", map_location="cpu", weights_only=True
    )
    word2vec_parameters = word2vec["model_state"][
        "input_embeddings.weight"
    ].numel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    configuration = {
        "seeds": args.seeds,
        "heads": args.heads,
        "fixed_word2vec": WORD2VEC_CODE,
        "fixed_data_splits": True,
        "device": str(device),
        "selection": "uniform_without_replacement_over_all_head_weights",
        "mask_seed_offset": 60_000,
        "retraining_seed_offset": 70_000,
        "initial_epochs": args.initial_epochs,
        "classifier_batch_size": CLASSIFIER_BATCH_SIZE,
        "classifier_learning_rate": CLASSIFIER_LEARNING_RATE,
        "classifier_patience_for_m1": CLASSIFIER_PATIENCE,
        "pruning_step": args.pruning_step,
        "retraining_epochs_per_step": args.retraining_epochs,
        "retraining_patience": None,
        "biases_pruned": False,
        "word2vec_pruned": False,
    }
    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as file:
        json.dump(configuration, file, ensure_ascii=False, indent=2)

    for seed_index, seed in enumerate(args.seeds, start=1):
        seed_dir = args.output_dir / f"seed_{seed}"
        completed_path = seed_dir / "completed.json"
        if completed_path.exists() and not args.rerun_completed:
            print(
                f"[{seed_index}/{len(args.seeds)}] Graine {seed} déjà terminée : ignorée.",
                flush=True,
            )
            continue
        seed_dir.mkdir(parents=True, exist_ok=True)
        seed_started = perf_counter()
        seed_rows: list[dict] = []
        seed_trajectory: list[dict] = []
        seed_training_rows: list[dict] = []

        for head in args.heads:
            full_dims = HEAD_ARCHITECTURES[head]["full"]
            direct_dims = HEAD_ARCHITECTURES[head]["direct"]
            target = target_sparsity(train_vectors.shape[1], full_dims, direct_dims)

            print(f"\n=== Graine {seed} — {head}/M1 ===", flush=True)
            seed_everything(seed)
            baseline = SentimentHead(train_vectors.shape[1], full_dims).to(device)
            initial_started = perf_counter()
            initial_history = train_classifier(
                baseline,
                train_vectors,
                train_labels,
                validation_vectors,
                validation_labels,
                epochs=args.initial_epochs,
                batch_size=CLASSIFIER_BATCH_SIZE,
                lr=CLASSIFIER_LEARNING_RATE,
                patience=CLASSIFIER_PATIENCE,
            )
            initial_seconds = perf_counter() - initial_started
            baseline_row = evaluate_row(
                seed=seed,
                head=head,
                method="dense",
                model_code="M1",
                description="tete_dense_complete_reproduite",
                model=baseline,
                masks=None,
                target=target,
                validation_vectors=validation_vectors,
                validation_labels=validation_labels,
                test_vectors=test_vectors,
                test_labels=test_labels,
                word2vec_parameters=word2vec_parameters,
                additional_epoch_budget=0,
                procedure_seconds=initial_seconds,
                mask_seed=None,
            )
            seed_rows.append(baseline_row)

            mask_seed = seed + 60_000
            print(
                f"\n=== Graine {seed} — {head}/pruning aléatoire "
                f"(masque {mask_seed}) ===",
                flush=True,
            )
            random_started = perf_counter()
            random_m2, random_m3, masks, trajectory, training_rows = (
                iterative_random_pruning(
                    baseline,
                    target=target,
                    pruning_step=args.pruning_step,
                    retraining_epochs=args.retraining_epochs,
                    mask_seed=mask_seed,
                    retraining_seed=seed + 70_000,
                    train_vectors=train_vectors,
                    train_labels=train_labels,
                    validation_vectors=validation_vectors,
                    validation_labels=validation_labels,
                )
            )
            random_seconds = perf_counter() - random_started
            iterations = len(trajectory)
            for row in trajectory:
                seed_trajectory.append({"seed": seed, "head": head, **row})
            for row in training_rows:
                seed_training_rows.append({"seed": seed, "head": head, **row})

            for model_code, model, budget, description in (
                (
                    "RAND-M2",
                    random_m2,
                    (iterations - 1) * args.retraining_epochs,
                    "masque_aleatoire_avant_dernier_reentrainement",
                ),
                (
                    "RAND-M3",
                    random_m3,
                    iterations * args.retraining_epochs,
                    "masque_aleatoire_reentraine",
                ),
            ):
                row = evaluate_row(
                    seed=seed,
                    head=head,
                    method="random",
                    model_code=model_code,
                    description=description,
                    model=model,
                    masks=masks,
                    target=target,
                    validation_vectors=validation_vectors,
                    validation_labels=validation_labels,
                    test_vectors=test_vectors,
                    test_labels=test_labels,
                    word2vec_parameters=word2vec_parameters,
                    additional_epoch_budget=budget,
                    procedure_seconds=random_seconds,
                    mask_seed=mask_seed,
                )
                seed_rows.append(row)
                save_checkpoint(
                    seed_dir / f"classifier_W1_{head}_{model_code}.pt",
                    seed=seed,
                    mask_seed=mask_seed,
                    head=head,
                    model_code=model_code,
                    model=model,
                    masks=masks,
                    target=target,
                    metrics=row,
                )

            final_sparsity = seed_rows[-1]["head_sparsity"]
            print(
                f"{head} terminé — sparsité={final_sparsity:.3%}, "
                f"F1 test RAND-M2={seed_rows[-2]['test_f1']:.4f}, "
                f"RAND-M3={seed_rows[-1]['test_f1']:.4f}.",
                flush=True,
            )

        pd.DataFrame(seed_rows).to_csv(seed_dir / "metrics.csv", index=False)
        pd.DataFrame(seed_trajectory).to_csv(
            seed_dir / "pruning_trajectory.csv", index=False
        )
        pd.DataFrame(seed_training_rows).to_csv(
            seed_dir / "training_history.csv", index=False
        )
        elapsed = perf_counter() - seed_started
        with completed_path.open("w", encoding="utf-8") as file:
            json.dump(
                {"seed": seed, "elapsed_seconds": elapsed, "rows": len(seed_rows)},
                file,
                indent=2,
            )
        save_aggregates(args.output_dir)
        print(
            f"\n[{seed_index}/{len(args.seeds)}] Graine {seed} terminée en "
            f"{elapsed / 60:.1f} min.",
            flush=True,
        )

    save_aggregates(args.output_dir)
    print(
        "\nAgrégation terminée : metrics_by_seed.csv, metrics_summary.csv "
        "et paired_differences_summary.csv.",
        flush=True,
    )


if __name__ == "__main__":
    main()
