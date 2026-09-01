"""Répète les expériences de classification sur plusieurs graines.

Le backbone W1, les vecteurs de textes et les partitions restent fixes. Seules
les sources d'aléa propres aux têtes et aux méthodes de pruning sont répétées.
Les sorties sont isolées dans ``artifacts/repeated_runs`` afin de ne jamais
écraser les artefacts du run détaillé utilisé dans le rapport.
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
    MINT_GROUPS_PER_LAYER,
    MINT_MAXIMUM_PRUNING_FRACTION,
    MINT_PARTITION_SIZE,
    MINT_RETRAINING_EPOCHS,
    MINT_SAMPLES_PER_CLASS,
    OBD_HESSIAN_VECTORS,
    OBD_PRUNING_STEP,
    OBD_RETRAINING_EPOCHS,
)
from src.mint import (
    balanced_calibration_indices,
    collect_sentiment_head_activations,
    mint_group_scores,
    mint_weight_mask,
)
from src.pruning import (
    iterative_prune_classifier_weights,
    obd_classifier_weight_scores,
)


DEFAULT_SEEDS = (42, 43, 44, 45, 46)
WORD2VEC_CODE = "W1"
ADJUSTMENT_ALPHA = 0.4
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
RAW_SUMMARY_METRICS = tuple(
    f"{split}_{metric}"
    for split in ("validation", "test")
    for metric in METRIC_NAMES
)
ADJUSTED_SUMMARY_METRICS = (
    "test_loss_adjusted_pipeline_parameters",
    "test_accuracy_adjusted_head_parameters",
    "test_accuracy_adjusted_pipeline_parameters",
    "test_accuracy_adjusted_head_macs",
    "test_f1_adjusted_head_parameters",
    "test_f1_adjusted_pipeline_parameters",
    "test_f1_adjusted_head_macs",
)
SUMMARY_METRICS = RAW_SUMMARY_METRICS + ADJUSTED_SUMMARY_METRICS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Répète M1, OBD-M2/M3, M4 et MINT-M2/M3 sur plusieurs "
            "graines, puis agrège les métriques."
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
        help="Têtes à répéter.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/repeated_runs"),
        help="Répertoire isolé recevant les résultats.",
    )
    parser.add_argument(
        "--skip-mint",
        action="store_true",
        help="Ignore MINT, utile pour un test rapide ou une étude OBD seule.",
    )
    parser.add_argument(
        "--no-dense-controls",
        action="store_true",
        help="N'entraîne pas les témoins denses à budget comparable.",
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
        "--obd-retraining-epochs",
        type=int,
        default=OBD_RETRAINING_EPOCHS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--hessian-vectors",
        type=int,
        default=OBD_HESSIAN_VECTORS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--mint-retraining-epochs",
        type=int,
        default=MINT_RETRAINING_EPOCHS,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("La liste des graines contient des doublons.")
    if any(seed < 0 for seed in args.seeds):
        raise ValueError("Les graines doivent être positives ou nulles.")
    for name in (
        "initial_epochs",
        "obd_retraining_epochs",
        "hessian_vectors",
        "mint_retraining_epochs",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} doit être strictement positif.")


def seed_everything(seed: int) -> None:
    """Initialise les générateurs utilisés par Python, NumPy et PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


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


def weight_names(model: SentimentHead) -> list[str]:
    return [
        *(f"hidden_layers.{index}.weight" for index in range(len(model.hidden_layers))),
        "output_layer.weight",
    ]


def target_sparsity(
    input_dim: int,
    full_dims: tuple[int, ...],
    direct_dims: tuple[int, ...],
) -> float:
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


def pruning_iterations(total_weights: int, sparsity: float) -> int:
    target_pruned = round(total_weights * sparsity)
    regular_step = max(1, round(total_weights * OBD_PRUNING_STEP))
    return math.ceil(target_pruned / regular_step)


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
    additional_epoch_budget: int = 0,
    procedure_seconds: float | None = None,
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
        "pipeline_effective_parameters": (
            word2vec_parameters + active_weights + biases
        ),
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


def train_dense_controls(
    baseline: SentimentHead,
    *,
    seed: int,
    iterations: int,
    train_vectors: torch.Tensor,
    train_labels: np.ndarray,
    validation_vectors: torch.Tensor,
    validation_labels: np.ndarray,
    epochs_per_iteration: int,
) -> tuple[SentimentHead, SentimentHead]:
    """Poursuit M1 avec le même budget maximal que les phases OBD."""
    control = copy.deepcopy(baseline)
    before_last = copy.deepcopy(control)
    for iteration in range(iterations):
        seed_everything(seed + 20_000 + iteration)
        train_classifier(
            control,
            train_vectors,
            train_labels,
            validation_vectors,
            validation_labels,
            epochs=epochs_per_iteration,
            batch_size=CLASSIFIER_BATCH_SIZE,
            lr=CLASSIFIER_LEARNING_RATE,
            patience=None,
        )
        if iteration == iterations - 2:
            before_last = copy.deepcopy(control)
    if iterations == 1:
        before_last = copy.deepcopy(baseline)
    return before_last, control


def mint_models(
    baseline: SentimentHead,
    *,
    seed: int,
    target: float,
    train_vectors: torch.Tensor,
    train_labels: np.ndarray,
    validation_vectors: torch.Tensor,
    validation_labels: np.ndarray,
    retraining_epochs: int,
) -> tuple[
    SentimentHead,
    SentimentHead,
    dict[str, torch.Tensor],
    list[dict],
    float,
]:
    calibration_generator = torch.Generator().manual_seed(seed + 30_000)
    calibration_indices = balanced_calibration_indices(
        train_labels,
        samples_per_class=MINT_SAMPLES_PER_CLASS,
        generator=calibration_generator,
    )
    activations = collect_sentiment_head_activations(
        baseline,
        train_vectors[calibration_indices],
    )
    names = weight_names(baseline)
    parameters = dict(baseline.named_parameters())
    masks: dict[str, torch.Tensor] = {}
    layer_rows = []

    scoring_started = perf_counter()
    for layer_index, ((inputs, outputs), groups_count, name) in enumerate(
        zip(activations, MINT_GROUPS_PER_LAYER, names)
    ):
        layer_started = perf_counter()
        generator = torch.Generator().manual_seed(seed + 40_000 + layer_index)
        scores, groups = mint_group_scores(
            inputs,
            outputs,
            number_of_groups=groups_count,
            partition_size=MINT_PARTITION_SIZE,
            generator=generator,
        )
        mask = mint_weight_mask(
            scores,
            groups,
            weight_shape=tuple(parameters[name].shape),
            target_sparsity=target,
            maximum_pruning_fraction=MINT_MAXIMUM_PRUNING_FRACTION,
        )
        masks[name] = mask
        layer_rows.append(
            {
                "seed": seed,
                "head": "H2",
                "layer": name,
                "mean_score": float(scores.mean()),
                "score_variance": float(scores.var(unbiased=True)),
                "active_weights": int(mask.sum()),
                "total_weights": mask.numel(),
                "weight_sparsity": 1.0 - int(mask.sum()) / mask.numel(),
                "scoring_seconds": perf_counter() - layer_started,
            }
        )
    scoring_seconds = perf_counter() - scoring_started

    pruned = copy.deepcopy(baseline)
    apply_masks(pruned, masks)
    retrained = copy.deepcopy(pruned)
    seed_everything(seed + 50_000)
    train_classifier(
        retrained,
        train_vectors,
        train_labels,
        validation_vectors,
        validation_labels,
        epochs=retraining_epochs,
        batch_size=CLASSIFIER_BATCH_SIZE,
        lr=CLASSIFIER_LEARNING_RATE,
        patience=CLASSIFIER_PATIENCE,
        masks=masks,
    )
    apply_masks(retrained, masks)
    return pruned, retrained, cpu_masks(masks), layer_rows, scoring_seconds


def dense_mint_control(
    baseline: SentimentHead,
    *,
    seed: int,
    train_vectors: torch.Tensor,
    train_labels: np.ndarray,
    validation_vectors: torch.Tensor,
    validation_labels: np.ndarray,
    epochs: int,
) -> SentimentHead:
    control = copy.deepcopy(baseline)
    seed_everything(seed + 50_000)
    train_classifier(
        control,
        train_vectors,
        train_labels,
        validation_vectors,
        validation_labels,
        epochs=epochs,
        batch_size=CLASSIFIER_BATCH_SIZE,
        lr=CLASSIFIER_LEARNING_RATE,
        patience=CLASSIFIER_PATIENCE,
    )
    return control


def summarize_metrics(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_columns = ["head", "method", "model", "description"]
    wide_rows = []
    long_rows = []
    for keys, group in rows.groupby(group_columns, sort=False):
        wide = dict(zip(group_columns, keys))
        wide["n"] = len(group)
        for metric in SUMMARY_METRICS:
            values = group[metric].astype(float).to_numpy()
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
            statistics = {
                "mean": mean,
                "std": standard_deviation,
                "variance": variance,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
            }
            for statistic, value in statistics.items():
                wide[f"{metric}_{statistic}"] = value
            long_rows.append(
                {
                    **dict(zip(group_columns, keys)),
                    "metric": metric,
                    "n": count,
                    **statistics,
                }
            )
        wide_rows.append(wide)
    return pd.DataFrame(wide_rows), pd.DataFrame(long_rows)


def paired_comparisons(rows: pd.DataFrame) -> pd.DataFrame:
    comparisons = []
    requested = [
        ("OBD-M2", "M1"),
        ("OBD-M3", "M1"),
        ("M4", "M1"),
        ("OBD-M2", "OBD-M3"),
        ("OBD-M2", "Dense-control-M2"),
        ("OBD-M3", "Dense-control-M3"),
        ("MINT-M2", "M1"),
        ("MINT-M3", "M1"),
        ("MINT-M3", "OBD-M2"),
        ("MINT-M3", "Dense-control-MINT"),
    ]
    for head, head_rows in rows.groupby("head", sort=False):
        indexed = head_rows.set_index(["seed", "model"])
        models = set(head_rows["model"])
        for candidate, reference in requested:
            if candidate not in models or reference not in models:
                continue
            common_seeds = sorted(
                set(head_rows.loc[head_rows["model"] == candidate, "seed"])
                & set(head_rows.loc[head_rows["model"] == reference, "seed"])
            )
            for metric in SUMMARY_METRICS:
                differences = np.asarray(
                    [
                        float(indexed.loc[(seed, candidate), metric])
                        - float(indexed.loc[(seed, reference), metric])
                        for seed in common_seeds
                    ]
                )
                count = len(differences)
                mean = float(differences.mean())
                variance = (
                    float(differences.var(ddof=1)) if count > 1 else math.nan
                )
                standard_deviation = (
                    math.sqrt(variance) if count > 1 else math.nan
                )
                if count > 1:
                    critical = float(student_t.ppf(0.975, df=count - 1))
                    half_width = critical * standard_deviation / math.sqrt(count)
                    ci_low, ci_high = mean - half_width, mean + half_width
                else:
                    ci_low = ci_high = math.nan
                comparisons.append(
                    {
                        "head": head,
                        "candidate": candidate,
                        "reference": reference,
                        "metric": metric,
                        "n_pairs": count,
                        "mean_difference": mean,
                        "std_difference": standard_deviation,
                        "variance_difference": variance,
                        "ci95_low": ci_low,
                        "ci95_high": ci_high,
                    }
                )
    return pd.DataFrame(comparisons)


def add_adjusted_metrics(rows: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les critères ajustés employés dans les figures du rapport."""
    rows = rows.copy()
    cost_columns = {
        "head_parameters": "head_effective_parameters",
        "pipeline_parameters": "pipeline_effective_parameters",
        "head_macs": "head_effective_macs",
    }
    for (_, _), indices in rows.groupby(["seed", "head"]).groups.items():
        group = rows.loc[indices]
        reference = group.loc[group["model"] == "M1"]
        if len(reference) != 1:
            raise ValueError("Chaque paire graine/tête doit contenir exactement un M1.")
        reference = reference.iloc[0]
        for suffix, cost_column in cost_columns.items():
            ratio = rows.loc[indices, cost_column] / float(reference[cost_column])
            rows.loc[
                indices, f"test_accuracy_adjusted_{suffix}"
            ] = rows.loc[indices, "test_accuracy"] - ADJUSTMENT_ALPHA * ratio
            rows.loc[
                indices, f"test_f1_adjusted_{suffix}"
            ] = rows.loc[indices, "test_f1"] - ADJUSTMENT_ALPHA * ratio
        pipeline_ratio = (
            rows.loc[indices, "pipeline_effective_parameters"]
            / float(reference["pipeline_effective_parameters"])
        )
        rows.loc[
            indices, "test_loss_adjusted_pipeline_parameters"
        ] = rows.loc[indices, "test_loss"] + ADJUSTMENT_ALPHA * pipeline_ratio
    return rows


def save_aggregates(output_dir: Path) -> None:
    metric_paths = sorted(output_dir.glob("seed_*/metrics.csv"))
    if not metric_paths:
        return
    rows = pd.concat((pd.read_csv(path) for path in metric_paths), ignore_index=True)
    rows = rows.sort_values(["seed", "head", "method", "model"])
    rows = add_adjusted_metrics(rows)
    rows.to_csv(output_dir / "metrics_by_seed.csv", index=False)
    summary_wide, summary_long = summarize_metrics(rows)
    summary_wide.to_csv(output_dir / "metrics_summary.csv", index=False)
    summary_long.to_csv(output_dir / "metrics_summary_long.csv", index=False)
    paired_comparisons(rows).to_csv(
        output_dir / "paired_differences_summary.csv",
        index=False,
    )


def run_seed(
    seed: int,
    args: argparse.Namespace,
    *,
    train_vectors: torch.Tensor,
    train_labels: np.ndarray,
    validation_vectors: torch.Tensor,
    validation_labels: np.ndarray,
    test_vectors: torch.Tensor,
    test_labels: np.ndarray,
    word2vec_parameters: int,
    device: torch.device,
) -> tuple[list[dict], list[dict], list[dict]]:
    rows: list[dict] = []
    pruning_rows: list[dict] = []
    mint_layer_rows: list[dict] = []
    input_dim = train_vectors.shape[1]

    for head in args.heads:
        full_dims = HEAD_ARCHITECTURES[head]["full"]
        direct_dims = HEAD_ARCHITECTURES[head]["direct"]
        target = target_sparsity(input_dim, full_dims, direct_dims)

        print(f"\n=== Graine {seed} — {head}/M1 ===", flush=True)
        seed_everything(seed)
        baseline = SentimentHead(input_dim, full_dims).to(device)
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
        rows.append(
            evaluate_row(
                seed=seed,
                head=head,
                method="dense",
                model_code="M1",
                description="tete_dense_complete",
                model=baseline,
                masks=None,
                target=target,
                validation_vectors=validation_vectors,
                validation_labels=validation_labels,
                test_vectors=test_vectors,
                test_labels=test_labels,
                word2vec_parameters=word2vec_parameters,
                procedure_seconds=initial_seconds,
            )
        )

        total_weights = sum(
            parameter.numel()
            for name, parameter in baseline.named_parameters()
            if name.endswith("weight")
        )
        iterations = pruning_iterations(total_weights, target)

        if not args.no_dense_controls:
            print(
                f"\n=== Graine {seed} — {head}/témoins denses OBD ===",
                flush=True,
            )
            control_started = perf_counter()
            control_m2, control_m3 = train_dense_controls(
                baseline,
                seed=seed,
                iterations=iterations,
                train_vectors=train_vectors,
                train_labels=train_labels,
                validation_vectors=validation_vectors,
                validation_labels=validation_labels,
                epochs_per_iteration=args.obd_retraining_epochs,
            )
            control_seconds = perf_counter() - control_started
            for code, model, budget in (
                (
                    "Dense-control-M2",
                    control_m2,
                    (iterations - 1) * args.obd_retraining_epochs,
                ),
                (
                    "Dense-control-M3",
                    control_m3,
                    iterations * args.obd_retraining_epochs,
                ),
            ):
                rows.append(
                    evaluate_row(
                        seed=seed,
                        head=head,
                        method="dense_control",
                        model_code=code,
                        description="m1_prolonge_budget_obd",
                        model=model,
                        masks=None,
                        target=target,
                        validation_vectors=validation_vectors,
                        validation_labels=validation_labels,
                        test_vectors=test_vectors,
                        test_labels=test_labels,
                        word2vec_parameters=word2vec_parameters,
                        additional_epoch_budget=budget,
                        procedure_seconds=control_seconds,
                    )
                )

        print(f"\n=== Graine {seed} — {head}/OBD ===", flush=True)
        seed_everything(seed + 10_000)
        obd_started = perf_counter()
        m2, m3, masks, pruning_history = iterative_prune_classifier_weights(
            copy.deepcopy(baseline),
            train_vectors,
            train_labels,
            validation_vectors,
            validation_labels,
            target_sparsity=target,
            score_function=obd_classifier_weight_scores,
            pruning_step=OBD_PRUNING_STEP,
            retraining_epochs=args.obd_retraining_epochs,
            batch_size=CLASSIFIER_BATCH_SIZE,
            hessian_vectors=args.hessian_vectors,
            lr=CLASSIFIER_LEARNING_RATE,
            patience=None,
        )
        obd_seconds = perf_counter() - obd_started
        for iteration, step in enumerate(pruning_history, start=1):
            pruning_rows.append(
                {
                    "seed": seed,
                    "head": head,
                    "iteration": iteration,
                    "sparsity": step["sparsity"],
                    "scoring_seconds": step["scoring_seconds"],
                    **{
                        f"before_{name}": value
                        for name, value in step[
                            "validation_metrics_before_retraining"
                        ].items()
                    },
                    **{
                        f"after_{name}": value
                        for name, value in step[
                            "validation_metrics_after_retraining"
                        ].items()
                    },
                }
            )
        for code, model, budget, description in (
            (
                "OBD-M2",
                m2,
                (iterations - 1) * args.obd_retraining_epochs,
                "obd_prune_avant_dernier_reentrainement",
            ),
            (
                "OBD-M3",
                m3,
                iterations * args.obd_retraining_epochs,
                "obd_prune_reentraine",
            ),
        ):
            rows.append(
                evaluate_row(
                    seed=seed,
                    head=head,
                    method="OBD",
                    model_code=code,
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
                    procedure_seconds=obd_seconds,
                )
            )

        print(f"\n=== Graine {seed} — {head}/M4 ===", flush=True)
        seed_everything(seed)
        m4 = SentimentHead(input_dim, direct_dims).to(device)
        m4_started = perf_counter()
        train_classifier(
            m4,
            train_vectors,
            train_labels,
            validation_vectors,
            validation_labels,
            epochs=args.initial_epochs,
            batch_size=CLASSIFIER_BATCH_SIZE,
            lr=CLASSIFIER_LEARNING_RATE,
            patience=CLASSIFIER_PATIENCE,
        )
        m4_seconds = perf_counter() - m4_started
        rows.append(
            evaluate_row(
                seed=seed,
                head=head,
                method="dense",
                model_code="M4",
                description="architecture_dense_reduite",
                model=m4,
                masks=None,
                target=target,
                validation_vectors=validation_vectors,
                validation_labels=validation_labels,
                test_vectors=test_vectors,
                test_labels=test_labels,
                word2vec_parameters=word2vec_parameters,
                procedure_seconds=m4_seconds,
            )
        )

        if head == "H2" and not args.skip_mint:
            print(f"\n=== Graine {seed} — H2/MINT ===", flush=True)
            mint_started = perf_counter()
            mint_m2, mint_m3, mint_masks, layers, scoring_seconds = mint_models(
                baseline,
                seed=seed,
                target=target,
                train_vectors=train_vectors,
                train_labels=train_labels,
                validation_vectors=validation_vectors,
                validation_labels=validation_labels,
                retraining_epochs=args.mint_retraining_epochs,
            )
            mint_seconds = perf_counter() - mint_started
            mint_layer_rows.extend(layers)
            for code, model, budget, description in (
                ("MINT-M2", mint_m2, 0, "mint_prune_sans_reentrainement"),
                (
                    "MINT-M3",
                    mint_m3,
                    args.mint_retraining_epochs,
                    "mint_prune_reentraine",
                ),
            ):
                row = evaluate_row(
                    seed=seed,
                    head=head,
                    method="MINT",
                    model_code=code,
                    description=description,
                    model=model,
                    masks=mint_masks,
                    target=target,
                    validation_vectors=validation_vectors,
                    validation_labels=validation_labels,
                    test_vectors=test_vectors,
                    test_labels=test_labels,
                    word2vec_parameters=word2vec_parameters,
                    additional_epoch_budget=budget,
                    procedure_seconds=mint_seconds,
                )
                row["scoring_seconds"] = scoring_seconds
                rows.append(row)

            if not args.no_dense_controls:
                print(
                    f"\n=== Graine {seed} — H2/témoin dense MINT ===",
                    flush=True,
                )
                control_started = perf_counter()
                control = dense_mint_control(
                    baseline,
                    seed=seed,
                    train_vectors=train_vectors,
                    train_labels=train_labels,
                    validation_vectors=validation_vectors,
                    validation_labels=validation_labels,
                    epochs=args.mint_retraining_epochs,
                )
                rows.append(
                    evaluate_row(
                        seed=seed,
                        head=head,
                        method="dense_control",
                        model_code="Dense-control-MINT",
                        description="m1_prolonge_budget_mint",
                        model=control,
                        masks=None,
                        target=target,
                        validation_vectors=validation_vectors,
                        validation_labels=validation_labels,
                        test_vectors=test_vectors,
                        test_labels=test_labels,
                        word2vec_parameters=word2vec_parameters,
                        additional_epoch_budget=args.mint_retraining_epochs,
                        procedure_seconds=perf_counter() - control_started,
                    )
                )

        print(
            f"{head} terminé : {len(initial_history['train_loss'])} "
            "époques observées pour M1.",
            flush=True,
        )

    return rows, pruning_rows, mint_layer_rows


def main() -> None:
    args = parse_args()
    validate_args(args)
    artifacts = Path("artifacts")
    args.output_dir.mkdir(parents=True, exist_ok=True)

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
        "initial_epochs": args.initial_epochs,
        "classifier_batch_size": CLASSIFIER_BATCH_SIZE,
        "classifier_learning_rate": CLASSIFIER_LEARNING_RATE,
        "classifier_patience": CLASSIFIER_PATIENCE,
        "obd_hessian_vectors": args.hessian_vectors,
        "obd_pruning_step": OBD_PRUNING_STEP,
        "obd_retraining_epochs": args.obd_retraining_epochs,
        "mint_enabled": not args.skip_mint,
        "mint_groups_per_layer": MINT_GROUPS_PER_LAYER,
        "mint_samples_per_class": MINT_SAMPLES_PER_CLASS,
        "mint_partition_size": MINT_PARTITION_SIZE,
        "mint_maximum_pruning_fraction": MINT_MAXIMUM_PRUNING_FRACTION,
        "mint_retraining_epochs": args.mint_retraining_epochs,
        "dense_controls": not args.no_dense_controls,
        "adjustment_alpha": ADJUSTMENT_ALPHA,
    }
    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as file:
        json.dump(configuration, file, ensure_ascii=False, indent=2)

    for index, seed in enumerate(args.seeds, start=1):
        seed_dir = args.output_dir / f"seed_{seed}"
        completed_path = seed_dir / "completed.json"
        if completed_path.exists() and not args.rerun_completed:
            print(
                f"[{index}/{len(args.seeds)}] Graine {seed} déjà terminée : ignorée.",
                flush=True,
            )
            continue
        seed_dir.mkdir(parents=True, exist_ok=True)
        started = perf_counter()
        rows, pruning_rows, mint_rows = run_seed(
            seed,
            args,
            train_vectors=train_vectors,
            train_labels=train_labels,
            validation_vectors=validation_vectors,
            validation_labels=validation_labels,
            test_vectors=test_vectors,
            test_labels=test_labels,
            word2vec_parameters=word2vec_parameters,
            device=device,
        )
        pd.DataFrame(rows).to_csv(seed_dir / "metrics.csv", index=False)
        pd.DataFrame(pruning_rows).to_csv(
            seed_dir / "obd_pruning_trajectory.csv", index=False
        )
        if mint_rows:
            pd.DataFrame(mint_rows).to_csv(
                seed_dir / "mint_layer_metrics.csv", index=False
            )
        elapsed = perf_counter() - started
        with completed_path.open("w", encoding="utf-8") as file:
            json.dump(
                {"seed": seed, "elapsed_seconds": elapsed, "rows": len(rows)},
                file,
                indent=2,
            )
        save_aggregates(args.output_dir)
        print(
            f"\n[{index}/{len(args.seeds)}] Graine {seed} terminée en "
            f"{elapsed / 60:.1f} min.",
            flush=True,
        )

    save_aggregates(args.output_dir)
    print(
        "\nAgrégation terminée : metrics_by_seed.csv, metrics_summary.csv, "
        "metrics_summary_long.csv et paired_differences_summary.csv.",
        flush=True,
    )


if __name__ == "__main__":
    main()
