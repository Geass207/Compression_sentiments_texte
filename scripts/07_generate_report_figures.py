"""Génère les figures de synthèse annoncées dans le rapport.

Les figures purement déduites des dimensions et des masques sont rangées
dans ``figures/theoriques``. Celles qui utilisent une loss, une mesure de
qualité, un temps de calcul ou une taille sérialisée sont rangées dans
``figures/experimentales``.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.classifier import SentimentHead
from src.word2vec import CBOW, text_vectors


ARTIFACTS = Path("artifacts")
THEORETICAL_DIR = Path("figures") / "theoriques"
EXPERIMENTAL_DIR = Path("figures") / "experimentales"
LATENCY_SAMPLE_COUNT = 2_048
LATENCY_BATCH_SIZE = 512
LATENCY_WARMUPS = 2
LATENCY_REPEATS = 7
ALPHAS = np.linspace(0.0, 1.0, 101)

MODEL_ORDER = ["M1", "M2", "M3", "M4", "MINT-M2", "MINT-M3"]
MODEL_COLORS = {
    "M1": "#4c78a8",
    "M2": "#f58518",
    "M3": "#54a24b",
    "M4": "#e45756",
    "MINT-M2": "#b279a2",
    "MINT-M3": "#72b7b2",
}
MODEL_LABELS = {
    "M1": "M1 dense",
    "M2": "M2 OBD",
    "M3": "M3 OBD réentraîné",
    "M4": "M4 compact",
    "MINT-M2": "M2 MINT",
    "MINT-M3": "M3 MINT réentraîné",
}


def save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def ordered_models(frame: pd.DataFrame) -> pd.DataFrame:
    order = {model: index for index, model in enumerate(MODEL_ORDER)}
    return frame.assign(_order=frame["model"].map(order)).sort_values("_order")


def pareto_indices(costs: np.ndarray, qualities: np.ndarray) -> list[int]:
    """Indices non dominés lorsque le coût est minimisé et la qualité maximisée."""
    indices = []
    for index, (cost, quality) in enumerate(zip(costs, qualities)):
        dominated = np.any(
            (costs <= cost)
            & (qualities >= quality)
            & ((costs < cost) | (qualities > quality))
        )
        if not dominated:
            indices.append(index)
    return sorted(indices, key=lambda index: costs[index])


def plot_dense_parameter_distribution(results: pd.DataFrame) -> None:
    baselines = ordered_models(results[results["model"] == "M1"])
    labels = [f"{row.head}/{row.model}" for row in baselines.itertuples()]
    backbone = baselines["word2vec_inference_parameters"].to_numpy()
    heads = baselines["head_parameters"].to_numpy()
    totals = backbone + heads

    figure, axis = plt.subplots(figsize=(8.2, 5.2))
    positions = np.arange(len(labels))
    axis.bar(positions, backbone, label="Backbone (inference)", color="#9ecae9")
    axis.bar(positions, heads, bottom=backbone, label="Tête", color="#fdae6b")
    for position, backbone_count, head_count, total in zip(
        positions, backbone, heads, totals
    ):
        axis.text(
            position,
            backbone_count + head_count / 2,
            f"{head_count / total:.1%}\nde la tête",
            ha="center",
            va="center",
            fontsize=9,
        )
        axis.text(
            position,
            total,
            f"{int(total):,}".replace(",", " "),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Nombre de paramètres denses")
    axis.set_title("Répartition paramétrique du pipeline avant pruning")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    save_figure(figure, THEORETICAL_DIR / "repartition_parametres_denses.png")


def plot_theoretical_sparsity_effect(results: pd.DataFrame) -> None:
    sparsities = np.linspace(0.0, 0.9, 91)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey="col")

    for row_index, head in enumerate(("H1", "H2")):
        baseline = results[(results["head"] == head) & (results["model"] == "M1")].iloc[0]
        weights = float(baseline["head_total_weights"])
        biases = float(baseline["head_bias_parameters"])
        backbone = float(baseline["word2vec_inference_parameters"])
        active = weights * (1.0 - sparsities)

        head_ratio = (active + biases) / (weights + biases)
        pipeline_ratio = (backbone + active + biases) / (
            backbone + weights + biases
        )
        axes[row_index, 0].plot(
            sparsities * 100, head_ratio * 100, label="Tête seule", linewidth=2
        )
        axes[row_index, 0].plot(
            sparsities * 100,
            pipeline_ratio * 100,
            label="Pipeline complet",
            linewidth=2,
        )
        axes[row_index, 1].plot(
            sparsities * 100,
            (1.0 - sparsities) * 100,
            color="#54a24b",
            linewidth=2,
        )
        axes[row_index, 0].set_ylabel(f"{head} — coût restant (%)")
        axes[row_index, 0].grid(alpha=0.25)
        axes[row_index, 1].grid(alpha=0.25)

    axes[0, 0].set_title("Paramètres effectifs idéaux")
    axes[0, 1].set_title("MAC effectives idéales de la tête")
    axes[-1, 0].set_xlabel("Sparsité des poids (%)")
    axes[-1, 1].set_xlabel("Sparsité des poids (%)")
    axes[0, 0].legend()
    figure.suptitle(
        "Effet théorique de la sparsification, indépendamment de la qualité",
        fontsize=14,
    )
    figure.tight_layout()
    save_figure(figure, THEORETICAL_DIR / "effet_theorique_sparsification.png")


def add_resource_ratios(results: pd.DataFrame) -> pd.DataFrame:
    enriched = results.copy()
    for head in enriched["head"].unique():
        selection = enriched["head"] == head
        baseline = enriched[selection & (enriched["model"] == "M1")].iloc[0]
        enriched.loc[selection, "ratio_parametres_tete"] = (
            enriched.loc[selection, "head_effective_parameters"]
            / baseline["head_parameters"]
        )
        enriched.loc[selection, "ratio_parametres_pipeline"] = (
            enriched.loc[selection, "effective_parameters"]
            / baseline["effective_parameters"]
        )
        enriched.loc[selection, "ratio_mac_tete"] = (
            enriched.loc[selection, "head_effective_macs"]
            / baseline["head_dense_macs"]
        )
    return enriched


def plot_quality_resource(
    results: pd.DataFrame,
    ratio_column: str,
    resource_label: str,
    filename: str,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), sharex="col")
    for row_index, head in enumerate(("H1", "H2")):
        subset = ordered_models(results[results["head"] == head])
        costs = subset[ratio_column].to_numpy(dtype=float)
        for column_index, (metric, metric_label) in enumerate(
            (("test_f1", "F1 test"), ("test_accuracy", "Accuracy test"))
        ):
            axis = axes[row_index, column_index]
            qualities = subset[metric].to_numpy(dtype=float)
            for model, cost, quality in zip(subset["model"], costs, qualities):
                axis.scatter(
                    cost,
                    quality,
                    s=58,
                    color=MODEL_COLORS[model],
                    label=MODEL_LABELS[model],
                    zorder=3,
                )
            frontier = pareto_indices(costs, qualities)
            axis.plot(
                costs[frontier],
                qualities[frontier],
                color="black",
                linestyle="--",
                linewidth=1.2,
                label="Frontière de Pareto",
            )
            axis.set_ylabel(f"{head} — {metric_label}")
            axis.grid(alpha=0.25)
            if row_index == 1:
                axis.set_xlabel(resource_label)
    handles, labels = axes[1, 1].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    figure.legend(
        unique.values(), unique.keys(), loc="lower center", ncol=4, fontsize=9
    )
    figure.suptitle(
        "Qualité test mesurée en fonction du coût effectif estimé",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0.10, 1, 0.96))
    save_figure(figure, EXPERIMENTAL_DIR / filename)


def plot_adjusted_quality(results: pd.DataFrame) -> None:
    resources = (
        (
            "ratio_parametres_tete",
            "mémoire de la tête",
            "qualite_ajustee_memoire_tete.png",
        ),
        (
            "ratio_parametres_pipeline",
            "mémoire du pipeline",
            "qualite_ajustee_memoire_pipeline.png",
        ),
        ("ratio_mac_tete", "MAC de la tête", "qualite_ajustee_calcul_tete.png"),
    )
    for ratio_column, resource_label, filename in resources:
        figure, axes = plt.subplots(1, 2, figsize=(13, 5.3), sharey=True)
        for axis, head in zip(axes, ("H1", "H2")):
            subset = ordered_models(results[results["head"] == head])
            for row in subset.itertuples():
                adjusted = row.test_f1 - ALPHAS * getattr(row, ratio_column)
                axis.plot(
                    ALPHAS,
                    adjusted,
                    label=MODEL_LABELS[row.model],
                    color=MODEL_COLORS[row.model],
                    linewidth=1.8,
                )
            axis.set_title(head)
            axis.set_xlabel(r"Pénalité $\alpha$")
            axis.grid(alpha=0.25)
        axes[0].set_ylabel(r"Qualité ajustée : $F1 - \alpha\,C_{relatif}$")
        handles, labels = axes[1].get_legend_handles_labels()
        figure.legend(handles, labels, loc="lower center", ncol=3, fontsize=9)
        figure.suptitle(f"Sensibilité au compromis qualité–{resource_label}")
        figure.tight_layout(rect=(0, 0.13, 1, 0.95))
        save_figure(figure, EXPERIMENTAL_DIR / filename)


def plot_obd_score_distributions() -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.3))
    for axis, head in zip(axes, ("H1", "H2")):
        history = torch.load(
            ARTIFACTS / f"classifier_pruning_history_W1_{head}.pt",
            map_location="cpu",
            weights_only=True,
        )
        initial_scores = history[0].get("initial_scores_by_layer")
        if not initial_scores:
            raise RuntimeError(
                f"L'historique OBD {head} ne contient pas les scores initiaux."
            )
        for layer_name, scores in initial_scores.items():
            values = scores.detach().cpu().numpy().reshape(-1)
            transformed = np.log10(np.maximum(values, 1e-18))
            axis.hist(
                transformed,
                bins=70,
                density=True,
                histtype="step",
                linewidth=1.5,
                label=layer_name.replace("hidden_layers.", "cachée ").replace(".weight", "").replace("output_layer", "sortie"),
            )
        axis.set_title(head)
        axis.set_xlabel(r"$\log_{10}(S_i)$")
        axis.set_ylabel("Densité")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle("Distribution expérimentale des saliences OBD initiales")
    figure.tight_layout()
    save_figure(figure, EXPERIMENTAL_DIR / "distribution_scores_obd.png")


def plot_scoring_costs() -> None:
    obd = pd.read_csv(ARTIFACTS / "classifier_pruning_metrics.csv")
    mint = pd.read_csv(ARTIFACTS / "mint_layer_metrics.csv")
    if "scoring_seconds" not in obd:
        raise RuntimeError("Les temps de scoring OBD sont absents du fichier de mesures.")
    obd_unique = obd.drop_duplicates(["head", "iteration"])
    labels = ["OBD H1", "OBD H2", "MINT H2"]
    seconds = [
        obd_unique.loc[obd_unique["head"] == "H1", "scoring_seconds"].sum(),
        obd_unique.loc[obd_unique["head"] == "H2", "scoring_seconds"].sum(),
        mint["scoring_seconds"].sum(),
    ]
    figure, axis = plt.subplots(figsize=(8.2, 5.2))
    bars = axis.bar(labels, seconds, color=["#4c78a8", "#f58518", "#b279a2"])
    axis.bar_label(bars, labels=[f"{value:.1f} s" for value in seconds], padding=3)
    axis.set_ylabel("Temps cumulé de calcul des scores (s)")
    axis.set_title("Coût hors ligne du scoring de pruning")
    axis.grid(axis="y", alpha=0.25)
    save_figure(figure, EXPERIMENTAL_DIR / "cout_scoring_pruning.png")


def checkpoint_path(head: str, model: str) -> Path:
    suffix = model.replace("MINT-", "MINT_")
    return ARTIFACTS / f"classifier_W1_{head}_{suffix}.pt"


@torch.inference_mode()
def head_inference(model: SentimentHead, features: torch.Tensor) -> None:
    for start in range(0, len(features), LATENCY_BATCH_SIZE):
        model(features[start : start + LATENCY_BATCH_SIZE])


def median_runtime_seconds(operation) -> float:
    for _ in range(LATENCY_WARMUPS):
        operation()
    durations = []
    for _ in range(LATENCY_REPEATS):
        started = perf_counter()
        operation()
        durations.append(perf_counter() - started)
    return float(np.median(durations))


def deployment_size_bytes(
    word2vec_input_embeddings: torch.Tensor,
    classifier_state: dict[str, torch.Tensor],
) -> int:
    """Mesure un artefact d'inférence dense, sans table CBOW de sortie ni masque."""
    with TemporaryDirectory(prefix="compression-deployment-") as directory:
        path = Path(directory) / "model.pt"
        torch.save(
            {
                "input_embeddings": word2vec_input_embeddings,
                "classifier_state": classifier_state,
            },
            path,
        )
        return path.stat().st_size


def measure_runtime(results: pd.DataFrame) -> pd.DataFrame:
    test_vectors = torch.load(
        ARTIFACTS / "test_vectors_W1.pt", map_location="cpu", weights_only=True
    )[:LATENCY_SAMPLE_COUNT]
    test_encoded = pd.read_pickle(ARTIFACTS / "test_encoded.pkl").iloc[
        :LATENCY_SAMPLE_COUNT
    ]
    word2vec_checkpoint = torch.load(
        ARTIFACTS / "word2vec_W1.pt", map_location="cpu", weights_only=True
    )
    word2vec = CBOW(
        word2vec_checkpoint["vocab_size"], word2vec_checkpoint["embedding_dim"]
    )
    word2vec.load_state_dict(word2vec_checkpoint["model_state"])
    word2vec.eval()
    embedding = word2vec.input_embeddings.weight.detach().cpu()

    representation_seconds = median_runtime_seconds(
        lambda: text_vectors(word2vec, test_encoded)
    )
    rows = []
    for result in ordered_models(results).itertuples():
        path = checkpoint_path(result.head, result.model)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = SentimentHead(checkpoint["input_dim"], checkpoint["hidden_dims"])
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        head_seconds = median_runtime_seconds(lambda: head_inference(model, test_vectors))
        state = {
            name: tensor.detach().cpu()
            for name, tensor in checkpoint["model_state"].items()
        }
        rows.append(
            {
                "head": result.head,
                "model": result.model,
                "samples": len(test_vectors),
                "batch_size": LATENCY_BATCH_SIZE,
                "repetitions": LATENCY_REPEATS,
                "head_latency_ms_per_sample": head_seconds * 1_000 / len(test_vectors),
                "representation_latency_ms_per_sample": representation_seconds
                * 1_000
                / len(test_vectors),
                "pipeline_latency_ms_per_sample": (
                    head_seconds + representation_seconds
                )
                * 1_000
                / len(test_vectors),
                "deployment_bytes": deployment_size_bytes(embedding, state),
            }
        )
    measurements = pd.DataFrame(rows)
    measurements.to_csv(ARTIFACTS / "runtime_measurements.csv", index=False)
    return measurements


def plot_theory_vs_reality(results: pd.DataFrame, runtime: pd.DataFrame) -> None:
    merged = results.merge(runtime, on=["head", "model"], validate="one_to_one")
    subset = ordered_models(merged[merged["head"] == "H2"])
    baseline = subset[subset["model"] == "M1"].iloc[0]
    labels = subset["model"].tolist()
    positions = np.arange(len(subset))
    width = 0.24

    parameter_reduction = 100 * (
        1 - subset["head_effective_parameters"] / baseline["head_parameters"]
    )
    file_reduction = 100 * (
        1 - subset["deployment_bytes"] / baseline["deployment_bytes"]
    )
    mac_reduction = 100 * (
        1 - subset["head_effective_macs"] / baseline["head_dense_macs"]
    )
    latency_reduction = 100 * (
        1
        - subset["head_latency_ms_per_sample"]
        / baseline["head_latency_ms_per_sample"]
    )
    pipeline_latency_reduction = 100 * (
        1
        - subset["pipeline_latency_ms_per_sample"]
        / baseline["pipeline_latency_ms_per_sample"]
    )

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)
    axes[0].bar(
        positions - width / 2,
        parameter_reduction,
        width,
        label="Paramètres effectifs (idéal)",
        color="#4c78a8",
    )
    axes[0].bar(
        positions + width / 2,
        file_reduction,
        width,
        label="Artefact dense mesuré",
        color="#f58518",
    )
    axes[1].bar(
        positions - width,
        mac_reduction,
        width,
        label="MAC effectives (idéal)",
        color="#54a24b",
    )
    axes[1].bar(
        positions,
        latency_reduction,
        width,
        label="Latence tête mesurée",
        color="#e45756",
    )
    axes[1].bar(
        positions + width,
        pipeline_latency_reduction,
        width,
        label="Latence pipeline mesurée",
        color="#b279a2",
    )
    for axis, title in zip(
        axes, ("Gain mémoire", "Gain de calcul et de latence")
    ):
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(positions, labels, rotation=20)
        axis.set_title(title)
        axis.set_ylabel("Réduction par rapport à H2/M1 (%)")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("Écart entre gains idéaux et gains observés")
    figure.tight_layout()
    save_figure(figure, EXPERIMENTAL_DIR / "gain_theorique_vs_reel.png")


def main() -> None:
    THEORETICAL_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTAL_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(ARTIFACTS / "classification_results.csv")
    required_columns = {
        "head_total_weights",
        "head_bias_parameters",
        "head_effective_parameters",
        "head_dense_macs",
        "head_effective_macs",
    }
    missing = required_columns - set(results.columns)
    if missing:
        raise RuntimeError(
            "Les nouveaux indicateurs sont absents des résultats : "
            + ", ".join(sorted(missing))
        )
    results = add_resource_ratios(results)

    plot_dense_parameter_distribution(results)
    plot_theoretical_sparsity_effect(results)
    plot_quality_resource(
        results,
        "ratio_parametres_tete",
        "Paramètres effectifs relatifs de la tête",
        "qualite_memoire_tete_pareto.png",
    )
    plot_quality_resource(
        results,
        "ratio_parametres_pipeline",
        "Paramètres effectifs relatifs du pipeline",
        "qualite_memoire_pipeline_pareto.png",
    )
    plot_quality_resource(
        results,
        "ratio_mac_tete",
        "MAC effectives relatives de la tête",
        "qualite_calcul_tete_pareto.png",
    )
    plot_adjusted_quality(results)
    plot_obd_score_distributions()
    plot_scoring_costs()
    runtime = measure_runtime(results)
    plot_theory_vs_reality(results, runtime)

    theoretical_count = len(list(THEORETICAL_DIR.glob("*.png")))
    experimental_count = len(list(EXPERIMENTAL_DIR.glob("*.png")))
    print(
        f"Figures du rapport générées : {theoretical_count} théoriques et "
        f"{experimental_count} expérimentales."
    )


if __name__ == "__main__":
    main()
