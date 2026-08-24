"""Fonctions de visualisation des résultats du pipeline."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid", palette="tab10")
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "figure.titlesize": 14,
    }
)


def _save_figure(figure: plt.Figure, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_word2vec_pruning_trajectory(
    pruning: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Trace la loss Word2Vec avant et après chaque réentraînement."""
    required_columns = {
        "dimension",
        "validation_loss_before_retraining",
        "validation_loss_after_retraining",
    }
    missing_columns = required_columns.difference(pruning.columns)
    if missing_columns:
        raise ValueError(
            "Colonnes Word2Vec manquantes: "
            + ", ".join(sorted(missing_columns))
        )

    data = pruning.sort_values("dimension", ascending=False)
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(
        data["dimension"],
        data["validation_loss_before_retraining"],
        marker="o",
        linestyle="--",
        label="Avant réentraînement",
    )
    axis.plot(
        data["dimension"],
        data["validation_loss_after_retraining"],
        marker="s",
        linewidth=2,
        label="Après réentraînement",
    )
    axis.invert_xaxis()
    axis.set_title("Compression structurée de Word2Vec par OBD")
    axis.set_xlabel("Dimension de plongement restante")
    axis.set_ylabel("Loss de validation")
    axis.legend()
    _save_figure(figure, output_path)


def plot_training_losses(
    history: pd.DataFrame,
    head: str,
    output_path: str | Path,
) -> None:
    """Compare les losses de validation des entraînements d'une tête."""
    data = history[history["head"] == head].copy()
    if data.empty:
        raise ValueError(f"Aucun historique d'entraînement pour {head}.")

    regular = data[data["phase"].isin(["initial_training", "direct_training"])]
    pruning = data[data["phase"] == "pruning_retraining"].copy()

    figure, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.lineplot(
        data=regular,
        x="epoch",
        y="validation_loss",
        hue="word2vec",
        style="model",
        ax=axes[0],
    )
    axes[0].set_title(f"Entraînement initial et direct — {head}")
    axes[0].set_xlabel("Époque")
    axes[0].set_ylabel("Loss de validation")

    if not pruning.empty:
        epochs_per_iteration = int(pruning["epoch"].max())
        pruning["global_epoch"] = (
            (pruning["pruning_iteration"].astype(int) - 1)
            * epochs_per_iteration
            + pruning["epoch"]
        )
        sns.lineplot(
            data=pruning,
            x="global_epoch",
            y="validation_loss",
            hue="word2vec",
            marker="o",
            ax=axes[1],
        )
    axes[1].set_title(f"Réentraînement entre les tours de pruning — {head}")
    axes[1].set_xlabel("Époque cumulée de réentraînement")
    axes[1].set_ylabel("Loss de validation")
    _save_figure(figure, output_path)


def plot_classifier_pruning_trajectory(
    pruning: pd.DataFrame,
    head: str,
    output_path: str | Path,
) -> None:
    """Montre l'impact du pruning et la récupération par réentraînement."""
    data = pruning[pruning["head"] == head].copy()
    if data.empty:
        raise ValueError(f"Aucune métrique de pruning pour {head}.")

    stage_labels = {
        "before_retraining": "Avant réentraînement",
        "after_retraining": "Après réentraînement",
    }
    data["stage"] = data["stage"].replace(stage_labels)

    figure, axis = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=data,
        x="sparsity",
        y="f1",
        hue="word2vec",
        style="stage",
        markers=True,
        dashes=True,
        ax=axis,
    )
    axis.set_title(f"Trajectoire du pruning OBD — {head}")
    axis.set_xlabel("Sparsité des poids")
    axis.set_ylabel("F1 de validation")
    axis.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda value, _: f"{value:.0%}")
    )
    _save_figure(figure, output_path)


def plot_adjusted_loss_benchmark(
    results: pd.DataFrame,
    head: str,
    output_path: str | Path,
    alpha: float = 0.4,
) -> None:
    """Compare performance et taille via la loss ajustée du benchmark."""
    data = results[results["head"] == head].copy()
    if data.empty:
        raise ValueError(f"Aucun résultat de classification pour {head}.")

    max_parameters = data["effective_parameters"].max()
    data["adjusted_loss"] = data["test_loss"] + alpha * (
        data["effective_parameters"] / max_parameters
    )

    figure, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.barplot(
        data=data,
        x="word2vec",
        y="adjusted_loss",
        hue="model",
        errorbar=None,
        ax=axes[0],
    )
    axes[0].set_title(f"Loss ajustée (alpha={alpha}) — {head}")
    axes[0].set_xlabel("Variante Word2Vec")
    axes[0].set_ylabel("Loss ajustée (plus basse = meilleure)")

    sns.scatterplot(
        data=data,
        x="effective_parameters",
        y="test_f1",
        hue="model",
        style="word2vec",
        s=100,
        ax=axes[1],
    )
    axes[1].set_title(f"Performance / compression — {head}")
    axes[1].set_xlabel("Paramètres effectifs")
    axes[1].set_ylabel("F1 de test")
    axes[1].ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    _save_figure(figure, output_path)


def plot_roc_curves(
    roc_curves: dict[tuple[str, str], tuple],
    head: str,
    output_path: str | Path,
) -> None:
    """Trace les courbes ROC pour chaque variante Word2Vec fournie."""
    word2vec_variants = tuple(
        sorted({word2vec for word2vec, _ in roc_curves})
    )
    model_variants = ("M1", "M2", "M3", "M4")
    figure, axes = plt.subplots(
        1,
        len(word2vec_variants),
        figsize=(6 * len(word2vec_variants), 5),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for axis, word2vec in zip(axes[0], word2vec_variants):
        for model in model_variants:
            key = (word2vec, model)
            if key not in roc_curves:
                continue
            false_positive_rate, true_positive_rate, area = roc_curves[key]
            axis.plot(
                false_positive_rate,
                true_positive_rate,
                label=f"{model} (AUC={area:.3f})",
            )
        axis.plot([0, 1], [0, 1], "k--", linewidth=1)
        axis.set_title(word2vec)
        axis.set_xlabel("Taux de faux positifs")
        axis.set_ylabel("Taux de vrais positifs")
        axis.legend(loc="lower right")

    figure.suptitle(f"Courbes ROC sur le jeu de test — {head}")
    _save_figure(figure, output_path)
