import sys
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import auc, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.classifier import SentimentHead, predict_classifier
from src.visualization import (
    plot_adjusted_loss_benchmark,
    plot_classifier_pruning_trajectory,
    plot_roc_curves,
    plot_training_losses,
    plot_word2vec_pruning_trajectory,
)

# Chargement des artefacts de résultats
results = pd.read_csv("artifacts/classification_results.csv")
training_history = pd.read_csv("artifacts/classifier_training_history.csv")
classifier_pruning = pd.read_csv("artifacts/classifier_pruning_metrics.csv")
w2v_pruning = pd.read_csv("artifacts/word2vec_pruning_metrics.csv")

test = pd.read_pickle("artifacts/test_encoded.pkl")
test_labels = test["sentiment"].to_numpy()

figures_dir = Path("figures")
figures_dir.mkdir(exist_ok=True)

# 1. Visualisation de la réduction Word2Vec
plot_word2vec_pruning_trajectory(
    w2v_pruning,
    figures_dir / "word2vec_pruning_trajectory.png",
)

# 2. Visualisations par tête de classification (H1 et H2)
for head in ("H1", "H2"):
    # Courbes d'entraînement
    plot_training_losses(
        training_history,
        head,
        figures_dir / f"training_losses_{head}.png",
    )

    # Trajectoire de cicatrisation OBD
    plot_classifier_pruning_trajectory(
        classifier_pruning,
        head,
        figures_dir / f"pruning_trajectory_{head}.png",
    )

    # Loss ajustée et compromis de compression
    plot_adjusted_loss_benchmark(
        results,
        head,
        figures_dir / f"adjusted_loss_benchmark_{head}.png",
        alpha=0.4,
    )

    # Calcul et tracé des courbes ROC
    roc_curves = {}
    for word2vec in ("W1", "W2", "W3", "W4"):
        test_vectors = torch.load(
            f"artifacts/test_vectors_{word2vec}.pt",
            weights_only=True,
        )
        for model_code in ("M1", "M2", "M3", "M4"):
            checkpoint = torch.load(
                f"artifacts/classifier_{word2vec}_{head}_{model_code}.pt",
                map_location="cpu",
                weights_only=True,
            )
            model = SentimentHead(
                checkpoint["input_dim"],
                checkpoint["hidden_dims"],
            )
            model.load_state_dict(checkpoint["model_state"])

            probabilities = predict_classifier(model, test_vectors)
            fpr, tpr, _ = roc_curve(test_labels, probabilities)
            roc_curves[(word2vec, model_code)] = (
                fpr,
                tpr,
                auc(fpr, tpr),
            )

    plot_roc_curves(
        roc_curves,
        head,
        figures_dir / f"roc_curves_{head}.png",
    )

print("Toutes les figures ont été générées avec succès dans 'figures/'.")