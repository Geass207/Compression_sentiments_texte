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
    plot_mint_score_distributions,
    plot_mint_summary,
    plot_roc_curves,
    plot_training_losses,
)

ADJUSTED_LOSS_ALPHA = 0.4

# Chargement des artefacts de résultats (OBD puis MINT)
results = pd.read_csv("artifacts/classification_results.csv")
training_history = pd.read_csv("artifacts/classifier_training_history.csv")
classifier_pruning = pd.read_csv("artifacts/classifier_pruning_metrics.csv")
mint_results = pd.read_csv("artifacts/mint_results.csv")
mint_scores = pd.read_csv("artifacts/mint_scores.csv")
mint_training_history = pd.read_csv("artifacts/mint_training_history.csv")
test = pd.read_pickle("artifacts/test_encoded.pkl")
test_labels = test["sentiment"].to_numpy()

figures_dir = Path("figures")
figures_dir.mkdir(exist_ok=True)

# Visualisations par tête de classification (H1 et H2)
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
        alpha=ADJUSTED_LOSS_ALPHA,
    )

    # Calcul et tracé des courbes ROC
    roc_curves = {}
    for word2vec in ("W1",):
        test_vectors = torch.load(
            f"artifacts/test_vectors_{word2vec}.pt",
            weights_only=True,
        )
        checkpoint_paths = {
            "M1": f"artifacts/classifier_{word2vec}_{head}_M1.pt",
            "M2": f"artifacts/classifier_{word2vec}_{head}_M2.pt",
            "M3": f"artifacts/classifier_{word2vec}_{head}_M3.pt",
            "M4": f"artifacts/classifier_{word2vec}_{head}_M4.pt",
        }
        if head == "H2":
            checkpoint_paths.update(
                {
                    "MINT-M2": "artifacts/classifier_W1_H2_MINT_M2.pt",
                    "MINT-M3": "artifacts/classifier_W1_H2_MINT_M3.pt",
                }
            )

        for model_code, checkpoint_path in checkpoint_paths.items():
            checkpoint = torch.load(
                checkpoint_path,
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

plot_mint_score_distributions(
    mint_scores,
    figures_dir / "mint_score_distributions_W1_H2.png",
)
plot_mint_summary(
    mint_results,
    mint_training_history,
    figures_dir / "mint_summary_W1_H2.png",
)

# Même loss ajustée que pour OBD, limitée à la référence et aux étapes MINT.
mint_comparison = results[
    (results["word2vec"] == "W1")
    & (results["head"] == "H2")
    & (results["model"].isin(["M1", "MINT-M2", "MINT-M3"]))
].copy()
plot_adjusted_loss_benchmark(
    mint_comparison,
    "H2",
    figures_dir / "adjusted_loss_mint_W1_H2.png",
    alpha=ADJUSTED_LOSS_ALPHA,
)

print("Toutes les figures ont été générées avec succès dans 'figures/'.")
