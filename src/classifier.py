import copy

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class SentimentHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: int | tuple[int, ...] | list[int] = (64,),
    ):
        super().__init__()

        if isinstance(hidden_dims, int):
            hidden_dims = (hidden_dims,)
        hidden_dims = tuple(hidden_dims)
        dimensions = (input_dim, *hidden_dims)
        self.hidden_layers = nn.ModuleList(
            nn.Linear(dimensions[i], dimensions[i + 1])
            for i in range(len(hidden_dims))
        )
        self.output_layer = nn.Linear(dimensions[-1], 1)
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        for layer in self.hidden_layers:
            features = F.relu(layer(features))
        return self.output_layer(features).squeeze(1)


def train_classifier(
    model: SentimentHead,
    train_features: torch.Tensor | np.ndarray,
    train_labels: torch.Tensor | np.ndarray,
    validation_features: torch.Tensor | np.ndarray,
    validation_labels: torch.Tensor | np.ndarray,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int | None = 4,
    masks: dict[str, torch.Tensor] | None = None,
) -> dict[str, list[float]]:
    train_features = torch.as_tensor(train_features, dtype=torch.float32)
    train_labels = torch.tensor(
        np.asarray(train_labels), dtype=torch.float32
    )
    dataset = TensorDataset(train_features, train_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    masks = masks or {}
    masks = {name: mask.to(device) for name, mask in masks.items()}

    history = {
        "train_loss": [],
        "validation_loss": [],
        "validation_accuracy": [],
        "validation_f1": [],
    }
    best_loss = float("inf")
    best_state = None
    stale_epochs = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(features), labels)
            loss.backward()

            for name, parameter in model.named_parameters():
                if name in masks:
                    parameter.grad.mul_(masks[name])

            optimizer.step()

            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if name in masks:
                        parameter.mul_(masks[name])

            total_loss += loss.item() * len(labels)

        metrics, _ = evaluate_classifier(
            model,
            validation_features,
            validation_labels,
            batch_size=batch_size,
        )
        history["train_loss"].append(total_loss / len(dataset))
        history["validation_loss"].append(metrics["loss"])
        history["validation_accuracy"].append(metrics["accuracy"])
        history["validation_f1"].append(metrics["f1"])

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- loss: {history['train_loss'][-1]:.4f} "
            f"- val_loss: {metrics['loss']:.4f} "
            f"- val_f1: {metrics['f1']:.4f}"
        )

        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if patience is not None and stale_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


@torch.no_grad()
def predict_classifier(
    model: SentimentHead,
    features: torch.Tensor | np.ndarray,
    batch_size: int = 1024,
) -> np.ndarray:
    features = torch.as_tensor(features, dtype=torch.float32)
    dataloader = DataLoader(features, batch_size=batch_size)
    device = next(model.parameters()).device
    probabilities = []
    model.eval()

    for batch in dataloader:
        probabilities.append(torch.sigmoid(model(batch.to(device))).cpu())

    return torch.cat(probabilities).numpy()


def evaluate_classifier(
    model: SentimentHead,
    features: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
    batch_size: int = 1024,
) -> tuple[dict[str, float], np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = predict_classifier(model, features, batch_size)
    predictions = (probabilities >= 0.5).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0,
    )
    clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)

    metrics = {
        "loss": float(
            -np.mean(
                labels * np.log(clipped)
                + (1 - labels) * np.log(1 - clipped)
            )
        ),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
    }
    return metrics, probabilities
