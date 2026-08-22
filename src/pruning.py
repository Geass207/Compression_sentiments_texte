"""Scores OBD et moteurs de pruning itératif."""

from collections.abc import Callable
import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.classifier import SentimentHead, evaluate_classifier, train_classifier
from src.word2vec import (
    CBOW,
    evaluate_cbow,
    negative_sampling_loss,
    sample_negatives,
    train_cbow,
)


def autograd_hessian_diagonal(loss: torch.Tensor, named_parameters: list[tuple[str, torch.Tensor]], number_of_vectors: int = 10,
) -> dict[str, torch.Tensor]:
    """Estime diag(H) avec autograd et des produits Hessienne-vecteur.

    Pour un vecteur aléatoire v composé de -1 et 1 :
    E[v * (H @ v)] = diag(H).
    """
    names = [name for name, _ in named_parameters]
    parameters = [parameter for _, parameter in named_parameters]

    if loss.numel() != 1:
        raise ValueError(
            "Le calcul de la Hessienne exige une loss scalaire, "
            f"mais sa forme est {tuple(loss.shape)}."
        )
    
    gradients = torch.autograd.grad(loss, parameters, create_graph=True)

    #initialisation de la diagonale 
    diagonal = {
        name: torch.zeros_like(parameter)
        for name, parameter in named_parameters
    }

    
    for vector_index in range(number_of_vectors):
        vectors = []
        for parameter in parameters:
            vector = torch.empty_like(parameter).bernoulli_(0.5)
            vectors.append(vector.mul_(2).sub_(1))

        gradient_dot_vector = sum( (gradient * vector).sum() for gradient, vector in zip(gradients, vectors) )
        hessian_times_vector = torch.autograd.grad(gradient_dot_vector, parameters, retain_graph=vector_index < number_of_vectors - 1,)

        for name, vector, product in zip(names, vectors, hessian_times_vector,):
            diagonal[name] += (vector * product).detach()

    return {
        name: values / number_of_vectors
        for name, values in diagonal.items()
    }



def obd_classifier_weight_scores(
    model: SentimentHead,
    features: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
    batch_size: int = 1024,
    hessian_vectors: int = 10,
) -> dict[str, torch.Tensor]:
    """Retourne le score OBD S = 1/2 * H_ii * w²."""

    device = next(model.parameters()).device
    if isinstance(features, np.ndarray):
        features = features[:batch_size].copy()
    else:
        features = features[:batch_size]
    features = torch.as_tensor(features, dtype=torch.float32, device=device)
    if isinstance(labels, np.ndarray):
        labels = labels[:batch_size].copy()
    else:
        labels = labels[:batch_size]
    labels = torch.as_tensor(labels, dtype=torch.float32, device=device)

    model.eval()
    logits = model(features).squeeze(-1)
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    named_params = [
        (name, param)
        for name, param in model.named_parameters()
        if param.requires_grad
    ]
    hessian_diag = autograd_hessian_diagonal(
        loss=loss,
        named_parameters=named_params,
        number_of_vectors=hessian_vectors,
    )
    scores = {}
    for name, param in named_params:
        # 1/2 * H_ii * w^2
        scores[name] = 0.5 * hessian_diag[name].abs() * (param.data ** 2)

    return scores


    

@torch.no_grad()
def obd_cbow_dimension_scores(
    model: CBOW,
    dataset: TensorDataset,
    probabilities: torch.Tensor,
    k: int = 5,
    batch_size: int = 1024,
    hessian_vectors: int = 10,
) -> torch.Tensor:
    """Retourne un score OBD par dimension du Word2Vec (forme: [embedding_dim])."""
    device = next(model.parameters()).device
    model.eval()

    # 1. Extraction d'un batch de données
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    contexts, targets = next(iter(loader))
    contexts = contexts.to(device)
    targets = targets.to(device)

    # 2. Tirage des exemples négatifs (Negative Sampling)
    negative_samples = sample_negatives(
        probabilities,
        targets.to(probabilities.device),
        k,
    ).to(device)

    # 3. Calcul de la loss avec gradient actif (car la fonction a @torch.no_grad)
    with torch.enable_grad():
        positive_scores, negative_scores = model(
            contexts,
            targets,
            negative_samples,
        )
        loss = negative_sampling_loss(positive_scores, negative_scores)

        # Récupération des paramètres de plongement (embeddings)
        named_params = [
            (name, param)
            for name, param in model.named_parameters()
            if param.requires_grad
        ]

        # Estimation de la diagonale de la Hessienne
        hessian_diag = autograd_hessian_diagonal(
            loss=loss,
            named_parameters=named_params,
            number_of_vectors=hessian_vectors,
        )

    # 4. Calcul et agrégation des scores par dimension
    embedding_dim = next(model.parameters()).size(1)
    dimension_scores = torch.zeros(embedding_dim, device=device)

    for name, param in named_params:
        # Score OBD individuel par coefficient : 1/2 * |H_ii| * w² (shape: [vocab_size, embedding_dim])
        weight_saliency = 0.5 * hessian_diag[name].abs() * (param.data ** 2)
        
        # Somme sur tout le vocabulaire (dim=0) pour obtenir un score par dimension
        dimension_scores += weight_saliency.sum(dim=0)

    return dimension_scores












@torch.no_grad()
def prune_cbow_dimensions(
    model: CBOW,
    dimensions_to_keep: torch.Tensor,
) -> CBOW:
    """Construit un CBOW compact en copiant uniquement les dimensions gardées."""
    weight = model.input_embeddings.weight
    dimensions_to_keep = dimensions_to_keep.to(weight.device)

    compact_model = CBOW(
        model.input_embeddings.num_embeddings,
        len(dimensions_to_keep),
    ).to(weight.device)
    compact_model.input_embeddings.weight.copy_(
        model.input_embeddings.weight[:, dimensions_to_keep]
    )
    compact_model.output_embeddings.weight.copy_(
        model.output_embeddings.weight[:, dimensions_to_keep]
    )
    return compact_model


def iterative_prune_cbow_dimensions(
    model: CBOW,
    train_dataset: TensorDataset,
    probabilities: torch.Tensor,
    target_dimension: int,
    validation_dataset: TensorDataset | None = None,
    score_function: Callable[..., torch.Tensor] = obd_cbow_dimension_scores,
    pruning_step: int = 10,
    retraining_epochs: int = 1,
    k: int = 5,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> tuple[CBOW, CBOW, torch.Tensor, list[dict]]:
    """Répète score -> pruning -> réentraînement jusqu'à la taille cible."""
    current_model = copy.deepcopy(model)
    model_just_pruned = copy.deepcopy(model)
    original_dimensions = torch.arange(model.embedding_dim)
    history = []

    while current_model.embedding_dim > target_dimension:
        # 1. Recalculer les scores sur le modèle courant.
        scores = score_function(
            current_model,
            train_dataset,
            probabilities,
            k=k,
            batch_size=batch_size,
        )

        # 2. Retirer les dimensions ayant les plus petits scores.
        number_to_prune = min(
            pruning_step,
            current_model.embedding_dim - target_dimension,
        )
        dimensions_to_remove = torch.topk(
            scores,
            number_to_prune,
            largest=False,
        ).indices.cpu()

        keep_mask = torch.ones(current_model.embedding_dim, dtype=torch.bool)
        keep_mask[dimensions_to_remove] = False
        dimensions_to_keep = torch.arange(current_model.embedding_dim)[
            keep_mask
        ]
        removed_original_dimensions = original_dimensions[
            dimensions_to_remove
        ]
        original_dimensions = original_dimensions[dimensions_to_keep]

        current_model = prune_cbow_dimensions(
            current_model,
            dimensions_to_keep,
        )
        model_just_pruned = copy.deepcopy(current_model)

        validation_loss_before = None
        if validation_dataset is not None:
            validation_loss_before = evaluate_cbow(
                current_model,
                validation_dataset,
                probabilities,
                k=k,
                batch_size=batch_size,
            )

        # 3. Réentraîner, puis le prochain tour recalculera les scores.
        train_cbow(
            current_model,
            train_dataset,
            probabilities,
            k=k,
            batch_size=batch_size,
            epochs=retraining_epochs,
            lr=lr,
        )

        validation_loss_after = None
        if validation_dataset is not None:
            validation_loss_after = evaluate_cbow(
                current_model,
                validation_dataset,
                probabilities,
                k=k,
                batch_size=batch_size,
            )

        history.append(
            {
                "dimension": current_model.embedding_dim,
                "pruned_dimensions": removed_original_dimensions,
                "pruned_scores": scores[
                    dimensions_to_remove.to(scores.device)
                ].cpu(),
                "validation_loss_before_retraining": validation_loss_before,
                "validation_loss_after_retraining": validation_loss_after,
            }
        )
        print(
            "Pruning Word2Vec - dimension restante: "
            f"{current_model.embedding_dim}"
        )

    return (
        model_just_pruned,
        current_model,
        original_dimensions,
        history,
    )

#fonction principale de pruning principale 

def iterative_prune_classifier_weights(
    model: SentimentHead,
    train_features: torch.Tensor | np.ndarray,
    train_labels: torch.Tensor | np.ndarray,
    validation_features: torch.Tensor | np.ndarray,
    validation_labels: torch.Tensor | np.ndarray,
    target_sparsity: float,
    score_function: Callable[
        ..., dict[str, torch.Tensor]
    ] = obd_classifier_weight_scores,
    pruning_step: float = 0.1,
    retraining_epochs: int = 3,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int | None = None,
) -> tuple[
    SentimentHead,
    SentimentHead,
    dict[str, torch.Tensor],
    list[dict],
]:
    """Répète score -> pruning global -> réentraînement jusqu'à la sparsité cible."""
    current_model = copy.deepcopy(model)
    model_just_pruned = copy.deepcopy(model)
    device = next(current_model.parameters()).device

    masks = {}
    for name, parameter in current_model.named_parameters():
        if name.endswith("weight"):
            masks[name] = torch.ones_like(parameter, dtype=torch.bool)
    weight_names = list(masks)

    total_weights = sum(mask.numel() for mask in masks.values())
    target_pruned_weights = round(total_weights * target_sparsity)
    pruned_weights = 0
    history = []

    while pruned_weights < target_pruned_weights:
        # 1. Recalculer un score pour chaque poids du modèle courant.
        scores_by_layer = score_function(
            current_model,
            train_features,
            train_labels,
            batch_size=batch_size,
        )

        # On aplatit temporairement toutes les couches pour faire un pruning
        # global. Les poids déjà supprimés reçoivent +inf et sont donc ignorés.
        flat_scores = torch.cat(
            [scores_by_layer[name].to(device).flatten() for name in weight_names]
        )
        flat_mask = torch.cat([masks[name].flatten() for name in weight_names])
        flat_scores[~flat_mask] = torch.inf

        number_to_prune = min(
            max(1, round(total_weights * pruning_step)),
            target_pruned_weights - pruned_weights,
        )
        indices_to_prune = torch.topk(
            flat_scores,
            number_to_prune,
            largest=False,
        ).indices
        pruned_scores = flat_scores[indices_to_prune].detach().cpu()
        flat_mask[indices_to_prune] = False

        # 2. Remettre le masque aplati dans la forme de chaque couche.
        start = 0
        for name in weight_names:
            size = masks[name].numel()
            masks[name].copy_(
                flat_mask[start : start + size].reshape_as(masks[name])
            )
            start += size

        # Mettre réellement les poids sélectionnés à zéro.
        with torch.no_grad():
            for name, parameter in current_model.named_parameters():
                if name in masks:
                    parameter.mul_(masks[name])

        model_just_pruned = copy.deepcopy(current_model)
        metrics_before, _ = evaluate_classifier(
            current_model,
            validation_features,
            validation_labels,
            batch_size=batch_size,
        )

        # 3. Réentraîner sans autoriser les poids prunés à revenir.
        retraining_history = train_classifier(
            current_model,
            train_features,
            train_labels,
            validation_features,
            validation_labels,
            epochs=retraining_epochs,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
            masks=masks,
        )
        metrics_after, _ = evaluate_classifier(
            current_model,
            validation_features,
            validation_labels,
            batch_size=batch_size,
        )

        pruned_weights += number_to_prune
        history.append(
            {
                "sparsity": pruned_weights / total_weights,
                "pruned_scores": pruned_scores,
                "validation_metrics_before_retraining": metrics_before,
                "validation_metrics_after_retraining": metrics_after,
                "retraining_history": retraining_history,
            }
        )
        print(
            f"Pruning tête - sparsité: {pruned_weights / total_weights:.1%} "
            f"- val_f1: {metrics_after['f1']:.4f}"
        )

    masks_on_cpu = {name: mask.cpu() for name, mask in masks.items()}
    return model_just_pruned, current_model, masks_on_cpu, history
