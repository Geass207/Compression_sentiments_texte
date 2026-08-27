"""Primitives MINT pour le pruning par information mutuelle géométrique.

Ce module suit la construction décrite dans ``Doc_urgent`` : un bootstrap
conditionnel 1-NN produit une population synthétique, puis la statistique de
Friedman--Rafsky sur un arbre couvrant minimal estime la GMI conditionnelle.
Il ne modifie aucun modèle ; l'intégration au pipeline de pruning sera faite
séparément.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from src.classifier import SentimentHead


def standardize_activations(
    activations: torch.Tensor | np.ndarray,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Standardise chaque unité à partir de l'ensemble de calibration."""
    values = _as_matrix(activations, "activations")
    return (values - values.mean(dim=0, keepdim=True)) / values.std(
        dim=0, keepdim=True, unbiased=False
    ).clamp_min(eps)


def balanced_calibration_indices(
    labels: torch.Tensor | np.ndarray,
    samples_per_class: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sélectionne le même nombre d'observations pour chaque classe."""
    if samples_per_class <= 0:
        raise ValueError("samples_per_class doit être strictement positif.")

    labels = torch.as_tensor(labels).reshape(-1).cpu()
    classes = torch.unique(labels, sorted=True)
    selections = []
    for class_id in classes:
        candidates = torch.nonzero(labels == class_id, as_tuple=False).flatten()
        if len(candidates) < samples_per_class:
            raise ValueError(
                f"La classe {class_id.item()} contient {len(candidates)} exemples, "
                f"mais {samples_per_class} sont demandés."
            )
        order = torch.randperm(len(candidates), generator=generator)
        selections.append(candidates[order[:samples_per_class]])
    return torch.cat(selections)


def sequential_balanced_groups(
    number_of_units: int,
    number_of_groups: int,
) -> list[torch.Tensor]:
    """Découpe les indices en groupes séquentiels de tailles équilibrées."""
    if not 1 <= number_of_groups <= number_of_units:
        raise ValueError(
            "number_of_groups doit être compris entre 1 et number_of_units."
        )
    return list(torch.tensor_split(torch.arange(number_of_units), number_of_groups))


def nearest_neighbor_bootstrap(
    x: torch.Tensor | np.ndarray,
    y: torch.Tensor | np.ndarray,
    z: torch.Tensor | np.ndarray,
    source_indices: torch.Tensor,
    donor_indices: torch.Tensor,
) -> torch.Tensor:
    """Construit Q avec le bootstrap conditionnel 1-NN de ``Y | Z``.

    Les indices source et donneur doivent être disjoints. Les observations
    synthétiques conservent ``(X, Z)`` des sources et empruntent ``Y`` au
    donneur dont ``Z`` est le plus proche au sens euclidien.
    """
    x, y, z = _validate_triplet(x, y, z)
    source_indices = torch.as_tensor(source_indices, dtype=torch.long)
    donor_indices = torch.as_tensor(donor_indices, dtype=torch.long)
    if len(source_indices) == 0 or len(donor_indices) == 0:
        raise ValueError("Les partitions source et donneur ne peuvent pas être vides.")
    if torch.isin(source_indices, donor_indices).any():
        raise ValueError("Les partitions source et donneur doivent être disjointes.")

    source_z = z[source_indices]
    donor_z = z[donor_indices]
    nearest_donor = torch.cdist(source_z, donor_z).argmin(dim=1)
    borrowed_y = y[donor_indices[nearest_donor]]
    return torch.cat((x[source_indices], borrowed_y, source_z), dim=1)


def friedman_rafsky_cross_edges(
    points: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
) -> int:
    """Compte les arêtes inter-populations du MST euclidien des ``points``."""
    points = _as_matrix(points, "points")
    labels = torch.as_tensor(labels, dtype=torch.long).reshape(-1)
    if len(points) != len(labels):
        raise ValueError("points et labels doivent avoir le même nombre de lignes.")
    if len(points) < 2:
        raise ValueError("Le MST exige au moins deux points.")

    number_of_points = len(points)
    distances = torch.cdist(points, points)
    distances.fill_diagonal_(float("inf"))
    visited = torch.zeros(number_of_points, dtype=torch.bool)
    visited[0] = True
    nearest_distances = distances[0].clone()
    parents = torch.zeros(number_of_points, dtype=torch.long)
    nearest_distances[0] = float("inf")
    cross_edges = 0

    # Prim dense en O(n²) : pour chaque sommet non visité, on conserve
    # uniquement sa meilleure arête vers l'arbre courant. L'ancienne version
    # clonait toute la matrice n x n à chacune des n étapes, soit O(n³).
    for _ in range(number_of_points - 1):
        child = int(nearest_distances.argmin().item())
        parent = int(parents[child].item())
        if labels[parent] != labels[child]:
            cross_edges += 1
        visited[child] = True

        improvements = (~visited) & (distances[child] < nearest_distances)
        nearest_distances[improvements] = distances[child, improvements]
        parents[improvements] = child
        nearest_distances[visited] = float("inf")

    return cross_edges


def estimate_conditional_gmi(
    x: torch.Tensor | np.ndarray,
    y: torch.Tensor | np.ndarray,
    z: torch.Tensor | np.ndarray,
    partition_size: int,
    generator: torch.Generator | None = None,
    standardize: bool = True,
) -> float:
    """Estime ``I_G(X; Y | Z)`` par bootstrap 1-NN et MST.

    Trois partitions disjointes de ``partition_size`` observations sont
    tirées : une population réelle P, des sources et des donneurs pour la
    population synthétique Q. Le score retourné est borné dans ``[0, 1]``.
    """
    if partition_size <= 0:
        raise ValueError("partition_size doit être strictement positif.")
    x, y, z = _validate_triplet(x, y, z)
    required_samples = 3 * partition_size
    if len(x) < required_samples:
        raise ValueError(
            f"Il faut au moins {required_samples} observations pour trois partitions, "
            f"mais {len(x)} ont été fournies."
        )

    if standardize:
        x, y, z = (
            standardize_activations(x),
            standardize_activations(y),
            standardize_activations(z),
        )

    permutation = torch.randperm(len(x), generator=generator)[:required_samples]
    real_indices, source_indices, donor_indices = permutation.split(partition_size)
    real_points = torch.cat((x[real_indices], y[real_indices], z[real_indices]), dim=1)
    synthetic_points = nearest_neighbor_bootstrap(
        x, y, z, source_indices, donor_indices
    )
    points = torch.cat((real_points, synthetic_points), dim=0)
    labels = torch.cat(
        (
            torch.ones(partition_size, dtype=torch.long),
            torch.zeros(partition_size, dtype=torch.long),
        )
    )
    cross_edges = friedman_rafsky_cross_edges(points, labels)
    return float(max(0.0, min(1.0, 1.0 - cross_edges / partition_size)))


def mint_group_scores(
    inputs: torch.Tensor | np.ndarray,
    outputs: torch.Tensor | np.ndarray,
    number_of_groups: int,
    partition_size: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Calcule un score MINT pour chaque unité aval et groupe amont.

    ``inputs`` est l'activation en entrée d'une couche linéaire et ``outputs``
    sa sortie activée (ou son logit pour la sortie binaire). Le résultat a la
    forme ``[nombre_d_unites_avales, number_of_groups]``.
    """
    inputs = standardize_activations(inputs)
    outputs = standardize_activations(outputs)
    if len(inputs) != len(outputs):
        raise ValueError("inputs et outputs doivent avoir le même nombre de lignes.")
    groups = sequential_balanced_groups(inputs.shape[1], number_of_groups)
    scores = torch.empty((outputs.shape[1], len(groups)), dtype=torch.float32)

    for output_index in range(outputs.shape[1]):
        for group_index, group in enumerate(groups):
            complement = torch.ones(inputs.shape[1], dtype=torch.bool)
            complement[group] = False
            scores[output_index, group_index] = estimate_conditional_gmi(
                outputs[:, output_index],
                inputs[:, group],
                inputs[:, complement],
                partition_size=partition_size,
                generator=generator,
                standardize=False,
            )
    return scores, groups


@torch.no_grad()
def collect_sentiment_head_activations(
    model: SentimentHead,
    features: torch.Tensor | np.ndarray,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Retourne ``(entrée, sortie)`` pour chaque couche linéaire de la tête.

    Les couches cachées utilisent la sortie après ReLU. La dernière sortie est
    le logit, conformément au choix décrit dans le document de référence.
    """
    device = next(model.parameters()).device
    current = torch.as_tensor(features, dtype=torch.float32, device=device)
    activations = []
    model.eval()

    for layer in model.hidden_layers:
        layer_input = current
        current = F.relu(layer(current))
        activations.append((layer_input.cpu(), current.cpu()))

    activations.append((current.cpu(), model.output_layer(current).cpu()))
    return activations


def mint_classifier_group_scores(
    model: SentimentHead,
    features: torch.Tensor | np.ndarray,
    groups_per_layer: Sequence[int],
    partition_size: int,
    generator: torch.Generator | None = None,
) -> list[tuple[torch.Tensor, list[torch.Tensor]]]:
    """Calcule les scores MINT de toutes les couches linéaires d'une tête."""
    activations = collect_sentiment_head_activations(model, features)
    if len(groups_per_layer) != len(activations):
        raise ValueError(
            "groups_per_layer doit contenir un nombre de groupes pour chaque "
            "couche linéaire."
        )
    return [
        mint_group_scores(inputs, outputs, number_of_groups, partition_size, generator)
        for (inputs, outputs), number_of_groups in zip(activations, groups_per_layer)
    ]


def mint_weight_mask(
    scores: torch.Tensor,
    input_groups: Sequence[torch.Tensor],
    weight_shape: tuple[int, int],
    target_sparsity: float | None = None,
    threshold: float | None = None,
    maximum_pruning_fraction: float = 1.0,
) -> torch.Tensor:
    """Construit un masque de poids par blocs à partir de scores MINT faibles.

    Choisir exactement un critère : ``target_sparsity`` sélectionne les blocs
    aux scores les plus faibles, tandis que ``threshold`` sélectionne les
    scores strictement inférieurs à ce seuil. La limite ``maximum_pruning_fraction``
    implémente la protection de couche ``gamma``.
    """
    if (target_sparsity is None) == (threshold is None):
        raise ValueError("Fournir exactement target_sparsity ou threshold.")
    if not 0.0 <= maximum_pruning_fraction <= 1.0:
        raise ValueError("maximum_pruning_fraction doit être dans [0, 1].")
    if target_sparsity is not None and not 0.0 <= target_sparsity <= 1.0:
        raise ValueError("target_sparsity doit être dans [0, 1].")
    if scores.shape != (weight_shape[0], len(input_groups)):
        raise ValueError("La forme de scores ne correspond pas aux blocs de poids.")
    if sum(len(group) for group in input_groups) != weight_shape[1]:
        raise ValueError("Les groupes ne couvrent pas toutes les entrées de la couche.")

    maximum_blocks = int(scores.numel() * maximum_pruning_fraction)
    if target_sparsity is not None:
        count = min(int(scores.numel() * target_sparsity), maximum_blocks)
        selected = torch.zeros_like(scores, dtype=torch.bool)
        if count:
            selected.flatten()[
                torch.topk(scores.flatten(), count, largest=False).indices
            ] = True
    else:
        selected = scores < threshold
        if selected.sum() > maximum_blocks:
            selected = torch.zeros_like(scores, dtype=torch.bool)
            if maximum_blocks:
                selected.flatten()[
                    torch.topk(scores.flatten(), maximum_blocks, largest=False).indices
                ] = True

    mask = torch.ones(weight_shape, dtype=torch.bool)
    for output_index, group_index in torch.nonzero(selected, as_tuple=False).tolist():
        mask[output_index, input_groups[group_index]] = False
    return mask


def _as_matrix(values: torch.Tensor | np.ndarray, name: str) -> torch.Tensor:
    values = torch.as_tensor(values, dtype=torch.float32).detach().cpu()
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError(f"{name} doit être un tenseur à une ou deux dimensions.")
    if not torch.isfinite(values).all():
        raise ValueError(f"{name} contient des valeurs non finies.")
    return values


def _validate_triplet(
    x: torch.Tensor | np.ndarray,
    y: torch.Tensor | np.ndarray,
    z: torch.Tensor | np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x, y, z = _as_matrix(x, "x"), _as_matrix(y, "y"), _as_matrix(z, "z")
    if len(x) != len(y) or len(x) != len(z):
        raise ValueError("x, y et z doivent avoir le même nombre d'observations.")
    return x, y, z
