import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def make_cbow_pairs(token_ids: list[int], window_size: int = 2) -> list[tuple[list[int], int]]:
    pairs = []

    for i in range(window_size, len(token_ids) - window_size):
        context = token_ids[i - window_size:i] + token_ids[i + 1:i + window_size + 1]
        pairs.append((context, token_ids[i]))

    return pairs


def make_cbow_dataset(data: pd.DataFrame, window_size: int = 2) -> TensorDataset:
    contexts = []
    targets = []

    for token_ids in data["input_ids"]:
        pairs_temp = make_cbow_pairs(token_ids, window_size)
        for context, target in pairs_temp:
            contexts.append(context)
            targets.append(target)

    contexts = torch.tensor(contexts, dtype=torch.long).reshape(
        -1, 2 * window_size
    )
    targets = torch.tensor(targets, dtype=torch.long)
    
    return TensorDataset(contexts, targets)


def negative_sampling_distribution(data: pd.DataFrame, vocab_size: int) -> torch.Tensor:
    counts = torch.zeros(vocab_size)

    for token_ids in data["input_ids"]:
        index = torch.as_tensor(token_ids, dtype=torch.long)
        src = torch.ones(len(token_ids))
        counts.scatter_add_(0, index, src)

    probabilities = counts.pow(0.75)
    return probabilities / probabilities.sum()


def sample_negatives(probabilities: torch.Tensor, targets: int | torch.Tensor, k: int = 5, generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Échantillonne les négatifs d'un exemple ou de tout un batch."""
    targets = torch.as_tensor(targets, device=probabilities.device)
    scalar_target = targets.ndim == 0
    targets = targets.reshape(-1)

    negatives = torch.multinomial(
        probabilities,
        num_samples=targets.numel() * k,
        replacement=True,
        generator=generator,
    ).reshape(targets.numel(), k)

    collisions = negatives.eq(targets[:, None])
    while collisions.any():
        negatives[collisions] = torch.multinomial(
            probabilities,
            num_samples=int(collisions.sum()),
            replacement=True,
            generator=generator,
        )
        collisions = negatives.eq(targets[:, None])

    return negatives[0] if scalar_target else negatives


class CBOW(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()

        self.input_embeddings = nn.Embedding(vocab_size, embedding_dim) #(V, d)
        self.output_embeddings = nn.Embedding(vocab_size, embedding_dim) #(V, d)

        #initialisation 
        bound = 0.5 / embedding_dim
        nn.init.uniform_(self.input_embeddings.weight, -bound, bound)
        nn.init.zeros_(self.output_embeddings.weight)

    @property
    def embedding_dim(self) -> int:
        return self.input_embeddings.embedding_dim

    def forward(
        self,
        contexts: torch.Tensor,  #(B, C)
        targets: torch.Tensor,  #(B,)
        negatives: torch.Tensor, #(B, K)
    ) -> tuple[torch.Tensor, torch.Tensor]:  #[(B,), (B, K)]
        context_vectors = self.input_embeddings(contexts).mean(dim=1) #(B, d) = (B, C, d).mean(dim=1)
        positive_embeddings = self.output_embeddings(targets) #(B, d)
        negative_embeddings = self.output_embeddings(negatives) #(B, K, d)

        positive_scores = (context_vectors * positive_embeddings).sum(dim=1) # (B,) = [ (B, d) * (B, d) ].mean(dim=1)
        negative_scores = torch.bmm(
            negative_embeddings, context_vectors.unsqueeze(2)   # [ (B, K, d) * (B, d, 1) ] = (B, K, 1) -> (B, K)
        ).squeeze(2) 

        return positive_scores, negative_scores


def negative_sampling_loss(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
) -> torch.Tensor:
    positive_loss = F.logsigmoid(positive_scores)
    negative_loss = F.logsigmoid(-negative_scores).sum(dim=1)
    return -(positive_loss + negative_loss).mean()


def train_cbow(
    model: CBOW,
    data: TensorDataset,
    probabilities: torch.Tensor,
    window_size: int = 2,
    k: int = 5,
    batch_size: int = 128,
    epochs: int = 5,
    lr: float = 0.01,
) -> CBOW:
    
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    device = next(model.parameters()).device
    probabilities = probabilities.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for contexts, targets in dataloader:
            contexts = contexts.to(device)
            targets = targets.to(device)
            negatives = sample_negatives(probabilities, targets, k)

            optimizer.zero_grad(set_to_none=True)
            positive_scores, negative_scores = model(
                contexts, targets, negatives
            )
            loss = negative_sampling_loss(positive_scores, negative_scores)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(targets)

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- loss: {total_loss / len(data):.4f}"
        )

    return model


@torch.no_grad()
def evaluate_cbow(
    model: CBOW,
    data: pd.DataFrame | TensorDataset,
    probabilities: torch.Tensor,
    window_size: int = 2,
    k: int = 5,
    batch_size: int = 1024,
    seed: int = 42,
) -> float:
    dataset = (
        make_cbow_dataset(data, window_size)
        if isinstance(data, pd.DataFrame)
        else data
    )
    dataloader = DataLoader(dataset, batch_size=batch_size)
    device = next(model.parameters()).device
    probabilities = probabilities.to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    total_loss = 0.0
    model.eval()

    for contexts, targets in dataloader:
        contexts = contexts.to(device)
        targets = targets.to(device)
        negatives = sample_negatives(
            probabilities, targets, k, generator=generator
        )
        positive_scores, negative_scores = model(contexts, targets, negatives)
        loss = negative_sampling_loss(positive_scores, negative_scores)
        total_loss += loss.item() * len(targets)

    return total_loss / len(dataset)


@torch.no_grad()
def text_vectors(model: CBOW, data: pd.DataFrame) -> torch.Tensor:
    """Retourne la moyenne des embeddings d'entrée pour chaque texte."""
    sequences = data["input_ids"].tolist()
    lengths = torch.tensor([len(ids) for ids in sequences], dtype=torch.long)
    offsets = torch.cat((torch.zeros(1, dtype=torch.long), lengths.cumsum(0)[:-1]))
    token_ids = torch.tensor(
        [token_id for ids in sequences for token_id in ids], dtype=torch.long
    )
    device = next(model.parameters()).device

    return F.embedding_bag(
        token_ids.to(device),
        model.input_embeddings.weight,
        offsets.to(device),
        mode="mean",
    ).cpu()
