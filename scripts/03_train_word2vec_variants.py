import sys
from pathlib import Path

import pandas as pd
import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.word2vec import (
    CBOW,
    evaluate_cbow,
    make_cbow_dataset,
    negative_sampling_distribution,
    text_vectors,
    train_cbow,
)


FULL_DIMENSION = 200
WINDOW_SIZE = 2
NEGATIVE_SAMPLES = 5
BATCH_SIZE = 512
TRAINING_EPOCHS = 5
LEARNING_RATE = 1e-3
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)

train = pd.read_pickle("artifacts/train_encoded.pkl")
validation = pd.read_pickle("artifacts/validation_encoded.pkl")
test = pd.read_pickle("artifacts/test_encoded.pkl")
tokenizer = Tokenizer.from_file("artifacts/tokenizer.json")

vocab_size = tokenizer.get_vocab_size()
probabilities = negative_sampling_distribution(train, vocab_size)
train_dataset = make_cbow_dataset(train, WINDOW_SIZE)
validation_dataset = make_cbow_dataset(validation, WINDOW_SIZE)

# W1 : unique Word2Vec partagé par toutes les expériences.
w1 = CBOW(vocab_size, FULL_DIMENSION).to(device)
train_cbow(
    w1,
    train_dataset,
    probabilities,
    k=NEGATIVE_SAMPLES,
    batch_size=BATCH_SIZE,
    epochs=TRAINING_EPOCHS,
    lr=LEARNING_RATE,
)

splits = {"train": train, "validation": validation, "test": test}
validation_loss = evaluate_cbow(
    w1,
    validation_dataset,
    probabilities,
    k=NEGATIVE_SAMPLES,
    batch_size=BATCH_SIZE,
    seed=SEED,
)
training_parameter_count = sum(parameter.numel() for parameter in w1.parameters())
inference_parameter_count = w1.input_embeddings.weight.numel()

pd.DataFrame(
    [
        {
            "word2vec": "W1",
            "description": "complet_d200",
            "embedding_dim": w1.embedding_dim,
            "training_parameters": training_parameter_count,
            "inference_parameters": inference_parameter_count,
            "validation_loss": validation_loss,
        }
    ]
).to_csv("artifacts/word2vec_metrics.csv", index=False)

torch.save(
    {
        "variant": "W1",
        "description": "complet_d200",
        "model_state": w1.state_dict(),
        "vocab_size": vocab_size,
        "embedding_dim": w1.embedding_dim,
        "window_size": WINDOW_SIZE,
        "negative_samples": NEGATIVE_SAMPLES,
        "kept_original_dimensions": torch.arange(w1.embedding_dim),
    },
    "artifacts/word2vec_W1.pt",
)

for split_name, data in splits.items():
    torch.save(
        text_vectors(w1, data),
        f"artifacts/{split_name}_vectors_W1.pt",
    )
