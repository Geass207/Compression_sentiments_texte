import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tokenizer import train_bpe, encode_dataframe
from src.experiment_config import MIN_TOKEN_FREQUENCY, VOCAB_SIZE


train = pd.read_pickle("artifacts/train_clean.pkl")
validation = pd.read_pickle("artifacts/validation_clean.pkl")
test = pd.read_pickle("artifacts/test_clean.pkl")

# entraînement uniquement sur train
bpe = train_bpe(
    train["text"],
    vocab_size=VOCAB_SIZE,
    min_frequency=MIN_TOKEN_FREQUENCY,
)

# sauvegarde du tokenizer
bpe.save("artifacts/tokenizer.json")

# encodage
train = encode_dataframe(train, bpe)
validation = encode_dataframe(validation, bpe)
test = encode_dataframe(test, bpe)

# sauvegarde des DataFrames encodés
train.to_pickle("artifacts/train_encoded.pkl")
validation.to_pickle("artifacts/validation_encoded.pkl")
test.to_pickle("artifacts/test_encoded.pkl")
