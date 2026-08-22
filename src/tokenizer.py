from tokenizers import ByteLevelBPETokenizer, Encoding 
import pandas as pd 


def train_bpe(train_texts: pd.Series, vocab_size: int = 6000, min_frequency: int = 2) -> ByteLevelBPETokenizer:
    tokenizer = ByteLevelBPETokenizer(lowercase=True, add_prefix_space=True)
    tokenizer.train_from_iterator(train_texts, vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=["<PAD>", "<UNK>"], show_progress=True)
    return tokenizer

def encode_texts(tokenizer: ByteLevelBPETokenizer, texts: pd.Series) -> list[Encoding]:
    return [tokenizer.encode(text) for text in texts]


def encode_dataframe(data: pd.DataFrame, tokenizer: ByteLevelBPETokenizer) -> pd.DataFrame:
    if "text" not in data.columns:
        raise ValueError("Pas de colonne 'text' dans le DataFrame")
    data = data.copy()
    input_ids = pd.Series([tokenizer.encode(text).ids for text in data["text"]], index=data.index)
    data.insert(0, "input_ids", input_ids)
    return data

