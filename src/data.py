from datasets import load_dataset
import pandas as pd 


def load_data(dataset_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 
    data = load_dataset(dataset_name); 
    train = pd.DataFrame(data["train"])
    validation = pd.DataFrame(data["validation"])
    test = pd.DataFrame(data["test"])
    return train, validation, test 


def keep_binary_sentiment(dataset: pd.DataFrame) -> pd.DataFrame:
    if "sentiment" not in dataset.columns:
        raise ValueError("Le dataset n'a pas de colonne 'sentiment'")

    values = dataset["sentiment"].unique()

    if "positive" not in values or "negative" not in values:
        raise ValueError("La colonne 'sentiment' doit contenir 'positive' et 'negative'")

    dataset = dataset[dataset["sentiment"].isin(["positive", "negative"])].copy()

    dataset["sentiment"] = dataset["sentiment"].map({
        "negative": 0,
        "positive": 1,
    })

    return dataset