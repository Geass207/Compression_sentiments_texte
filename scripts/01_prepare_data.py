import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_data, keep_binary_sentiment


#chargement du jeu de données 
train, validation, test = load_data("Sp1786/multiclass-sentiment-analysis-dataset")

#suppression de la classe neutre
train = keep_binary_sentiment(train)
validation = keep_binary_sentiment(validation)
test = keep_binary_sentiment(test)

#sauvegarde
train.to_pickle("artifacts/train_clean.pkl")
validation.to_pickle("artifacts/validation_clean.pkl")
test.to_pickle("artifacts/test_clean.pkl")
