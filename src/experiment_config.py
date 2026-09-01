"""Configuration centrale du pipeline expérimental.

La configuration donne volontairement plus de poids paramétrique aux têtes de
classification qu'au backbone Word2Vec utilisé à l'inférence. Cela rend l'effet
du pruning visible à la fois sur la tête seule et sur le pipeline complet.
"""

DATASET_NAME = "Sp1786/multiclass-sentiment-analysis-dataset"

VOCAB_SIZE = 1_000
MIN_TOKEN_FREQUENCY = 2

EMBEDDING_DIM = 100
CBOW_WINDOW_SIZE = 2
NEGATIVE_SAMPLES = 5
WORD2VEC_BATCH_SIZE = 512
WORD2VEC_EPOCHS = 5
WORD2VEC_LEARNING_RATE = 1e-3

HEAD_ARCHITECTURES = {
    "H1": {"full": (1_024,), "direct": (512,)},
    "H2": {"full": (1_024, 128), "direct": (512, 128)},
}
CLASSIFIER_BATCH_SIZE = 256
CLASSIFIER_EPOCHS = 30
CLASSIFIER_LEARNING_RATE = 1e-3
CLASSIFIER_PATIENCE = 4

OBD_HESSIAN_VECTORS = 10
OBD_PRUNING_STEP = 0.10
OBD_RETRAINING_EPOCHS = 3

MINT_GROUPS_PER_LAYER = (10, 16, 8)
MINT_SAMPLES_PER_CLASS = 96
MINT_PARTITION_SIZE = 64
MINT_MAXIMUM_PRUNING_FRACTION = 0.80
MINT_RETRAINING_EPOCHS = 10

SEED = 42

