#!/bin/bash
set -e

if [ -x ".venv/bin/python" ]; then
    PYTHON_EXECUTABLE=".venv/bin/python"
else
    PYTHON_EXECUTABLE="python"
fi

# Si des résultats existent déjà, on les archive avec la date/heure
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

if [ -d "artifacts" ] && [ "$(ls -A artifacts)" ]; then
    echo "Archivage du run précédent vers backups_${TIMESTAMP}..."
    mkdir -p "backups/run_${TIMESTAMP}"
    mv artifacts "backups/run_${TIMESTAMP}/artifacts"
    [ -d "figures" ] && mv figures "backups/run_${TIMESTAMP}/figures"
fi

mkdir -p artifacts figures

echo "=== [1/6] Préparation des données ==="
"${PYTHON_EXECUTABLE}" scripts/01_prepare_data.py

echo "=== [2/6] Tokenisation ==="
"${PYTHON_EXECUTABLE}" scripts/02_tokenize.py

echo "=== [3/6] Entraînement du Word2Vec W1 complet (sans pruning) ==="
"${PYTHON_EXECUTABLE}" scripts/03_train_word2vec_variants.py

echo "=== [4/6] Entraînement et benchmark OBD des classifieurs ==="
"${PYTHON_EXECUTABLE}" scripts/04_train_32_combinations.py

echo "=== [5/6] Pruning MINT de la tête W1/H2 ==="
"${PYTHON_EXECUTABLE}" scripts/05_mint.py

echo "=== [6/6] Visualisations ==="
MPLCONFIGDIR=/tmp/compression-matplotlib "${PYTHON_EXECUTABLE}" scripts/06_visualize_results.py

echo "Nouveau run terminé !"
