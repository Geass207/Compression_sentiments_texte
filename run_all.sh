#!/bin/bash
set -e

# Si des résultats existent déjà, on les archive avec la date/heure
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

if [ -d "artifacts" ] && [ "$(ls -A artifacts)" ]; then
    echo "Archivage du run précédent vers backups_${TIMESTAMP}..."
    mkdir -p "backups/run_${TIMESTAMP}"
    mv artifacts "backups/run_${TIMESTAMP}/artifacts"
    [ -d "figures" ] && mv figures "backups/run_${TIMESTAMP}/figures"
fi

mkdir -p artifacts figures

echo "=== [1/5] Préparation des données ==="
python scripts/01_prepare_data.py

echo "=== [2/5] Tokenisation ==="
python scripts/02_tokenize.py

echo "=== [3/5] Word2Vec & Pruning CBOW ==="
python scripts/03_train_word2vec_variants.py

echo "=== [4/5] Classifieurs ==="
python scripts/04_train_32_combinations.py

echo "=== [5/5] Visualisations ==="
python scripts/05_visualize_results.py

echo "Nouveau run terminé !"