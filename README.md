# Compression de Modèles NLP : Word2Vec & Classifieurs de Sentiment

Ce projet explore et compare différentes stratégies de compression de modèles de traitement automatique du langage naturel (NLP). Il applique notamment la méthode d'élagage de second ordre Optimal Brain Damage (OBD) à la fois de manière structurée sur les dimensions d'un modèle Word2Vec (CBOW) et de manière non structurée sur des têtes de classification (Sentiment Heads).

---

## Structure du Projet

Compression/
├── artifacts/              # Fichiers générés (checkpoints, datasets encodés, métriques CSV)
├── figures/                # Visualisations et courbes générées (PNG)
├── backups/                # Archives horodatées des runs précédents
├── scripts/                # Pipeline d'exécution séquentiel
│   ├── 01_prepare_data.py
│   ├── 02_tokenize.py
│   ├── 03_train_word2vec_variants.py
│   ├── 04_train_32_combinations.py
│   └── 05_visualize_results.py
├── src/                    # Code source modulaire
│   ├── classifier.py       # Architecture et entraînement des Sentiment Heads
│   ├── pruning.py          # Fonctions OBD, calcul d'Hessienne et boucles d'élagage
│   ├── visualization.py    # Fonctions de tracé de courbes (Loss, ROC)
│   └── word2vec.py         # Implémentation du CBOW et Negative Sampling
├── requirements.txt        # Dépendances Python
├── run_all.sh              # Script Bash d'automatisation
└── README.md

---

## Installation

1. Créer et activer un environnement virtuel :
   python -m venv .venv
   source .venv/bin/activate

2. Installer les dépendances :
   pip install -r requirements.txt

---

## Exécution

### 1. Exécution automatique complète (Recommandée)

Le script `run_all.sh` gère l'archivage automatique du run précédent (dans `backups/run_YYYYMMDD_HHMMSS/`) pour éviter d'écraser les résultats, puis lance toutes les étapes dans l'ordre :

   chmod +x run_all.sh
   ./run_all.sh

Pour lancer le pipeline en tâche de fond (sans interruption si fermeture du terminal) :
   nohup ./run_all.sh > run.log 2>&1 &
   tail -f run.log

---

### 2. Exécution manuelle étape par étape

Si tu souhaites exécuter ou débugger chaque script individuellement depuis la racine du projet :

   # Étape 1 : Nettoyage et séparation des données
   python scripts/01_prepare_data.py

   # Étape 2 : Entraînement du tokenizer et encodage du corpus
   python scripts/02_tokenize.py

   # Étape 3 : Entraînement Word2Vec & Pruning CBOW (W1 à W4)
   python scripts/03_train_word2vec_variants.py

   # Étape 4 : Entraînement et élagage des classifieurs (32 combinaisons M1 à M4)
   python scripts/04_train_32_combinations.py

   # Étape 5 : Génération des graphiques et courbes de performance
   python scripts/05_visualize_results.py

---

## Variantes & Modèles Évalués

### Word2Vec (CBOW)
- W1 : Modèle complet d'origine (dim = 200).
- W2 : Modèle élagué via OBD (dim = 56, sans réentraînement).
- W3 : Modèle élagué via OBD (dim = 56, avec réentraînement/cicatrisation).
- W4 : Modèle directement entraîné avec la dimension finale (dim = 56).

### Têtes de classification (Sentiment Heads)
- M1 : Architecture complète de référence.
- M2 : Modèle élagué non structuré (masqué sans réentraînement).
- M3 : Modèle élagué avec réentraînement itératif (Lottery Ticket Hypothesis).
- M4 : Modèle directement entraîné avec la capacité réduite cible.

---

## Métriques et Évaluations Clés

- F1-score vs Nombre de paramètres effectifs.
- Loss Ajustée : Pénalisation de la complexité via Loss_ajustee = Loss + alpha * (P / P_max).
