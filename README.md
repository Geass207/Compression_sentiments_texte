# Compression de Modèles NLP : Word2Vec & Classifieurs de Sentiment

Ce projet explore et compare différentes stratégies de compression de modèles de traitement automatique du langage naturel (NLP). Il utilise un unique Word2Vec (CBOW) non élagué et applique la méthode Optimal Brain Damage (OBD) aux têtes de classification de sentiment.

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
│   ├── 05_mint.py
│   └── 06_visualize_results.py
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

   # Étape 3 : Entraînement du Word2Vec W1
   python scripts/03_train_word2vec_variants.py

   # Étape 4 : Entraînement et élagage des classifieurs avec W1 (M1 à M4)
   python scripts/04_train_32_combinations.py

   # Étape 5 : Pruning MINT one-shot de la tête H2 avec W1 complet
   python scripts/05_mint.py

   # Étape 6 : Génération des graphiques et courbes de performance
   python scripts/06_visualize_results.py

---

## Variantes & Modèles Évalués

### Word2Vec (CBOW)
- W1 : unique modèle complet partagé par toutes les expériences (dim = 200).

### Têtes de classification (Sentiment Heads)
- M1 : Architecture complète de référence.
- M2 : Modèle élagué non structuré (masqué sans réentraînement).
- M3 : Modèle élagué avec réentraînement itératif (Lottery Ticket Hypothesis).
- M4 : Modèle directement entraîné avec la capacité réduite cible.
- MINT-M2 : Tête H2 élaguée en une passe avec les scores de GMI conditionnelle.
- MINT-M3 : Même masque MINT, suivi d'un réentraînement avec poids masqués figés.

Le pruning MINT est appliqué uniquement à la tête H2. Le checkpoint Word2Vec
W1 reste complet en dimension 200 et n'est jamais modifié par cette étape.

---

## Métriques et Évaluations Clés

- F1-score vs Nombre de paramètres effectifs.
- Loss Ajustée : Pénalisation de la complexité via Loss_ajustee = Loss + alpha * (P / P_max).
- Comparaison dédiée de la loss ajustée entre H2-M1, MINT-M2 et MINT-M3
  dans `figures/adjusted_loss_mint_W1_H2.png` (alpha = 0.4).
