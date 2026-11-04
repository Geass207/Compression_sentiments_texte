# Guide expérimental

[← Présentation du projet](../README.md)

Ce guide décrit le pipeline existant et les fichiers qu’il produit. Pour consulter les résultats déjà enregistrés, voir le [rapport](../rapport/main.pdf), les [métriques agrégées](../artifacts/repeated_runs/metrics_summary_long.csv) et les [figures](../figures/experimentales/).

## Sommaire

- [Données et configuration](#données-et-configuration)
- [Variantes comparées](#variantes-comparées)
- [Installation](#installation)
- [Pipeline principal](#pipeline-principal)
- [Répétitions et témoin aléatoire](#répétitions-et-témoin-aléatoire)
- [Lire les sorties](#lire-les-sorties)

## Données et configuration

Le jeu de données configuré est `Sp1786/multiclass-sentiment-analysis-dataset`. Le pipeline conserve les classes **positive** et **negative**, encodées respectivement `1` et `0`, et utilise les partitions train, validation et test fournies. Le tokenizer est entraîné sur la partition d’entraînement.

Les paramètres sont centralisés dans [`src/experiment_config.py`](../src/experiment_config.py).

| Élément | Configuration |
| :--- | :--- |
| Tokenisation | BPE ; vocabulaire cible de 1 000 tokens |
| Représentation | Word2Vec CBOW ; dimension 100 ; negative sampling |
| W1 | Modèle complet partagé, figé pendant les expériences sur les têtes |
| H1 complète → réduite | `100 → 1024 → 1` → `100 → 512 → 1` |
| H2 complète → réduite | `100 → 1024 → 128 → 1` → `100 → 512 → 128 → 1` |
| Entraînement des têtes | Adam ; taux d’apprentissage `1e-3` ; batch de 256 ; maximum de 30 époques ; patience de 4 |
| OBD | 10 vecteurs pour l’estimation hessienne ; pas de pruning de 0,10 ; 3 époques de réentraînement par étape |
| MINT | H2 uniquement ; masque en une passe ; 10 époques de réentraînement |
| Répétitions enregistrées | Graines `42`, `43`, `44`, `45`, `46` ; W1, vecteurs et partitions fixes |

La configuration enregistrée des répétitions est disponible dans [`artifacts/repeated_runs/run_config.json`](../artifacts/repeated_runs/run_config.json).

## Variantes comparées

| Identifiant | Description |
| :--- | :--- |
| **M1** | Tête dense complète de référence |
| **M2 / OBD-M2** | Tête élaguée par OBD, évaluée avant le dernier réentraînement du calendrier itératif |
| **M3 / OBD-M3** | Même tête après le dernier réentraînement sous masque |
| **M4** | Architecture dense réduite, entraînée directement |
| **MINT-M2** | Tête H2 juste après l’application du masque MINT en une passe |
| **MINT-M3** | Même masque MINT, après réentraînement en maintenant les poids masqués à zéro |
| **RAND-M2** | Tête élaguée aléatoirement, avant le dernier réentraînement |
| **RAND-M3** | Même masque aléatoire, après le dernier réentraînement sous masque |
| **Dense-control** | Témoin dense poursuivant l’entraînement avec un budget supplémentaire correspondant à OBD ou MINT |

Les répétitions utilisent les identifiants explicites `OBD-M2` et `OBD-M3` ; les fichiers du pipeline principal les nomment `M2` et `M3`. OBD et RAND suivent un calendrier itératif : leurs variantes M2 ont donc déjà bénéficié des réentraînements intermédiaires. MINT-M2 correspond à un masque appliqué en une seule passe.

## Installation

Les commandes suivantes sont à lancer depuis la racine du dépôt, dans un terminal Bash. Les dépendances sont listées dans [`requirements.txt`](../requirements.txt), sans versions figées.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Le chargement initial des données nécessite un accès à Hugging Face. Les étapes d’entraînement constituent des expériences complètes ; elles ne sont pas nécessaires pour lire le rapport ou les résultats publiés.

## Pipeline principal

### Exécution automatique

```bash
bash run_all.sh
```

[`run_all.sh`](../run_all.sh) utilise `.venv/bin/python` si disponible, puis lance les étapes **01 à 07** dans l’ordre. Si `artifacts/` contient déjà des résultats, le script déplace ce dossier et les figures existantes vers `backups/run_YYYYMMDD_HHMMSS/` avant de créer les nouvelles sorties. Les étapes 08 et 09 sont lancées séparément.

### Détail des étapes

| Étape | Script | Rôle |
| :--- | :--- | :--- |
| 01 | [`01_prepare_data.py`](../scripts/01_prepare_data.py) | Charger les partitions et conserver les deux classes de sentiment |
| 02 | [`02_tokenize.py`](../scripts/02_tokenize.py) | Entraîner le tokenizer BPE et encoder les textes |
| 03 | [`03_train_word2vec_variants.py`](../scripts/03_train_word2vec_variants.py) | Entraîner l’unique Word2Vec W1 et produire les vecteurs |
| 04 | [`04_train_32_combinations.py`](../scripts/04_train_32_combinations.py) | Entraîner les têtes H1/H2 et comparer M1, OBD-M2/M3 et M4 |
| 05 | [`05_mint.py`](../scripts/05_mint.py) | Appliquer MINT à H2 et réentraîner sous masque |
| 06 | [`06_visualize_results.py`](../scripts/06_visualize_results.py) | Générer les courbes expérimentales |
| 07 | [`07_generate_report_figures.py`](../scripts/07_generate_report_figures.py) | Produire les figures de synthèse du rapport |

Les noms historiques des scripts 03 et 04 sont conservés : le protocole actuel utilise **un seul W1** et **deux têtes**, comme défini dans la configuration centrale.

Pour une exécution manuelle, respecter cet ordre. Les scripts réutilisent les sorties des étapes précédentes ; contrairement au lanceur, cette séquence n’archive pas les résultats existants.

```bash
mkdir -p artifacts figures
python scripts/01_prepare_data.py
python scripts/02_tokenize.py
python scripts/03_train_word2vec_variants.py
python scripts/04_train_32_combinations.py
python scripts/05_mint.py
python scripts/06_visualize_results.py
python scripts/07_generate_report_figures.py
```

## Répétitions et témoin aléatoire

### Répéter les expériences

L’étape 08 réutilise W1, les vecteurs et les partitions produits par le pipeline principal. Elle conserve les résultats par graine et calcule moyennes, écarts-types, variances, intervalles de confiance à 95 % et différences appariées dans `artifacts/repeated_runs/`.

```bash
python scripts/08_repeat_experiments.py
```

### Comparer au pruning aléatoire

L’étape 09 reconstruit les mêmes références M1 et remplace les scores OBD par une permutation aléatoire uniforme des poids. Elle conserve la sparsité cible, le calendrier itératif et le budget de réentraînement d’OBD. Les sorties sont enregistrées dans `artifacts/random_pruning/`.

```bash
python scripts/09_random_pruning.py
```

Ces deux étapes ignorent les graines déjà terminées lors d’une reprise. L’option `--rerun-completed` force leur recalcul. Pour les comparaisons appariées avec OBD et MINT, les résultats correspondants de l’étape 08 doivent être disponibles.

## Lire les sorties

| Fichier ou dossier | Contenu |
| :--- | :--- |
| [`classification_results.csv`](../artifacts/classification_results.csv) | Métriques du pipeline principal et nombres de paramètres |
| [`mint_results.csv`](../artifacts/mint_results.csv) | Référence H2 et variantes MINT |
| [`runtime_measurements.csv`](../artifacts/runtime_measurements.csv) | Mesures de coût, stockage et latence |
| [`repeated_runs/metrics_by_seed.csv`](../artifacts/repeated_runs/metrics_by_seed.csv) | Résultats individuels des répétitions |
| [`repeated_runs/metrics_summary_long.csv`](../artifacts/repeated_runs/metrics_summary_long.csv) | Statistiques agrégées par tête, modèle et métrique |
| [`repeated_runs/paired_differences_summary.csv`](../artifacts/repeated_runs/paired_differences_summary.csv) | Différences appariées entre variantes |
| [`random_pruning/`](../artifacts/random_pruning/) | Checkpoints masqués, trajectoires, métriques et comparaisons du témoin aléatoire |
| [`figures/experimentales/`](../figures/experimentales/) | Figures utilisant qualité, loss, temps ou stockage mesuré |
| [`figures/theoriques/`](../figures/theoriques/) | Figures déduites uniquement des dimensions et des masques |

Le F1 présenté dans le README correspond au **F1 binaire** (`test_f1`), distinct du macro-F1 également enregistré. Les résultats d’une exécution principale et les statistiques sur cinq graines sont présentés séparément.

La **loss ajustée** combine la loss et une pénalisation de complexité :

```text
Loss ajustée = Loss + α × (P / P_max)
```

La comparaison dédiée à MINT utilise `α = 0,4` dans [`adjusted_loss_mint_W1_H2.png`](../figures/experimentales/adjusted_loss_mint_W1_H2.png). L’interprétation dépend de la quantité retenue pour `P` : les paramètres effectifs d’un modèle masqué décrivent sa sparsité, alors que sa taille sérialisée et sa latence reflètent son exécution avec des tenseurs denses.
