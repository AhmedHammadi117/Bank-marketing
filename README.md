1. Objectif du projet

Ce repository correspond au Sous-Projet A d’un ensemble de trois projets autour d’un cas réel de marketing bancaire (prédiction de conversion client).

Objectif de ce repo :
Nettoyer, analyser et transformer les données brutes pour produire un dataset propre, exploitable par des modèles de Machine Learning.

Ce travail représente ce qu’effectuerait un Data Engineer / Data Scientist junior dans une entreprise :

lecture robuste des données

nettoyage & standardisation

création de features utiles

documentation claire des variables

export de jeux de données segmentés

Ce repository est entièrement autonome, mais il est utilisé comme base pour les repos suivants.

2. Structure du repository

repo-1-data-features/
│
├── data/
│ └── bank-additional-full.csv
│
├── notebooks/
│ └── 01_data_features.ipynb
│
├── outputs/
│ ├── up_data.csv
│ ├── profil_cols.csv
│ ├── campagne_cols.csv
│ ├── macro_cols.csv
│ ├── columns_summary.csv
│ └── README_outputs.md
│
├── src/
│ └── features.py
│
├── tests/
│ └── test_features.py
│
├── requirements.txt
└── README.md

3. Données utilisées

Les données proviennent du dataset Bank Marketing de l’UCI.

Chaque ligne représente un contact avec un client.
La variable cible :

yes = le client a souscrit

no = le client n’a pas souscrit

Dataset : 41 188 lignes et 21 variables.

4. Étapes principales du notebook
Chargement robuste

Gestion automatique du séparateur

Ajout d’un identifiant unique (id)

Inspection (info, describe, head)

Nettoyage

Remplacement des valeurs “unknown” et “nonexistent” par NaN

Correction des valeurs extrêmes

Vérification des catégories

Segmentation des colonnes

profil client (age, job, marital…)

campagne marketing (duration, contact, campaign…)

macroéconomie (euribor, cons.conf.idx, etc.)

Feature Engineering

Création d’un âge catégorisé

Log-transform sur duration

Gestion intelligente de pdays

Mois → numéro → trimestre → saison

Encodage ordinal éducation

Variable cible binaire (y_bin)

Feature interaction (young_short_call)

Export final

Dataset enrichi : up_data.csv

Exports segmentés

Table de documentation des colonnes

5. Installation et exécution

Cloner le repo
git clone https://github.com/TON_COMPTE/repo-1-data-features.git

cd repo-1-data-features

Installer les dépendances
pip install -r requirements.txt

Lancer le notebook
jupyter notebook notebooks/01_data_features.ipynb

6. Exportations produites

up_data.csv : dataset final enrichi
profil_cols.csv : caractéristiques clients
campagne_cols.csv : données campagne et interactions
macro_cols.csv : variables macro
columns_summary.csv : documentation automatique des colonnes

Un fichier README_outputs.md dans /outputs décrit chaque fichier.

7. Tests unitaires

Les fonctions de preprocessing sont dans src/features.py.

Lancer les tests :
pytest tests/test_features.py

8. Relation avec les autres repositories

Repo 1 – Data & Features (ce repo)
→ Préparation des données

Repo 2 – Modeling
→ Modèles, comparatif, SHAP, optimisation

Repo 3 – Production
→ API FastAPI, Dashboard Streamlit/Power BI

9. Compétences démontrées

Ce repo montre une vraie compétence professionnelle :

architecture projet propre

pipeline clair

FE structuré

documentation automatique

tests unitaires

exports standardisés

maîtrise Python et Git

10. Prochaines étapes

Repo 2 – Modeling utilisera :

up_data.csv

profil_cols.csv

campagne_cols.csv

macro_cols.csv