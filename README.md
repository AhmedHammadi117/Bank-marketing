


# Projet Data Science bancaire — Prédiction de l’acceptation d’une offre

Ce projet présente une démarche complète de data science appliquée à un jeu de données bancaire réel et volumineux.
L’objectif principal est de prédire si un client acceptera ou non une offre bancaire, à partir de ses caractéristiques socio-démographiques et comportementales.
Le projet met l’accent sur la structuration du pipeline, la reproductibilité et la clarté des choix analytiques, plutôt que sur l’optimisation extrême des performances.

## Objectifs du projet

- Prédire l’acceptation d’une offre bancaire (problème de classification binaire).
- Analyser et comprendre les facteurs influençant la décision des clients.
- Mettre en place un pipeline clair de prétraitement et de modélisation.
- Séparer l’analyse exploratoire de l’automatisation.
- Garantir des résultats reproductibles et explicables.

## Points clés

- Approche métier : problématique réaliste de marketing bancaire.
- Lisibilité : décisions analytiques justifiées dans le notebook.
- Reproductibilité : pipeline entièrement scripté.
- Simplicité maîtrisée : modèles accessibles et comparables.
- Évolutivité : structure prête pour des améliorations futures.

## Structure du projet

- `data_exploration.ipynb` : Exploration, visualisations, décisions de nettoyage et d’ingénierie des features.
- `preprocess.py` : Génère automatiquement les données prêtes (`outputs/up_data.csv`) selon les choix du notebook.
- `train_predict.py` : Entraîne et évalue plusieurs modèles de classification (Logistic Regression, Random Forest, XGBoost) sur les données prêtes.
- `outputs/` : Données prêtes, modèles, métriques, graphiques.
- `data/` : Données brutes.

## Workflow conseillé

1. **Exploration & décisions** :
   - Utiliser le notebook pour comprendre les données, valider les choix de nettoyage et de features (EDA, visualisations, décisions métier).
2. **Production des données prêtes** :
   - Lancer `preprocess.py` pour générer `outputs/up_data.csv`.
3. **Modélisation** :
   - Lancer `train_predict.py` pour entraîner et comparer plusieurs modèles. Les métriques sont sauvegardées dans `outputs/metrics.txt`.

## Attentes et limites du projet

- Le modèle fourni n’est pas optimisé pour la performance métier : il sert de base honnête, facilement améliorable selon les besoins (feature engineering avancé, tuning, modèles plus complexes, gestion du déséquilibre, etc.).
- L’objectif est de montrer une démarche complète, claire et structurée, pas d’obtenir le meilleur score possible.
- Toutes les étapes de nettoyage et de feature engineering sont décidées dans le notebook puis codées dans `preprocess.py` pour garantir la reproductibilité.
- Les scripts d’entraînement n’ajoutent aucune transformation supplémentaire non décidée en amont.
- Ce projet peut servir de base pour des travaux plus avancés ou être adapté à d’autres contextes métier.

## Pistes d’amélioration

- Optimisation des hyperparamètres (GridSearch, RandomizedSearch, etc.).
- Sélection de features avancée, création de nouvelles variables métier.
- Gestion du déséquilibre (SMOTE, class_weight, etc.).
- Interprétabilité (SHAP, LIME, etc.).
- Intégration d’un pipeline de production, API, ou dashboard pour l’industrialisation.

---

**Résumé**

Ce projet vise à prédire l’acceptation d’une offre bancaire par un client à partir de données réelles.
Il démontre une démarche complète de data science : compréhension du problème métier, exploration des données, automatisation du prétraitement, entraînement de plusieurs modèles de classification et évaluation des performances.


