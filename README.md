


# Projet Data Science bancaire — Pipeline simple, clair et évolutif

Ce projet a été conçu pour illustrer une démarche complète de data scientist sur un jeu de données bancaire réel et volumineux. L’objectif n’est pas d’obtenir le meilleur score, mais de montrer une compréhension globale du métier, la capacité à structurer un projet, et à rendre chaque étape reproductible et compréhensible.

## Points forts du projet

- **Clarté et pédagogie** : chaque étape (exploration, nettoyage, feature engineering, modélisation) est explicitée et justifiée dans le notebook, puis automatisée dans des scripts.
- **Reproductibilité** : tout le pipeline est scripté, du nettoyage à la modélisation, pour garantir que les résultats sont cohérents et faciles à rejouer.
- **Simplicité assumée** : le projet reste volontairement simple et lisible, pour servir de base un premier projet professionnel.
- **Adaptabilité** : la structure permet d’aller plus loin facilement (ajout de modèles, tuning, pipeline production, etc.).

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

## Pour aller plus loin

- Optimisation des hyperparamètres (GridSearch, RandomizedSearch, etc.).
- Sélection de features avancée, création de nouvelles variables métier.
- Gestion du déséquilibre (SMOTE, class_weight, etc.).
- Interprétabilité (SHAP, LIME, etc.).
- Intégration d’un pipeline de production, API, ou dashboard pour l’industrialisation.

---

**Résumé**

Ce projet montre une démarche complète et honnête : exploration des données, décisions claires sur le nettoyage et les features, automatisation des étapes dans un script reproductible, entraînement de plusieurs modèles simples, et explication des choix. L’objectif est de prouver une compréhension globale du métier de data scientist, tout en gardant le projet accessible, évolutif et facilement explicable.

