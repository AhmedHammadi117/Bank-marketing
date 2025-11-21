1. Introduction
🎯 Contexte

Les banques mènent régulièrement des campagnes marketing pour proposer des produits financiers (dépôts à terme, assurances, prêts…).
Ces campagnes sont coûteuses, nécessitent du personnel et engagent la réputation de l’établissement.

🎯 Objectif du projet

L’objectif est de construire un modèle prédictif capable d’estimer si un client acceptera une offre lors d’une campagne téléphonique.

🎯 Valeur ajoutée

Réduction des coûts marketing

Ciblage plus intelligent

Meilleure allocation des ressources

Amélioration du taux de conversion

Tests de scénarios (who to call next?)

2. Questions Clés Explorées

Quels types de clients sont les plus susceptibles d’accepter une offre ?

Quelles variables influencent le plus la décision (âge, profession, historique marketing, conditions économiques…) ?

Peut-on prédire le succès d’une campagne avant son lancement ?

Quels modèles sont les plus efficaces dans ce contexte ?

Comment éviter les fuites de données (ex: variable duration) ?

Quel seuil de décision maximise le rendement d’une campagne ?

3. Dataset
📌 Source

Dataset “Bank Marketing Campaigns” (UCI / Kaggle)

📌 Taille

~40 000 lignes
21 variables explicatives
1 variable cible (y = yes/no)

📌 Structure

Variables client : age, job, marital, education…

Variables financières : housing, loan, default

Variables liées à la campagne : month, day_of_week, campaign, previous…

Variables macro-économiques : euribor3m, cons.price.idx, emp.var.rate…

📌 Prétraitement effectué (dans le notebook)

Nettoyage du CSV

Correction de types

Gestion valeurs “unknown”

Analyse et suppression des colonnes problématiques

Transformation “pdays_clean” & création features cohérentes

Regroupement des professions (feature engineering)

Création de variables dérivées : saison, nombre de tentatives, indicateurs économiques…

4. Méthodologie (alignée avec ton notebook + pipeline futur)
4.1 Data Cleaning

Suppression doublons

Correction types

Traitement valeurs “unknown”

Nettoyage de la colonne pdays

Inspection des distributions

Vérification fuite de données

4.2 Feature Engineering

(déjà présent dans ton notebook → je me baserai dessus)

Regroupement des professions

Catégorisation âge

Transformation saisonnière

Agrégation des variables macro

Variable “is_new_contact”

Variable “pressure_marketing” (nombre d’appels cumulés)

4.3 Exploration & Visualisation

Déjà présente dans ton code :

Distributions des profils client

Analyse taux d’acceptation par âge, métier, mois, etc.

Heatmaps macro-économie / acceptation

Visualisation campagne & pression marketing

4.4 Modélisation (à venir dans la suite du notebook)

Construction d’un pipeline preprocessing

Split train/val/test stratifié

Modèle baseline : Logistic Regression

Modèles avancés : RandomForest, XGBoost, SVM

Cross-validation

Optimisation hyperparamètres

Calibration des probabilités

Optimisation du seuil de décision

4.5 Évaluation

Accuracy

Precision / Recall

F1-score

AUC-ROC

Courbes PR

Matrice de confusion

Gains business selon différents seuils

5. Résultats attendus / obtenus

Comparaison claire des modèles

Identification des features influentes (SHAP, importance)

Détection fuites (duration → exclue du modèle final)

Courbes ROC & PR

Dashboard de visualisation des résultats

Recommandations pour entreprise :

Quels clients prioriser

Quand appeler

Quel canal privilégier

Quel budget attendre

6. Ingénierie & Déploiement

(version professionnelle pour un dossier d’alternance)

6.1 Pipeline ML

Du brut → features → modèle → prédiction

6.2 Industrialisation

Scripts train.py et predict.py

Sauvegarde du modèle en .joblib

Chargement automatique des données

Tests unitaires simples

6.3 Déploiement

API FastAPI / Flask

Dashboard Streamlit pour tester la prédiction en live

Conteneurisation Docker (optionnel)

7. Conclusion & Perspectives

Résumé des résultats

Impact potentiel pour une banque réelle

Recommandations stratégiques

Améliorations futures :

Ajouter des données externes (inflation, taux de chômage)

Modèles séquentiels (RNN) pour séries temporelles

Apprentissage actif pour améliorer ciblage en continu

Détection de drift en production

8. Documentation des Variables

👉 Je garde la structure que tu as écrite
👉 MAIS je l’améliore, corrige, et j’adapte au style pro
👉 Et j’ajoute un tag : Analyse / Prédiction / Notes