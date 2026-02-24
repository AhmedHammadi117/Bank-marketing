# Optimisation de Campagne Marketing - Prédiction de Conversion

> Projet data end-to-end : prédiction de la propension d'un client à souscrire une offre bancaire

---

##  Contexte

Dans le secteur bancaire, les campagnes marketing par téléphone coûtent cher. Optimiser le ciblage permet de réduire les coûts tout en améliorant le taux de conversion.

**Objectif** : Prédire si un client va souscrire à un dépôt à terme.

**Dataset** : 41 188 clients d'une banque portugaise, 20 variables, taux de conversion 11,3%

---

##  Structure du projet

```
├── README.md
├── requirements.txt
├── 00_Documentation/          Documentation métier
│   ├── Contexte_metier.md     Enjeux business
│   ├── Limites.md             Data leakage, limites
│   └── Impact_financier.md    Simulation ROI
│
├── 01_Donnees/
│   ├── raw/                   Données brutes
│   └── processed/             Données prétraitées
│
├── 02_Notebooks/
│   └── EDA.ipynb              Exploration, visualisations
│
├── 03_Pipeline/
│   └── preprocess.py          Nettoyage, feature engineering
│
├── 04_Modelisation/
│   └── train_model.py         Entraînement, comparaison modèles
│
└── 05_Resultats/
    ├── model.pkl              Meilleur modèle
    └── metrics.txt            Métriques
```

---

## Utilisation

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Exploration (optionnel)
```bash
jupyter notebook 02_Notebooks/EDA.ipynb
```

### 3. Prétraitement
```bash
python 03_Pipeline/preprocess.py
```

### 4. Modélisation
```bash
python 04_Modelisation/train_model.py
```

---

##  Démarche

1. **Exploration** : Statistiques, visualisations, segmentation clients
2. **Prétraitement** : Nettoyage, feature engineering métier
3. **Modélisation** : Comparaison de 3 modèles
4. **Évaluation** : Métriques adaptées au déséquilibre

---

##  Résultats

| Modèle              | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 0.86     | 0.44      | 0.90   | 0.59     |
| Random Forest       | 0.91     | 0.64      | 0.46   | 0.53     |
| **XGBoost**         | **0.88** | **0.49**  | **0.85** | **0.62** |

**Modèle sélectionné** : XGBoost (meilleur F1-Score)

**Stratégie** : Privilégier le Recall pour maximiser les conversions captées (85% avec XGBoost).

**Features importantes** :
- Historique de campagnes précédentes
- Profil socio-économique (âge, éducation, profession)
- Variables temporelles

---

##  Impact financier

**Sans modèle** : 10k clients → 50k€ coûts → 1 130 conversions → ROI 13%

**Avec modèle XGBoost** (top 30%, Recall 85%) : 3k clients → 15k€ coûts → 721 conversions → **ROI 140%**

- Réduction coûts : **-70%**
- Conversions captées : **+6%** (meilleur Recall)
- Profit net : **x3,3**

Détails : [Impact_financier.md](00_Documentation/Impact_financier.md)

---

##  Limites

- **Variable `duration`** : connue après l'appel (data leakage pour ciblage pré-contact)
- **Déséquilibre de classes** : 11% de conversions
- **Données 2008-2010** : contexte économique différent

Voir [Limites.md](00_Documentation/Limites.md)

---

## Compétences démontrées

- Analyse exploratoire et visualisations
- Préparation de données (nettoyage, feature engineering)
- Machine learning (classification, déséquilibre)
- Vision business (ROI, recommandations)
- Documentation claire

---

## Documentation

- [Contexte_metier.md](00_Documentation/Contexte_metier.md) - Enjeux, KPIs
- [Limites.md](00_Documentation/Limites.md) - Data leakage, hypothèses
- [Impact_financier.md](00_Documentation/Impact_financier.md) - Simulation ROI

---

**Secteurs** : Banque, Assurance, Marketing


