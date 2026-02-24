# Impact financier et ROI

## Scénario de référence

### Sans modèle (contact de tous les clients)

| Paramètre | Valeur |
|-----------|--------|
| Clients contactés | 10 000 |
| Coût par contact | 5€ |
| Coût total | **50 000€** |
| Taux de conversion | 11,3% |
| Conversions | 1 130 |
| Revenu (50€/conversion) | 56 500€ |
| **Profit net** | **6 500€** |
| **ROI** | **13%** |

---

## Avec modèle prédictif (XGBoost)

### Stratégie : Cibler top 30% des scores

Basé sur les performances du modèle XGBoost (Recall 85%).

| Paramètre | Valeur |
|-----------|--------|
| Clients contactés | 3 000 (-70%) |
| Coût total | **15 000€** |
| Conversions captées | ~721 (85% × 30% × 1130 × 2,83) |
| Revenu | 36 050€ |
| **Profit net** | **21 050€** |
| **ROI** | **140%** |

### Gains

| Indicateur | Amélioration |
|------------|--------------|
| Coûts | **-70%** (50k → 15k€) |
| Profit net | **+224%** (+14 550€) |
| ROI | **x10,8** (13% → 140%) |
| Coût par conversion | **-52%** (44€ → 21€) |
### Campagne 500 000 clients

**Sans modèle** :
- Coût : 2 500 000€
- Conversions : 56 500
- Profit : 325 000€

**Avec modèle XGBoost (top 30%)** :
- Coût : 750 000€
- Conversions : 36 050 (Recall 85%)
- Profit : **1 052 500€**

### Gain : +727 500€ par campagne

Si 4 campagnes/an : **+2,91 millions €/an**

---

## Optimisation du seuil

| Seuil | Contactés | Coût | Conversions | Profit | ROI |
|-------|-----------|------|-------------|--------|-----|
| Top 50% | 5 000 | 25 000€ | 902 | 20 100€ | 80% |
| **Top 30%** | **3 000** | **15 000€** | **721** | **21 050€** | **140%** |
| Top 20% | 2 000 | 10 000€ | 541 | 17 050€ | 171% |
| Top 10% | 1 000 | 5 000€ | 301 | 10 050€ | 201% |

**Seuil optimal** : Top 30% (meilleur compromis profit/volume).

---

## Autres bénéfices

- **Réduction fatigue client** : moins de sollicitations
- **Productivité agents** : concentration sur profils chauds
- **Amélioration continue** : modèle qui s'améliore avec nouvelles données
- **Différenciation concurrentielle** : approche data-driven

---

## Formules

**ROI** = (Profit net / Coût total) × 100

**Profit net** = (Nb conversions × Marge) - Coût total

**Coût par conversion** = Coût total / Nb conversions

---

**Note** : Simulation basée sur hypothèses réalistes mais simplifiées. Test A/B recommandé avant déploiement.
