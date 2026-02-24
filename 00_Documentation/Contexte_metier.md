# Contexte métier

## Problématique

Dans le secteur bancaire, les **campagnes de télémarketing** coûtent cher :
- Temps agent (salaire, formation)
- Infrastructure téléphonique et CRM
- Faible taux de conversion (5-15% en moyenne)

**Objectif** : Optimiser le ciblage pour contacter uniquement les clients à fort potentiel de conversion.

---

## Cas d'usage

### Ciblage pré-appel (recommandé)
Avant de lancer la campagne, scorer tous les clients et ne contacter que les top 20-30%.

**Avantages** :
- Réduction des coûts (moins de contacts)
- Meilleur taux de conversion
- Moins de fatigue client

**Contrainte** : Ne pas utiliser la variable `duration` (durée d'appel inconnue avant contact).

### Analyse post-campagne
Analyser les facteurs de conversion pour améliorer les campagnes futures.

---

## Dataset

**Source** : [Bank Marketing Dataset (UCI)](https://www.kaggle.com/code/benroshan/bank-marketing-campaign-predictive-analytics)

**Description** :
- 41 188 clients d'une banque portugaise
- Campagne télémarketing pour dépôt à terme
- Période : 2008-2010
- Taux de conversion : 11,3%

**Variables** :
- Socio-démographiques : âge, profession, éducation, situation
- Financières : défaut de paiement, prêt immobilier, prêt personnel
- Historique : nombre de contacts, campagnes précédentes
- Macroéconomiques : taux emploi, indice des prix

---

## KPIs métier

| KPI | Description | Objectif |
|-----|-------------|----------|
| **Taux de contact** | % clients contactés | ↓ -50% |
| **Taux de conversion** | % conversions / contactés | ↑ +30% |
| **Coût par conversion** | Coût total / conversions | ↓ -40% |
| **ROI campagne** | Profit / Coût total | ↑ x5-10 |

---

## Secteurs applicables

- Banque : prêts, cartes de crédit, assurance-vie
- Assurance : souscription, renouvellement
- Télécom : upgrade forfait, fidélisation
- E-commerce : ciblage email marketing
