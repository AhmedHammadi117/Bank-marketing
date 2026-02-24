# Limites et hypothèses

##  Data leakage : variable `duration`

### Le problème

La variable **`duration`** (durée d'appel en secondes) est très prédictive mais **connue uniquement après l'appel**.

**Impact** :
- ✅ Légitime pour analyse post-campagne
- ❌ Irréaliste pour ciblage pré-appel (la durée n'est pas encore connue)

### Solution

**Deux scénarios** :
1. **Modèle post-appel** (avec `duration`) : meilleure performance, usage limité
2. **Modèle pré-appel** (sans `duration`) : plus réaliste pour production

 Dans ce projet, le modèle **inclut `duration`** pour montrer la performance maximale. En production, il faudrait l'exclure.

---

## Déséquilibre de classes

- **11,3%** de conversions (classe positive)
- **88,7%** de non-conversions (classe négative)

**Conséquence** : L'accuracy seule est trompeuse. Privilégier F1-Score, Recall, Precision.

**Solutions appliquées** :
- `class_weight='balanced'` dans les modèles
- Métriques adaptées (F1, Precision, Recall)

---

## Valeurs manquantes

Certaines variables contiennent beaucoup de valeurs `unknown` :
- `education` : ~17%
- `default`, `housing`, `loan` : ~20%

**Traitement** :
- Catégorielles : imputation par catégorie "missing"
- Numériques : imputation par médiane

---

## Limites temporelles

**Période du dataset** : 2008-2010 (crise financière)

Les comportements clients peuvent avoir changé :
- Contexte économique différent
- Évolution des habitudes bancaires

**Recommandation** : Ré-entraîner le modèle régulièrement avec données récentes.

---

## Hypothèses

| Hypothèse | Impact |
|-----------|--------|
| Indépendance des observations | Pas de regroupement familial |
| Pas de feature selection avancée | Risque de multicolinéarité |
| Hyperparamètres par défaut | Performance non optimisée |
| Dataset représentatif (Portugal 2008-2010) | Généralisation limitée |

---

## Ce qui n'est pas fait

- Optimisation hyperparamètres (GridSearch)
- Techniques de rééchantillonnage (SMOTE, ADASYN)
- Interprétabilité avancée (SHAP, LIME)
- Feature engineering complexe
- Validation croisée k-fold

Ces améliorations sont possibles pour augmenter les performances.
