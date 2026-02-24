# train_model.py
# Entraîne et compare plusieurs modèles de classification

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# XGBoost optionnel
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠ XGBoost non disponible (pip install xgboost)")

# Charger les données
print("Chargement des données...")
df = pd.read_csv('05_Resultats/up_data.csv')

X = df.drop(columns=['id', 'y', 'y_bin'], errors='ignore')
y = df['y_bin']

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# Pipeline de prétraitement
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preproc = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

# Modèles à tester
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'RandomForest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
}

if HAS_XGBOOST:
    scale = (len(y) - sum(y)) / sum(y)
    models['XGBoost'] = XGBClassifier(use_label_encoder=False, eval_metric='logloss', 
                                       scale_pos_weight=scale, random_state=42)

# Entraînement et évaluation
print("\nEntraînement des modèles...")
results = {}
trained_models = {}

for name, clf in models.items():
    print(f"\n{name}:")
    pipe = Pipeline([('preproc', preproc), ('clf', clf)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    results[name] = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}
    trained_models[name] = pipe
    
    print(f"  Accuracy:  {acc:.3f}")
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall:    {rec:.3f}")
    print(f"  F1-Score:  {f1:.3f}")

# Sauvegarder le meilleur modèle (selon F1-Score)
best_model = max(results, key=lambda k: results[k]['f1'])
best_pipe = trained_models[best_model]

joblib.dump(best_pipe, '05_Resultats/model.pkl')
print(f"\n✓ Meilleur modèle sauvegardé: {best_model} (F1={results[best_model]['f1']:.3f})")

# Sauvegarder les métriques
with open('05_Resultats/metrics.txt', 'w', encoding='utf-8') as f:
    f.write(f"Meilleur modèle: {best_model}\n\n")
    for name, metrics in results.items():
        f.write(f"{name}:\n")
        f.write(f"  Accuracy:  {metrics['accuracy']:.4f}\n")
        f.write(f"  Precision: {metrics['precision']:.4f}\n")
        f.write(f"  Recall:    {metrics['recall']:.4f}\n")
        f.write(f"  F1-Score:  {metrics['f1']:.4f}\n\n")
print("✓ Métriques sauvegardées dans 05_Resultats/metrics.txt")
