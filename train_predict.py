# train_predict.py
# Entraîne un modèle pour prédire l'acceptation d'une offre bancaire

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib

# Charger les données prêtes
df = pd.read_csv('outputs/up_data.csv')

# Sélection des features (on enlève id, y, y_bin)
X = df.drop(columns=['id', 'y', 'y_bin'], errors='ignore')
y = df['y_bin']

# Séparer en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline de prétraitement : imputation, standardisation, one-hot encoding
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

# test de plusieurs modèles pour améliorer la précision et la robustesse 
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'RandomForest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=(len(y)-sum(y))/sum(y))
}

results = {}
for name, clf in models.items():
    pipe = Pipeline([
        ('preproc', preproc),
        ('clf', clf)
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    results[name] = {'accuracy': acc, 'precision': prec, 'recall': rec}
    print(f"{name} - Accuracy: {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f}")

# Sauvegarde du meilleur modèle
best_model = max(results, key=lambda k: results[k]['precision'])
best_pipe = None
for name, clf in models.items():
    if name == best_model:
        best_pipe = Pipeline([
            ('preproc', preproc),
            ('clf', clf)
        ])
        best_pipe.fit(X_train, y_train)
        break
if best_pipe is not None:
    joblib.dump(best_pipe, 'outputs/model.pkl')
    print(f"\nMeilleur modèle (précision): {best_model}")
    with open('outputs/metrics.txt', 'w') as f:
        f.write(f"Accuracy: {results[best_model]['accuracy']}\nPrecision: {results[best_model]['precision']}\nRecall: {results[best_model]['recall']}\n")
    joblib.dump(best_pipe, f'outputs/model_{best_model}.pkl')
else:
    print("Erreur : impossible de sauvegarder le meilleur modèle.")

