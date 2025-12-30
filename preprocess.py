

# preprocess.py
# Nettoyage et création des features principales pour le projet bancaire
# Pipeline simple et adapté a notre dataset

import pandas as pd
import numpy as np

def standardize_missing(df):
    # Remplace les valeurs manquantes classiques par NaN
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].replace(['unknown', 'nonexistent', 'NA', 'N/A'], np.nan)
    return df

def feature_engineering(df):
    # Ajoute les features principales utilisées pour la prédiction
    df['id'] = df.index + 1
    df['age_bin'] = pd.cut(df['age'], bins=[0,25,35,50,65,120], labels=['<25','25-34','35-49','50-64','65+'], include_lowest=True)
    df['pdays_never_contacted'] = (df['pdays'] == 999).astype(int)
    df['pdays_since'] = df['pdays'].replace(999, np.nan)
    df['duration_log1p'] = np.log1p(df['duration'])
    month_map = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
    df['month_num'] = df['month'].map(month_map)
    df['quarter'] = pd.to_datetime(df['month_num'], format='%m', errors='coerce').dt.quarter
    df['season'] = df['quarter'].map({1:'winter',2:'spring',3:'summer',4:'autumn'})
    df['young_short_call'] = ((df['age'] < 35) & (df['duration'] < df['duration'].median())).astype(int)
    edu_order = ['illiterate','basic.4y','basic.6y','basic.9y','high.school','university.degree','professional.course']
    df['education_ord'] = df['education'].astype(pd.CategoricalDtype(categories=edu_order, ordered=True)).cat.codes.replace(-1,np.nan)
    df['y_bin'] = df['y'].map({'yes':1,'no':0})
    return df

if __name__ == '__main__':
    # Charger les données
    df = pd.read_csv('data/bank-additional-full.csv', sep=';')
    df = standardize_missing(df)
    df = feature_engineering(df)
    # Sauvegarder le fichier prêt
    df.to_csv('outputs/up_data.csv', index=False)
    print('Fichier outputs/up_data.csv créé.')
