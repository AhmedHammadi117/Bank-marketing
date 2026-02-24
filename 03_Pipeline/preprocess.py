# preprocess.py
# Nettoyage et création des features pour le projet bancaire

import pandas as pd
import numpy as np

def standardize_missing(df):
    """Remplace les valeurs manquantes par NaN"""
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].replace(['unknown', 'nonexistent', 'NA', 'N/A'], np.nan)
    return df

def feature_engineering(df):
    """Crée les features métier"""
    df['id'] = df.index + 1
    
    # Catégorisation âge
    df['age_bin'] = pd.cut(df['age'], bins=[0,25,35,50,65,120], 
                           labels=['<25','25-34','35-49','50-64','65+'], include_lowest=True)
    
    # Historique de contact
    df['pdays_never_contacted'] = (df['pdays'] == 999).astype(int)
    df['pdays_since'] = df['pdays'].replace(999, np.nan)
    
    # Duration (attention : variable connue après l'appel)
    df['duration_log1p'] = np.log1p(df['duration'])
    
    # Variables temporelles
    month_map = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
                 'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
    df['month_num'] = df['month'].map(month_map)
    df['quarter'] = ((df['month_num'] - 1) // 3 + 1)
    df['season'] = df['quarter'].map({1:'hiver',2:'printemps',3:'été',4:'automne'})
    
    # Feature interaction
    df['young_short_call'] = ((df['age'] < 35) & (df['duration'] < df['duration'].median())).astype(int)
    
    # Education ordinale
    edu_order = ['illiterate','basic.4y','basic.6y','basic.9y','high.school','university.degree','professional.course']
    df['education_ord'] = df['education'].astype(pd.CategoricalDtype(categories=edu_order, ordered=True)).cat.codes.replace(-1,np.nan)
    
    # Variable cible
    df['y_bin'] = df['y'].map({'yes':1,'no':0})
    
    return df

if __name__ == '__main__':
    print("Chargement des données...")
    df = pd.read_csv('01_Donnees/raw/bank-additional-full.csv', sep=';')
    
    print("Nettoyage...")
    df = standardize_missing(df)
    
    print("Création des features...")
    df = feature_engineering(df)
    
    print("Sauvegarde...")
    df.to_csv('05_Resultats/up_data.csv', index=False)
    print(f"✓ Fichier 05_Resultats/up_data.csv créé ({len(df)} lignes)")
