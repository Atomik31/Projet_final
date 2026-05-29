"""
train_model_engineering.py — Expérimentation avec feature engineering temporel
Features : lags + rolling stats sur les capteurs clés
Usage    : recherche / amélioration future — NE PAS promouvoir en production sans adapter le dashboard
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv

load_dotenv()

MLFLOW_TRACKING_URI = "https://atomik31-mlflow-cdsd.hf.space"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("windscan-research")

# --- 1. CHARGEMENT ET FEATURE ENGINEERING ---
print(">>> Chargement des données...")
df = pd.read_csv('../data/raw/wind_turbine_maintenance_data.csv')

dates = pd.date_range(start='2022-01-01', periods=len(df)//2, freq='H')
df.loc[df['Turbine_ID'] == 1, 'Timestamp'] = dates
df.loc[df['Turbine_ID'] == 2, 'Timestamp'] = dates
df = df.sort_values(by=['Turbine_ID', 'Timestamp'])

print(">>> Génération des features temporelles (lags + rolling stats)...")
cols_cibles = ['Vibration_Level_mmps', 'Gearbox_Oil_Temp_C', 'Generator_Bearing_Temp_C']
for col in cols_cibles:
    df[f'{col}_Lag1']    = df.groupby('Turbine_ID')[col].shift(1)
    df[f'{col}_Lag2']    = df.groupby('Turbine_ID')[col].shift(2)
    df[f'{col}_Mean_6h'] = df.groupby('Turbine_ID')[col].transform(lambda x: x.rolling(6).mean())
    df[f'{col}_Std_6h']  = df.groupby('Turbine_ID')[col].transform(lambda x: x.rolling(6).std())
    df[f'{col}_Mean_24h']= df.groupby('Turbine_ID')[col].transform(lambda x: x.rolling(24).mean())

df = df.dropna().reset_index(drop=True)
df = df[df['Turbine_ID'] == 1].reset_index(drop=True)

EXCLUDE   = {'Turbine_ID', 'Timestamp', 'Maintenance_Label'}
FEATURES  = [c for c in df.columns if c not in EXCLUDE]
TARGET    = 'Maintenance_Label'

X = df[FEATURES]
y = df[TARGET]

split_idx = int(len(X) * 0.80)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# --- 2. TOURNOI DES MODÈLES ---
models = {
    "Baseline_LogReg":     Pipeline([('scaler', StandardScaler()),
                                     ('model',  LogisticRegression(class_weight='balanced', random_state=42))]),
    "Challenger_Lasso":    Pipeline([('scaler', StandardScaler()),
                                     ('model',  LogisticRegression(penalty='l1', solver='liblinear',
                                                                    C=0.5, class_weight='balanced', random_state=42))]),
    "Champion_RandomForest": Pipeline([('scaler', StandardScaler()),
                                       ('model',  RandomForestClassifier(n_estimators=100, max_depth=15,
                                                                          class_weight='balanced', random_state=42))]),
}

print("\n>>> Entraînement et tracking MLflow (experiment: windscan-research)...")
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    f1 = f1_score(y_test, pipe.predict(X_test), average="weighted", zero_division=0)

    with mlflow.start_run(run_name=name):
        mlflow.log_param("n_features", len(FEATURES))
        mlflow.log_param("feature_engineering", "lags + rolling_stats")
        mlflow.log_metric("f1_weighted", f1)
        mlflow.sklearn.log_model(pipe, "model")

    print(f"  {name:<30} F1 weighted : {f1:.4f}")

print("\nRuns trackés dans l'experiment 'windscan-research' sur MLflow.")
print("Pour promouvoir un modèle en production, adapter d'abord le dashboard")
print("pour calculer les features temporelles à la volée.")
