"""
train_model.py — Pipeline de production (compatible dashboard)
Features : 6 capteurs bruts uniquement
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv

load_dotenv()

MLFLOW_TRACKING_URI = "https://atomik31-mlflow-cdsd.hf.space"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("windscan-production")

# --- 1. CHARGEMENT ET PRÉPARATION ---
print(">>> Chargement des données...")
df = pd.read_csv('../data/raw/wind_turbine_maintenance_data.csv')
df = df[df['Turbine_ID'] == 1].reset_index(drop=True)

FEATURES = ['Rotor_Speed_RPM', 'Wind_Speed_mps', 'Power_Output_kW',
            'Gearbox_Oil_Temp_C', 'Generator_Bearing_Temp_C', 'Vibration_Level_mmps']
TARGET = 'Maintenance_Label'

X = df[FEATURES]
y = df[TARGET]

split_idx  = int(len(X) * 0.80)
X_train    = X.iloc[:split_idx]
X_test     = X.iloc[split_idx:]
y_train    = y.iloc[:split_idx]
y_test     = y.iloc[split_idx:]

# Preprocesseur commun
preprocessor = ColumnTransformer([
    ('scaler', Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('std',     StandardScaler())
    ]), FEATURES)
])

# --- 2. TOURNOI DES MODÈLES ---
best_run = {"name": None, "f1": -1, "pipeline": None}

models = {
    "Baseline_DummyClassifier": DummyClassifier(strategy="most_frequent", random_state=42),
    "LogisticRegression":       LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    "RandomForest":             RandomForestClassifier(n_estimators=100, max_depth=10,
                                                       class_weight='balanced', random_state=42),
}

for name, estimator in models.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
    pipe.fit(X_train, y_train)
    f1 = f1_score(y_test, pipe.predict(X_test), average="macro", zero_division=0)

    with mlflow.start_run(run_name=name):
        mlflow.log_metric("f1_macro", f1)
        mlflow.sklearn.log_model(pipe, "model")

    print(f"  {name:<35} F1 macro : {f1:.4f}")

    if f1 > best_run["f1"]:
        best_run = {"name": name, "f1": f1, "pipeline": pipe}

# --- 3. REGISTRATION DU MEILLEUR MODÈLE ---
print(f"\n>>> Meilleur modèle : {best_run['name']} (F1={best_run['f1']:.4f})")
print(">>> Enregistrement dans MLflow Model Registry...")

signature = infer_signature(X_train, best_run["pipeline"].predict(X_train))

with mlflow.start_run(run_name="production_model"):
    mlflow.log_metric("f1_macro", best_run["f1"])
    mlflow.sklearn.log_model(
        best_run["pipeline"],
        name="windscan_production",
        registered_model_name="WindTurbine_MaintenancePredictor",
        signature=signature,
        input_example=X_train.iloc[:3]
    )

client  = MlflowClient(MLFLOW_TRACKING_URI)
versions = client.get_registered_model("WindTurbine_MaintenancePredictor").latest_versions
latest   = versions[-1].version
client.set_registered_model_alias("WindTurbine_MaintenancePredictor", "production", latest)

print(f"Modèle version {latest} promu en 'production'")
