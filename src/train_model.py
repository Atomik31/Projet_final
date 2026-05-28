import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
import joblib

# --- 1. CHARGEMENT ET PRÉPARATION DES DONNÉES ---
print(">>> Chargement des données...")
df = pd.read_csv('../data/raw/wind_turbine_maintenance_data.csv')

# Création d'un index temporel simulé pour l'exercice
dates = pd.date_range(start='2022-01-01', periods=len(df)//2, freq='H')
df.loc[df['Turbine_ID'] == 1, 'Timestamp'] = dates
df.loc[df['Turbine_ID'] == 2, 'Timestamp'] = dates
df = df.sort_values(by=['Turbine_ID', 'Timestamp'])

# --- 2. FEATURE ENGINEERING (Pour les modèles avancés) ---
print(">>> Génération des Features Temporelles...")
df_eng = df.copy()
cols_cibles = ['Vibration_Level_mmps', 'Gearbox_Oil_Temp_C', 'Generator_Bearing_Temp_C']

for col in cols_cibles:
    # Lags (Passé immédiat)
    df_eng[f'{col}_Lag1'] = df_eng.groupby('Turbine_ID')[col].shift(1)
    df_eng[f'{col}_Lag2'] = df_eng.groupby('Turbine_ID')[col].shift(2)
    # Rolling Stats (Tendances et Volatilité)
    df_eng[f'{col}_Mean_6h'] = df_eng.groupby('Turbine_ID')[col].transform(lambda x: x.rolling(6).mean())
    df_eng[f'{col}_Std_6h'] = df_eng.groupby('Turbine_ID')[col].transform(lambda x: x.rolling(6).std())
    df_eng[f'{col}_Mean_24h'] = df_eng.groupby('Turbine_ID')[col].transform(lambda x: x.rolling(24).mean())

# Nettoyage des NaN
df_eng = df_eng.dropna().reset_index(drop=True)

# Pour la baseline, on prend les mêmes lignes (alignement)
df_raw = df.loc[df_eng.index].reset_index(drop=True)

# --- 3. DÉFINITION DU SPLIT CHRONOLOGIQUE ---
split_idx = int(len(df_eng) * 0.80)

# Jeux de données "pour rendre plus Intelligents" (Avec features)
X_train_eng = df_eng.iloc[:split_idx].drop(['Turbine_ID', 'Timestamp', 'Maintenance_Label'], axis=1)
y_train = df_eng.iloc[:split_idx]['Maintenance_Label']
X_test_eng = df_eng.iloc[split_idx:].drop(['Turbine_ID', 'Timestamp', 'Maintenance_Label'], axis=1)
y_test = df_eng.iloc[split_idx:]['Maintenance_Label']

# Jeux de données "Bruts" (Pour la baseline)
raw_features = ['Rotor_Speed_RPM', 'Wind_Speed_mps', 'Power_Output_kW', 'Gearbox_Oil_Temp_C', 
                'Generator_Bearing_Temp_C', 'Vibration_Level_mmps', 'Ambient_Temp_C', 'Humidity_pct']
X_train_raw = df_raw.iloc[:split_idx][raw_features]
X_test_raw = df_raw.iloc[split_idx:][raw_features]

# --- 4. LE TOURNOI DES MODÈLES ---

# A) BASELINE (Reg Log sur données brutes)
print("\n>>> 1. Entraînement Baseline (LogReg Simple)...")
model_baseline = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(class_weight='balanced', random_state=42))
])
model_baseline.fit(X_train_raw, y_train)
f1_baseline = f1_score(y_test, model_baseline.predict(X_test_raw), average='weighted')

# B) CHALLENGER (Lasso sur Features Temporelles)
print(">>> 2. Entraînement Challenger (Lasso + Features Temporelles)...")
model_lasso = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(penalty='l1', solver='liblinear', C=0.5, class_weight='balanced', random_state=42))
])
model_lasso.fit(X_train_eng, y_train)
f1_lasso = f1_score(y_test, model_lasso.predict(X_test_eng), average='weighted')

# C) CHAMPION (Random Forest sur Features Temporelles)
print(">>> 3. Entraînement Champion (Random Forest + Features Temporelles)...")
model_champion = RandomForestClassifier(n_estimators=100, max_depth=15, class_weight='balanced', random_state=42, n_jobs=-1)
model_champion.fit(X_train_eng, y_train)
f1_champion = f1_score(y_test, model_champion.predict(X_test_eng), average='weighted')

# --- 5. RÉSULTATS COMPARATIFS ---
print("\n" + "="*45)
print("   🏆 ÉVOLUTION DU PROJET (F1-SCORE WEIGHTED)")
print("="*45)
print(f"1. BASELINE (Naïve)       : {f1_baseline:.2%}")
print(f"   -> Approche : Données brutes, Modèle simple")
print("-" * 45)
print(f"2. CHALLENGER (Ingénierie): {f1_lasso:.2%}")
print(f"   -> Approche : Features Temporelles, Lasso")
print(f"   -> Gain vs Baseline : {f1_lasso - f1_baseline:+.2%}")
print("-" * 45)
print(f"3. CHAMPION (Non-Linéaire): {f1_champion:.2%}")
print(f"   -> Approche : Random Forest, Interactions")
print(f"   -> Gain vs Challenger : {f1_champion - f1_lasso:+.2%}")
print("="*45)

# --- 6. SAUVEGARDE DU VAINQUEUR ---
# Car le Random Forest gérera mieux les cas bizarres (non-linéaires) non vus dans le test.
print(f"\n>>> Sauvegarde du modèle CHAMPION (Random Forest)...")
joblib.dump(model_champion, '../models/gtc_model_advanced.pkl')
print("Modèle sauvegardé dans models/gtc_model_advanced.pkl")