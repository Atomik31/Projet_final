"""
run.py — Pipeline complet Wind Turbine Maintenance
Combine Data-cleaning + Training + Upload S3

Usage (depuis la racine du projet) :
    python src/run.py
"""

import os
import warnings
import io
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# ENV & WARNINGS
# ---------------------------------------------------------------------------

# Cherche .env dans notebook/ (où il est stocké)
ENV_FILE = Path(__file__).resolve().parent.parent / "notebook" / ".env"
load_dotenv(ENV_FILE)
os.environ["PYTHONWARNINGS"] = "ignore"

from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", message=".*disp.*",   category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*iprint.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*delayed.*", category=UserWarning)

import pandas as pd
import numpy as np
import joblib
import boto3
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, make_scorer

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

ROOT        = Path(__file__).resolve().parent.parent
DATA_RAW    = ROOT / "data" / "raw"
DATA_PROC   = ROOT / "data" / "processed"
MODELS_DIR  = ROOT / "models"
DATA_PROC.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# S3 CONFIG
# ---------------------------------------------------------------------------

S3_BUCKET   = "windscan"
S3_REGION   = "eu-north-1"
S3_DATA_KEY = "data/processed/"
S3_MODEL_KEY = "models/"

def get_s3_client():
    return boto3.client(
        "s3",
        region_name=S3_REGION,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

def upload_to_s3(local_path: Path, s3_key: str):
    """Upload a local file to S3 and print confirmation."""
    s3 = get_s3_client()
    s3.upload_file(str(local_path), S3_BUCKET, s3_key)
    print(f"  S3 upload OK : s3://{S3_BUCKET}/{s3_key}")

# ---------------------------------------------------------------------------
# STEP 1 — DATA CLEANING (test data)
# ---------------------------------------------------------------------------

def run_data_cleaning():
    print("\n" + "=" * 60)
    print("STEP 1 — DATA CLEANING (test dataset)")
    print("=" * 60)

    raw_path = DATA_RAW / "wind_turbine_maintenance_test_data.csv"
    dataset  = pd.read_csv(raw_path)
    print(f"Raw test data loaded : {dataset.shape}")

    # Filter Turbine 1
    turbine1_test = dataset[dataset["Turbine_ID"] == 1].copy()
    print(f"Turbine 1 only       : {turbine1_test.shape}")

    # Drop useless columns
    useless_cols  = ["Turbine_ID", "Ambient_Temp_C", "Humidity_pct"]
    turbine1_test = turbine1_test.drop(useless_cols, axis=1)

    # Verify columns
    expected_cols = [
        "Rotor_Speed_RPM", "Wind_Speed_mps", "Power_Output_kW",
        "Gearbox_Oil_Temp_C", "Generator_Bearing_Temp_C",
        "Vibration_Level_mmps", "Maintenance_Label",
    ]
    missing = [c for c in expected_cols if c not in turbine1_test.columns]
    extra   = [c for c in turbine1_test.columns if c not in expected_cols]
    if missing or extra:
        raise ValueError(f"Column mismatch — missing: {missing}, extra: {extra}")
    print("Column conformity     : OK")

    # Save locally
    out_path = DATA_PROC / "wind_turbine_maintenance_test_data_cleaned.csv"
    turbine1_test.to_csv(out_path, index=False)
    print(f"Saved locally         : {out_path.relative_to(ROOT)}")

    # Upload to S3
    s3_key = S3_DATA_KEY + out_path.name
    upload_to_s3(out_path, s3_key)

    return turbine1_test


# ---------------------------------------------------------------------------
# STEP 2 — TRAINING PIPELINE
# ---------------------------------------------------------------------------

def run_training():
    print("\n" + "=" * 60)
    print("STEP 2 — TRAINING PIPELINE")
    print("=" * 60)

    # -- 2.1  Load & prepare train data --
    dataset = pd.read_csv(DATA_RAW / "wind_turbine_maintenance_data.csv")
    turbine1 = dataset[dataset["Turbine_ID"] == 1].copy()
    turbine1 = turbine1.drop(["Turbine_ID", "Ambient_Temp_C", "Humidity_pct"], axis=1)

    # Save processed train data locally + S3
    proc_train_path = DATA_PROC / "dataset.csv"
    turbine1.to_csv(proc_train_path, index=False)
    print(f"Processed train saved : {proc_train_path.relative_to(ROOT)}")
    upload_to_s3(proc_train_path, S3_DATA_KEY + proc_train_path.name)

    # -- 2.2  Split X / Y --
    Y = turbine1["Maintenance_Label"]
    X = turbine1.drop("Maintenance_Label", axis=1)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, shuffle=False
    )
    print(f"Train : {X_train.shape} | Test : {X_test.shape}")

    # -- 2.3  Preprocessor --
    numeric_features      = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
    categorical_features  = [c for c in X_train.columns if not pd.api.types.is_numeric_dtype(X_train[c])]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler",  StandardScaler()),
    ])
    transformers = [("num", numeric_transformer, numeric_features)]
    if categorical_features:
        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
        ])
        transformers.append(("cat", categorical_transformer, categorical_features))

    preprocessor = ColumnTransformer(transformers)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc  = preprocessor.transform(X_test)
    print("Preprocessor fitted.")

    # -- 2.4  MLflow setup --
    mlflow.set_tracking_uri("https://atomik31-mlflow.hf.space")
    mlflow.set_experiment("Wind_Turbine_Maintenance")
    f1_scorer = make_scorer(f1_score, average="macro", zero_division=0)
    best_run  = {"name": None, "f1_test": -1, "model": None, "params": {}}

    def _log(run_name, model, params, X_tr, X_te):
        """Train, log to MLflow, update best_run."""
        model.fit(X_tr, Y_train)
        f1_tr = f1_score(Y_train, model.predict(X_tr), average="macro", zero_division=0)
        f1_te = f1_score(Y_test,  model.predict(X_te), average="macro", zero_division=0)
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(params)
            mlflow.log_metrics({"f1_train": f1_tr, "f1_test": f1_te})
            mlflow.sklearn.log_model(model, "model")
        print(f"  [{run_name}] F1 train={f1_tr:.4f}  test={f1_te:.4f}")
        if f1_te > best_run["f1_test"]:
            best_run.update({"name": run_name, "f1_test": f1_te,
                             "model": model, "params": params})
        return f1_te

    # -- 2.5  Baseline --
    print("\n--- Models ---")
    _log("Baseline_DummyClassifier",
         DummyClassifier(strategy="most_frequent", random_state=42),
         {"strategy": "most_frequent"},
         X_train_proc, X_test_proc)

    # -- 2.6  Logistic Regression --
    params_lr = {"solver": "lbfgs", "class_weight": "balanced", "max_iter": 1000}
    _log("LogisticRegression",
         LogisticRegression(**params_lr, random_state=42),
         params_lr, X_train_proc, X_test_proc)

    # -- 2.7  Lasso L1 --
    params_lasso = {"penalty": "l1", "solver": "saga", "C": 0.01,
                    "class_weight": "balanced", "max_iter": 2000}
    _log("Lasso_L1",
         LogisticRegression(**params_lasso, random_state=42),
         params_lasso, X_train_proc, X_test_proc)

    # -- 2.8  Random Forest --
    params_rf = {"n_estimators": 100, "class_weight": "balanced",
                 "max_depth": 10, "min_samples_split": 5,
                 "min_samples_leaf": 4, "max_features": "log2"}
    _log("RandomForest",
         RandomForestClassifier(**params_rf, random_state=42),
         params_rf, X_train_proc, X_test_proc)

    # -- 2.9  GridSearch LR --
    param_grid_lr = {"C": [0.005, 0.1, 1, 10, 100], "penalty": ["l1"],
                     "solver": ["saga"], "class_weight": ["balanced"]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_lr = GridSearchCV(
        LogisticRegression(random_state=42, max_iter=2000, solver="saga"),
        param_grid_lr, scoring=f1_scorer, cv=cv, verbose=0, n_jobs=1,
    )
    grid_lr.fit(X_train_proc, Y_train)
    best_lr = grid_lr.best_estimator_
    f1_te_gs_lr = f1_score(Y_test, best_lr.predict(X_test_proc), average="macro", zero_division=0)
    f1_tr_gs_lr = f1_score(Y_train, best_lr.predict(X_train_proc), average="macro", zero_division=0)
    with mlflow.start_run(run_name="GridSearch_LogisticRegression"):
        mlflow.log_params(grid_lr.best_params_)
        mlflow.log_metrics({"f1_train": f1_tr_gs_lr, "f1_test": f1_te_gs_lr,
                            "best_cv_score": grid_lr.best_score_})
        mlflow.sklearn.log_model(best_lr, "model")
    print(f"  [GridSearch_LR] F1 train={f1_tr_gs_lr:.4f}  test={f1_te_gs_lr:.4f}")
    if f1_te_gs_lr > best_run["f1_test"]:
        best_run.update({"name": "GridSearch_LR", "f1_test": f1_te_gs_lr,
                         "model": best_lr, "params": grid_lr.best_params_})

    # -- 2.10  GridSearch RF --
    param_grid_rf = {
        "n_estimators": [30, 50], "max_depth": [5, 10],
        "min_samples_split": [2, 3], "min_samples_leaf": [2, 4],
        "class_weight": ["balanced", "balanced_subsample"],
    }
    grid_rf = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid_rf, scoring=f1_scorer, cv=5, verbose=0, n_jobs=1,
    )
    grid_rf.fit(X_train_proc, Y_train)
    best_rf_gs = grid_rf.best_estimator_
    f1_te_gs_rf = f1_score(Y_test, best_rf_gs.predict(X_test_proc), average="macro", zero_division=0)
    f1_tr_gs_rf = f1_score(Y_train, best_rf_gs.predict(X_train_proc), average="macro", zero_division=0)
    with mlflow.start_run(run_name="GridSearch_RandomForest"):
        mlflow.log_params(grid_rf.best_params_)
        mlflow.log_metrics({"f1_train": f1_tr_gs_rf, "f1_test": f1_te_gs_rf,
                            "best_cv_score": grid_rf.best_score_})
        mlflow.sklearn.log_model(best_rf_gs, "model")
    print(f"  [GridSearch_RF] F1 train={f1_tr_gs_rf:.4f}  test={f1_te_gs_rf:.4f}")
    if f1_te_gs_rf > best_run["f1_test"]:
        best_run.update({"name": "GridSearch_RF", "f1_test": f1_te_gs_rf,
                         "model": best_rf_gs, "params": grid_rf.best_params_})

    # -- 2.11  Manual RF configs --
    configs = [
        {"n_estimators": 200, "max_depth": 15, "min_samples_split": 8,
         "min_samples_leaf": 2,  "note": "Overfitting_fort"},
        {"n_estimators": 100, "max_depth": 10, "min_samples_split": 5,
         "min_samples_leaf": 4,  "note": "Params_retenus"},
        {"n_estimators": 200, "max_depth": 5,  "min_samples_split": 5,
         "min_samples_leaf": 4,  "note": "Sous_apprentissage"},
    ]
    for cfg in configs:
        params = {k: v for k, v in cfg.items() if k != "note"}
        rf_tmp = RandomForestClassifier(**params, class_weight="balanced", random_state=42)
        _log(f"RF_manual_{cfg['note']}", rf_tmp,
             {**params, "class_weight": "balanced"},
             X_train_proc, X_test_proc)

    # -- 2.12  Save best model --
    print("\n" + "=" * 60)
    print(f"Best model : {best_run['name']}  (F1 test={best_run['f1_test']:.4f})")
    print("=" * 60)

    model_path = MODELS_DIR / "best_model.pkl"
    prep_path  = MODELS_DIR / "preprocessor.pkl"
    joblib.dump(best_run["model"], model_path)
    joblib.dump(preprocessor, prep_path)
    print(f"Saved locally : {model_path.relative_to(ROOT)}")
    print(f"Saved locally : {prep_path.relative_to(ROOT)}")

    # Upload to S3
    upload_to_s3(model_path, S3_MODEL_KEY + model_path.name)
    upload_to_s3(prep_path,  S3_MODEL_KEY + prep_path.name)

    # -- 2.13  Final report --
    Y_pred = best_run["model"].predict(X_test_proc)
    print("\nClassification report (test set) :")
    print(classification_report(
        Y_test, Y_pred,
        target_names=["Normal", "Maint. mineure", "Maint. majeure"],
        zero_division=0,
    ))

    return best_run["model"], preprocessor


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Pipeline Wind Turbine Maintenance")
    print(f"AWS key loaded : {bool(os.getenv('AWS_ACCESS_KEY_ID'))}")

    run_data_cleaning()
    run_training()

    print("\nPipeline terminé.")
