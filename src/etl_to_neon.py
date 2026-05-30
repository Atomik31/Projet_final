"""
ETL WindScan - S3 vers Neon DB
Charge les données capteurs depuis S3, valide le schéma, insère dans Neon DB.
Usage : python src/etl_to_neon.py
"""

import os
import io
import sys
import boto3
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parents[1] / "notebook" / ".env")

# ── Config S3 ─────────────────────────────────────────────────────────────────
S3_BUCKET   = "windscan"
S3_REGION   = "eu-north-1"
S3_DATA_KEY = "data/processed/wind_turbine_maintenance_test_data_cleaned.csv"

# ── Config Neon DB ────────────────────────────────────────────────────────────
PG_HOST     = os.getenv("PGHOST")
PG_DB       = os.getenv("PGDATABASE", "neondb")
PG_USER     = os.getenv("PGUSER",     "neondb_owner")
PG_PASSWORD = os.getenv("PGPASSWORD")
PG_SSL      = os.getenv("PGSSLMODE",  "require")

# ── Schéma attendu et règles de validation ────────────────────────────────────
REQUIRED_COLS = [
    "Rotor_Speed_RPM", "Wind_Speed_mps", "Power_Output_kW",
    "Gearbox_Oil_Temp_C", "Generator_Bearing_Temp_C",
    "Vibration_Level_mmps", "Maintenance_Label"
]

VALIDATION_RULES = {
    "Rotor_Speed_RPM":            (0,    25),
    "Wind_Speed_mps":             (0,    30),
    "Power_Output_kW":            (0,  3000),
    "Gearbox_Oil_Temp_C":         (-10,  120),
    "Generator_Bearing_Temp_C":   (-10,  120),
    "Vibration_Level_mmps":       (0,    10),
    "Maintenance_Label":          (0,     2),
}


def get_pg_conn():
    return psycopg2.connect(
        host=PG_HOST, dbname=PG_DB, user=PG_USER,
        password=PG_PASSWORD, sslmode=PG_SSL
    )


def extract_from_s3() -> pd.DataFrame:
    print(">>> [EXTRACT] Lecture depuis S3...")
    s3  = boto3.client("s3", region_name=S3_REGION,
                       aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                       aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"))
    obj = s3.get_object(Bucket=S3_BUCKET, Key=S3_DATA_KEY)
    df  = pd.read_csv(io.BytesIO(obj["Body"].read()))
    print(f"    {len(df)} lignes extraites depuis s3://{S3_BUCKET}/{S3_DATA_KEY}")
    return df


def transform(df: pd.DataFrame) -> pd.DataFrame:
    print(">>> [TRANSFORM] Validation du schéma...")
    initial = len(df)

    # 1. Colonnes obligatoires
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        print(f"    ERREUR — colonnes manquantes : {missing}")
        sys.exit(1)

    # 2. Suppression des lignes avec valeurs nulles sur les capteurs
    df = df.dropna(subset=REQUIRED_COLS)
    print(f"    Lignes après suppression nulls : {len(df)} ({initial - len(df)} supprimées)")

    # 3. Validation des plages de valeurs
    for col, (vmin, vmax) in VALIDATION_RULES.items():
        before = len(df)
        df = df[df[col].between(vmin, vmax)]
        removed = before - len(df)
        if removed > 0:
            print(f"    {col} [{vmin}–{vmax}] : {removed} lignes hors plage supprimées")

    # 4. Filtrer Turbine 1 uniquement
    if "Turbine_ID" in df.columns:
        df = df[df["Turbine_ID"] == 1].reset_index(drop=True)

    # 5. Ajout d'un index temporel simulé si absent
    if "timestamp" not in df.columns:
        df = df.reset_index(drop=True)
        df["row_index"] = df.index

    print(f"    {len(df)} lignes valides après transformation")
    return df


def load_to_neon(df: pd.DataFrame):
    print(">>> [LOAD] Chargement dans Neon DB...")
    conn = get_pg_conn()
    cur  = conn.cursor()

    # Création de la table si absente
    cur.execute("""
        CREATE TABLE IF NOT EXISTS windscan_sensors (
            id                        SERIAL PRIMARY KEY,
            row_index                 INTEGER UNIQUE,
            rotor_speed_rpm           FLOAT,
            wind_speed_mps            FLOAT,
            power_output_kw           FLOAT,
            gearbox_oil_temp_c        FLOAT,
            generator_bearing_temp_c  FLOAT,
            vibration_level_mmps      FLOAT,
            maintenance_label         INTEGER,
            loaded_at                 TIMESTAMP DEFAULT NOW()
        );
    """)
    conn.commit()

    # Insertion avec ON CONFLICT pour éviter les doublons
    rows = [
        (
            int(r.row_index),
            float(r.Rotor_Speed_RPM),
            float(r.Wind_Speed_mps),
            float(r.Power_Output_kW),
            float(r.Gearbox_Oil_Temp_C),
            float(r.Generator_Bearing_Temp_C),
            float(r.Vibration_Level_mmps),
            int(r.Maintenance_Label),
        )
        for r in df.itertuples()
    ]

    execute_values(cur, """
        INSERT INTO windscan_sensors
            (row_index, rotor_speed_rpm, wind_speed_mps, power_output_kw,
             gearbox_oil_temp_c, generator_bearing_temp_c,
             vibration_level_mmps, maintenance_label)
        VALUES %s
        ON CONFLICT (row_index) DO NOTHING;
    """, rows)

    conn.commit()

    cur.execute("SELECT COUNT(*) FROM windscan_sensors;")
    total = cur.fetchone()[0]
    cur.close()
    conn.close()

    print(f"    {len(rows)} lignes insérées | Total en base : {total}")


def run():
    print("=" * 55)
    print("ETL WindScan - S3 vers Neon DB")
    print("=" * 55)
    df = extract_from_s3()
    df = transform(df)
    load_to_neon(df)
    print("=" * 55)
    print("ETL terminé avec succès.")


if __name__ == "__main__":
    run()
