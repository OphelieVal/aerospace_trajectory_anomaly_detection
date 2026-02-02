import pandas as pd
import numpy as np
import json
import os

ESSENTIAL_COLUMNS = [
    "time", "latitude", "longitude", "baro_altitude", 
    "velocity", "heading", "on_ground"
]

OPTIONAL_COLUMNS = ["icao24"]

###############################################################################
# Utilities
###############################################################################

def _safe_list(v):
    """Converts numpy types to python types for JSON serialization."""
    if isinstance(v, (np.ndarray, list)):
        return [ _safe_list(x) for x in v ]
    if isinstance(v, (np.float32, np.float64)):
        return float(v)
    if isinstance(v, (np.int32, np.int64)):
        return int(v)
    return v

###############################################################################
# Schema validation
###############################################################################

def check_schema(df):
    """
    Vérifie que les colonnes essentielles sont présentes.
    Retourne un dict contenant les erreurs potentielles.
    """
    report = {"missing_columns": [], "extra_columns": []}

    missing = [c for c in ESSENTIAL_COLUMNS if c not in df.columns]
    extra   = [c for c in df.columns if c not in ESSENTIAL_COLUMNS + OPTIONAL_COLUMNS]

    report["missing_columns"] = missing
    report["extra_columns"] = extra

    return report


###############################################################################
# Row-level anomaly checks
###############################################################################

def find_inconsistent_rows(df):
    """
    Détecte incohérences dans les données.
    Ne modifie rien : retourne un dataframe 'flags' booléen.
    """

    flags = pd.DataFrame(index=df.index)

    # Coordonnées invalides
    flags["flag_bad_lat"] = ~df["latitude"].between(-90, 90)
    flags["flag_bad_lon"] = ~df["longitude"].between(-180, 180)

    # Altitude incohérente
    flags["flag_altitude_on_ground_mismatch"] = (
        (df["baro_altitude"] > 2) & (df["on_ground"] == True)
    )

    # Altitude nulle mais déclaré en vol
    flags["flag_zero_altitude_in_flight"] = (
        (df["baro_altitude"] == 0) & (df["on_ground"] == False)
    )

    # Vitesses négatives ou absurdes
    flags["flag_bad_velocity"] = (df["velocity"] < 0) | (df["velocity"] > 400)

    # Timestamps non strictement croissants
    if np.issubdtype(df["time"].dtype, np.datetime64):
        time_values = df["time"].astype(np.int64)
    else:
        time_values = df["time"]

    flags["flag_time_not_increasing"] = np.r_[False, np.diff(time_values) <= 0]

    # Données dupliquées exactes
    flags["flag_duplicate_rows"] = df.duplicated()

    return flags


###############################################################################
# Fix simple issues conservatively
###############################################################################

def repair_simple_issues(df, flags):
    """
    Répare uniquement les anomalies sûres :
    - coordonnées invalides → NaN
    - altitude_on_ground incohérente → forcer altitude=0 si l’avion est au sol
    - petites valeurs NA interpolées
    - time monotonicity → tri
    """

    df = df.copy()

    # Invalid lat/lon -> NaN
    df.loc[flags["flag_bad_lat"], "latitude"] = np.nan
    df.loc[flags["flag_bad_lon"], "longitude"] = np.nan

    # Avion au sol mais altitude > 2m : forcer altitude 0
    df.loc[flags["flag_altitude_on_ground_mismatch"], "baro_altitude"] = 0

    # Avion en vol mais altitude = 0 -> laisser NaN (corrigé par interpolation)
    df.loc[flags["flag_zero_altitude_in_flight"], "baro_altitude"] = np.nan

    # Tri par temps
    df = df.sort_values("time")

    # Interpolations (petits trous)
    for col in ["latitude", "longitude", "baro_altitude", "velocity", "heading"]:
        if col in df.columns:
            df[col] = df[col].interpolate(method="linear").fillna(method="bfill").fillna(method="ffill")

    df = df.reset_index(drop=True)

    return df


###############################################################################
# High-level validation pipeline
###############################################################################

def validate_flight(df, flight_id=None):
    """
    Effectue toutes les étapes de validation et de nettoyage.
    Retourne :
    - df_clean : dataframe corrigé
    - report : dictionnaire structuré contenant toutes les anomalies trouvées
    """

    report = {
        "flight_id": flight_id,
        "schema": {},
        "anomalies_summary": {},
        "rows_with_any_anomaly": 0,
        "total_rows": len(df)
    }

    # Convert time to datetime if needed
    if "time" in df.columns and not np.issubdtype(df["time"].dtype, np.datetime64):
        try:
            df["time"] = pd.to_datetime(df["time"])
        except:
            report["schema"]["time_format_error"] = True

    # --- Step 1 : Check schema ---
    schema_report = check_schema(df)
    report["schema"] = schema_report

    if len(schema_report["missing_columns"]) > 0:
        # On renvoie DF vide si schéma invalide
        report["error"] = "missing_columns"
        return None, report

    # --- Step 2 : Analyse anomalies ---
    flags = find_inconsistent_rows(df)

    # Compter anomalies
    anomaly_counts = flags.sum().to_dict()
    anomaly_counts = {k: int(v) for k, v in anomaly_counts.items()}
    report["anomalies_summary"] = anomaly_counts

    # Count rows with ANY anomaly
    report["rows_with_any_anomaly"] = int((flags.any(axis=1)).sum())

    # --- Step 3 : Repair simple problems ---
    df_clean = repair_simple_issues(df, flags)

    return df_clean, report


###############################################################################
# Save report
###############################################################################

def save_report(report, output_path):
    """
    Sauvegarde un rapport de validation en JSON.
    """
    with open(output_path, "w") as f:
        json.dump(_safe_list(report), f, indent=4)


###############################################################################
# Batch mode
###############################################################################

def validate_all(input_folder, output_folder, report_folder):
    """
    Valide tous les fichiers CSV d'un dossier.
    Produit :
    - CSV nettoyés dans output_folder
    - JSON de rapport dans report_folder
    """

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(report_folder, exist_ok=True)

    files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]

    print(f"[INFO] Validating {len(files)} raw trajectory files...")

    for filename in files:
        path = os.path.join(input_folder, filename)
        df = pd.read_csv(path)

        flight_id = filename.replace(".csv", "")

        cleaned, report = validate_flight(df, flight_id=flight_id)

        # Save report
        save_report(report, os.path.join(report_folder, f"{flight_id}.json"))

        if cleaned is None or len(cleaned) == 0:
            print(f"[WARNING] Skipped {filename} due to schema errors or empty result.")
            continue

        cleaned.to_csv(os.path.join(output_folder, filename), index=False)
        print(f"[INFO] Cleaned file saved → {filename}")