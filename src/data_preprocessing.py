import pandas as pd
import numpy as np
import os
import glob

RAW_DIR = os.path.join("..", "data", "raw")
PROCESSED_DIR = os.path.join("..", "data", "processed")
OUTPUT_DIR = OUTPUT_CSV = os.path.join(PROCESSED_DIR, "processed_file.csv")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# -------------------------------
# Helper : safe diff / safe division
# -------------------------------
def safe_diff(series):
    """
    Compute difference of a pandas Series safely.
    """
    if series is None:
        return 0
    return series.diff().fillna(0)

def safe_div(numerator, denominator):
    """
    Safely divide two pandas Series, handling division by zero.
    """
    denominator = denominator.replace(0, np.nan)
    return (numerator / denominator).replace([np.inf, -np.inf], 0).fillna(0)

def preprocess_file(filepath: str) -> pd.DataFrame:
    """
    Preprocess a single raw trajectory CSV.
    """
    print(f"[INFO] Preprocessing {filepath} ...")

    # Skip empty files
    if os.path.getsize(filepath) < 10:
        print(f"[WARNING] File is empty → skipping")
        return None

    df = pd.read_csv(filepath)

    # -------------------------------
    # STEP 1 – Required columns
    # -------------------------------
    essential_cols = ["time", "latitude", "longitude", "baro_altitude"]

    missing = [c for c in essential_cols if c not in df.columns]
    if missing:
        print(f"[WARNING] Missing essential columns {missing} → skipping file.")
        return None

    # If velocity missing, create placeholder (so pipeline continues)
    if "velocity" not in df.columns:
        print("[WARNING] velocity missing → filling with zeros")
        df["velocity"] = 0.0

    # -------------------------------
    # STEP 2 – Basic cleaning
    # -------------------------------
    df = df.dropna(subset=essential_cols).reset_index(drop=True)
    if df.empty:
        print("[WARNING] No usable rows after filtering → skipping")
        return None

    # Convert timestamp → datetime
    df["time"] = pd.to_datetime(df["time"], unit="s", errors="coerce")
    df = df.dropna(subset=["time"])  # remove bad timestamps

    df = df.sort_values("time").drop_duplicates().reset_index(drop=True)

    # -------------------------------
    # STEP 3 – Interpolation of numeric fields
    # -------------------------------
    numeric_cols = ["latitude", "longitude", "baro_altitude", "velocity"]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .interpolate(method="linear")
                .fillna(method="bfill")
                .fillna(method="ffill")
            )
        else:
            df[col] = 0
            print(f"[WARNING] Added missing numeric column: {col}")

    # -------------------------------
    # STEP 4 – Feature engineering
    # -------------------------------

    # Time delta (in seconds)
    df["delta_time"] = df["time"].diff().dt.total_seconds().fillna(1)
    df["delta_time"] = df["delta_time"].replace(0, 1)  # Avoid division by zero

    # Spatial deltas
    df["delta_lat"] = safe_diff(df["latitude"])
    df["delta_lon"] = safe_diff(df["longitude"])
    df["delta_altitude"] = safe_diff(df["baro_altitude"])
    df["delta_velocity"] = safe_diff(df["velocity"])

    # Approximate conversion degrees → meters
    DEG_TO_M = 111_000

    df["vx"] = safe_div(df["delta_lon"] * DEG_TO_M, df["delta_time"])
    df["vy"] = safe_div(df["delta_lat"] * DEG_TO_M, df["delta_time"])
    df["v_vertical"] = safe_div(df["delta_altitude"], df["delta_time"])

    # Accelerations
    df["ax"] = safe_div(safe_diff(df["vx"]), df["delta_time"])
    df["ay"] = safe_div(safe_diff(df["vy"]), df["delta_time"])
    df["a_vertical"] = safe_div(safe_diff(df["v_vertical"]), df["delta_time"])

    # Heading change (if available)
    if "heading" in df.columns:
        df["delta_heading"] = safe_diff(df["heading"])
    else:
        df["delta_heading"] = 0

    # ICAO24 handling
    df["icao24"] = df["icao24"].iloc[0] if "icao24" in df.columns else "unknown"

    return df


def preprocess_all():
    """
    Preprocess all raw CSV files and save processed versions.
    """
    files = glob.glob(os.path.join(RAW_DIR, "*.csv"))
    print(f"[INFO] Found {len(files)} files in raw data folder.")

    for filepath in files:
        df = preprocess_file(filepath)
        filename = os.path.basename(filepath)

        if df is None:
            print(f"[INFO] Skipped {filename}")
            continue

        df.to_csv(OUTPUT_CSV, index=False)
        print(f"[INFO] Processed and saved → {OUTPUT_CSV}")

if __name__ == "__main__":
    preprocess_all()
