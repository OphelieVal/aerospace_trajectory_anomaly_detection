import pandas as pd
import os
import glob

PROCESSED_DIR = os.path.join("..", "data", "processed")
OUTPUT_CSV = os.path.join(PROCESSED_DIR, "merged_dataset.csv")
OUTPUT_PARQUET = os.path.join(PROCESSED_DIR, "merged_dataset.parquet")

def merge_processed_files():
    """
    Merge all processed CSV files into a single dataset with a flight_id.
    """
    files = glob.glob(os.path.join(PROCESSED_DIR, "*.csv"))
    
    # Remove merged files from previous runs if they exist
    files = [f for f in files if "merged_dataset" not in f]

    if len(files) == 0:
        print("[ERROR] No processed CSV files found.")
        return None

    print(f"[INFO] Found {len(files)} processed CSV files.")
    
    all_dfs = []
    
    for idx, filepath in enumerate(files):
        df = pd.read_csv(filepath)
        
        # Generate flight ID (ex: flight_001, flight_002, ...)
        flight_id = f"flight_{idx+1:03d}"
        df["flight_id"] = flight_id

        all_dfs.append(df)
        print(f"[INFO] Loaded {filepath} as {flight_id}")

    # Concatenate all flights
    merged_df = pd.concat(all_dfs, ignore_index=True)
    
    # Optional: sort by flight and timestamp
    merged_df = merged_df.sort_values(by=["flight_id", "time"]).reset_index(drop=True)

    # Save CSV
    merged_df.to_csv(OUTPUT_CSV, index=False)
    print(f"[INFO] Merged dataset saved to: {OUTPUT_CSV}")

    try:
        merged_df.to_parquet(OUTPUT_PARQUET)
        print(f"[INFO] Also saved as Parquet: {OUTPUT_PARQUET}")
    except Exception as e:
        print("[WARNING] Could not save as Parquet (optional). Error:", e)

    return merged_df


if __name__ == "__main__":
    df = merge_processed_files()
    if df is not None:
        print("[INFO] Merge completed successfully.")
        print(df.head())
