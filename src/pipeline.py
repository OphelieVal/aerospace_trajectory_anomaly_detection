from validator import validate_all
from data_preprocessing import preprocess_all
from merge_processed import merge_processed_files

def run_pipeline():
    print("=== AEROSPACE TRAJECTORY PIPELINE ===")

    print("\n[1/3] VALIDATION STAGE")
    validate_all("data/raw", "data/validated", "data/validated/reports")

    print("\n[2/3] PREPROCESSING STAGE")
    preprocess_all()

    print("\n[3/3] MERGING STAGE")
    merge_processed_files()

    print("\n[DONE] Pipeline finished successfully.")

if __name__ == "__main__":
    run_pipeline()
