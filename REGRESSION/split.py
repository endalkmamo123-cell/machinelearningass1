import pandas as pd
from sklearn.model_selection import train_test_split
import os

# --- Configuration ---
# Source dataset is in the parent directory of this script's folder
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(os.path.dirname(script_dir), "train_test_network.csv")

# Output paths relative to this script
train_path = os.path.join(script_dir, "train.csv")
test_path  = os.path.join(script_dir, "test.csv")
valid_path = os.path.join(script_dir, "valid.csv")

def main():
    if not os.path.exists(dataset_path):
        # Try local folder if not in parent (case where user copied it)
        dataset_path_local = os.path.join(script_dir, "train_test_network.csv")
        if os.path.exists(dataset_path_local):
            path_to_use = dataset_path_local
        else:
            print(f"Error: Source dataset not found at {dataset_path}")
            return
    else:
        path_to_use = dataset_path

    print(f"Loading dataset from {path_to_use}...")
    df = pd.read_csv(path_to_use)
    print(f"Original dataset shape: {df.shape}")

    # --- Regression Split (Random) ---
    # We use a random split (no stratification) as this is for a general regression task
    # and the specific target has not been finalized.
    
    print("\nPerforming 70/20/10 random split...")
    
    # Step 1: Hold out 20% for testing
    trainval_df, test_df = train_test_split(
        df, test_size=0.20, random_state=42, shuffle=True
    )
    
    # Step 2: From the remaining 80%, hold out 12.5% for validation
    # (80% * 0.125 = 10% of total)
    train_df, val_df = train_test_split(
        trainval_df, test_size=0.125, random_state=42, shuffle=True
    )

    # --- Save Splits ---
    print(f"Saving splits to {train_path}, {test_path}, {valid_path}...")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    val_df.to_csv(valid_path, index=False)

    total_rows = len(df)
    print(f"\nSuccess! Regression splits saved.")
    print(f"  Train : {train_path}  — {len(train_df):,} rows ({len(train_df)/total_rows:.0%})")
    print(f"  Test  : {test_path}   — {len(test_df):,} rows ({len(test_df)/total_rows:.0%})")
    print(f"  Valid : {valid_path}  — {len(val_df):,} rows ({len(val_df)/total_rows:.0%})")

if __name__ == "__main__":
    main()
