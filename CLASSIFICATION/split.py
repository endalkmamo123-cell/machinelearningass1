import pandas as pd
from sklearn.model_selection import train_test_split
import os

# --- Configuration ---
# Source dataset file for classification is stored in the same folder as this script.
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)

dataset_name = "ai_ml_cybersecurity_dataset.csv"
dataset_path = os.path.join(script_dir, dataset_name)

# Output paths relative to this script
train_path = os.path.join(script_dir, "train_dataset.csv")
test_path = os.path.join(script_dir, "test_dataset.csv")
val_path = os.path.join(script_dir, "val_dataset.csv")

def main():
    if not os.path.exists(dataset_path):
        # Try local folder if not in parent (case where user copied it)
        dataset_path_local = os.path.join(script_dir, "train_test_network.csv")
        if os.path.exists(dataset_path_local):
            path_to_use = dataset_path_local
        else:
            print(f"Error: Dataset not found at {dataset_path}")
            return
    else:
        path_to_use = dataset_path

    print(f"Loading dataset from {path_to_use}...")
    df = pd.read_csv(path_to_use)
    print(f"Original dataset shape: {df.shape}")

    # ── 1. Identify Target Column ───────────────────────────────────────────
    # We'll use 'type' as the multi-class target based on dataset inspection.
    target_col = 'type'
    if target_col not in df.columns:
        print(f"Warning: '{target_col}' not found. Columns: {df.columns.tolist()}")
        if 'Attack Type' in df.columns:
            target_col = 'Attack Type'
        elif 'label' in df.columns:
            target_col = 'label'
        else:
            print("Error: No suitable target column found.")
            return

    print(f"Using '{target_col}' as target variable for stratification.")

    # ── 2. Three-way stratified split: 70% train / 10% val / 20% test ────────
    # Step A: hold out 20% as final test set
    trainval_df, test_df = train_test_split(
        df, test_size=0.20, random_state=42, stratify=df[target_col]
    )

    # Step B: from the remaining 80%, hold out 12.5% as validation set
    #         → 80% × 12.5% = 10% of total
    train_df, val_df = train_test_split(
        trainval_df, test_size=0.125, random_state=42, stratify=trainval_df[target_col]
    )

    # ── 3. Save splits ────────────────────────────────────────────────────────
    print(f"Saving splits to {train_path}, {val_path}, {test_path}...")
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    total = len(df)
    print(f"\nSuccess! Stratified 3-way splits saved.")
    print(f"  Train : {train_path} — {len(train_df):,} rows ({len(train_df)/total:.0%})")
    print(f"  Val   : {val_path}   — {len(val_df):,} rows ({len(val_df)/total:.0%})")
    print(f"  Test  : {test_path}  — {len(test_df):,} rows ({len(test_df)/total:.0%})")

    print(f"\nFinal distribution on {target_col}:")
    print(train_df[target_col].value_counts())

if __name__ == "__main__":
    main()
