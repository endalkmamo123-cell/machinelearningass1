# %% [markdown]
# # Cybersecurity Intrusion Detection Pipeline (Updated for Network Flows)
# Refactored for the `train_test_network.csv` schema.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, cross_val_score
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, roc_auc_score
)
import joblib
import warnings
import os
from io import StringIO

warnings.filterwarnings('ignore')

# %% [markdown]
# ## 1. Data Loading

# %%
script_dir = os.path.dirname(os.path.abspath(__file__))

EXPECTED_COLUMNS = [
    'Event ID', 'Timestamp', 'Source IP', 'Destination IP',
    'User Agent', 'Attack Type', 'Attack Severity',
    'Data Exfiltrated', 'Threat Intelligence', 'Response Action'
]

def _load_csv_with_header_repair(path, expected_columns=None):
    expected_columns = expected_columns or EXPECTED_COLUMNS
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        raw = f.read()

    # Repair cases where the header row is prefixed by stray characters
    header_start = raw.find(expected_columns[0])
    if header_start > 0:
        raw = raw[header_start:]

    # Repair cases where the header row is concatenated with the first data row
    first_newline = raw.find('\n')
    if first_newline != -1:
        first_line = raw[:first_newline]
        last_header = expected_columns[-1]
        last_index = first_line.find(last_header)
        if last_index != -1:
            after_header = first_line[last_index + len(last_header):]
            if after_header and not after_header.startswith(('\r', '\n')):
                raw = (
                    first_line[: last_index + len(last_header)]
                    + '\n'
                    + after_header
                    + raw[first_newline:]
                )

    return pd.read_csv(StringIO(raw))

train_df = _load_csv_with_header_repair(os.path.join(script_dir, 'train_dataset.csv'))
val_df = _load_csv_with_header_repair(os.path.join(script_dir, 'val_dataset.csv'))
test_df = _load_csv_with_header_repair(os.path.join(script_dir, 'test_dataset.csv'))

print(f"Train : {train_df.shape}")
print(f"Val   : {val_df.shape}")
print(f"Test  : {test_df.shape}")

# %%
def _ip_first_octet(ip_str):
    try:
        return int(str(ip_str).split('.')[0])
    except Exception:
        return 0

def preprocess_data(df, is_train=True, encoders=None, scaler=None):
    df = df.copy()

    # -- Identify Target -----------------------------------------------------
    target_col = 'type'
    if target_col not in df.columns:
        target_col = 'Attack Type' if 'Attack Type' in df.columns else 'label'

    y = df[target_col]
    X = df.drop(columns=[target_col, 'label'], errors='ignore')

    # -- Handle '-' and missing values ---------------------------------------
    X = X.replace('-', np.nan)
    
    # -- IP features ---------------------------------------------------------
    for col, prefix in [('src_ip', 'src'), ('dst_ip', 'dst')]:
        if col in X.columns:
            X[f'{prefix}_octet1'] = X[col].apply(_ip_first_octet)
            X = X.drop(columns=[col])

    # -- Categorical encoding -------------------------------------------------
    # Based on the new schema: proto, service, conn_state, and potentially others
    # We will encode all non-numeric columns
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    if is_train:
        encoders = {}
        # Target Encoding
        le_y = LabelEncoder()
        y = le_y.fit_transform(y.astype(str))
        encoders['target'] = le_y
        
        # Feature Encoding
        for col in cat_cols:
            le = LabelEncoder()
            # Fillna with 'unknown' before fit
            X[col] = le.fit_transform(X[col].astype(str).fillna('unknown'))
            encoders[col] = le
            
    else:
        # Target Encoding
        y = encoders['target'].transform(y.astype(str))
        
        # Feature Encoding
        for col in cat_cols:
            if col in encoders:
                X[col] = X[col].astype(str).fillna('unknown').apply(
                    lambda x: encoders[col].transform([x])[0]
                    if x in encoders[col].classes_ else -1
                )
            else:
                # If a new categorical column appeared in val/test that wasn't in train
                X[col] = 0 

    # -- Final Check for Numerics --------------------------------------------
    # Convert anything left to numeric, fill NaNs
    for col in X.columns:
        if X[col].dtype == 'object':
            # Attempt coercion
            X[col] = pd.to_numeric(X[col], errors='coerce')
    
    X = X.fillna(0) # Fill remaining NaNs with 0

    # -- Feature Correlation (train only) -----------------------------------
    if is_train:
        plt.figure(figsize=(14, 12))
        sns.heatmap(X.corr(), annot=False, cmap='coolwarm')
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(script_dir, 'correlation_heatmap.png'))
        plt.close()

    # -- Scaling --------------------------------------------------------------
    if is_train:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    return X_scaled, y, encoders, scaler

# %%
print("\nPreprocessing data...")
X_train, y_train, encoders, scaler = preprocess_data(train_df, is_train=True)
X_val,   y_val,   _,        _      = preprocess_data(val_df,   is_train=False, encoders=encoders, scaler=scaler)
X_test,  y_test,  _,        _      = preprocess_data(test_df,  is_train=False, encoders=encoders, scaler=scaler)

print(f"Features -> Train:{X_train.shape}  Val:{X_val.shape}  Test:{X_test.shape}")

# %% [markdown]
# ## 2. Model Training

# %%
print("\nTraining XGBoost...")
xgb_clf = XGBClassifier(
    n_estimators=100, 
    learning_rate=0.1, 
    max_depth=5, 
    random_state=42, 
    n_jobs=-1,
    eval_metric='mlogloss'
)
xgb_clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# %%
print("\nTraining Random Forest...")
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_clf.fit(X_train, y_train)

# %% [markdown]
# ## 3. Evaluation

# %%
target_names = list(encoders['target'].classes_)

def evaluate_model(model, X, y, model_name, split_name='Test'):
    print(f"\n--- {model_name} | {split_name} ---")
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y, y_pred, target_names=target_names))

    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f'CM: {model_name} ({split_name})')
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, f'{model_name}_{split_name}_confusion_matrix.png'))
    plt.close()

evaluate_model(xgb_clf, X_test, y_test, 'XGBoost')
evaluate_model(rf_clf,  X_test, y_test, 'Random_Forest')

# %% [markdown]
# ## 4. Saving Artifacts

# %%
print("\nSaving artifacts...")
joblib.dump(xgb_clf,  os.path.join(script_dir, 'best_xgb_model.pkl'))
joblib.dump(rf_clf,   os.path.join(script_dir, 'best_rf_model.pkl'))
joblib.dump(encoders, os.path.join(script_dir, 'encoders.pkl'))
joblib.dump(scaler,   os.path.join(script_dir, 'scaler.pkl'))
print("Complete.")
