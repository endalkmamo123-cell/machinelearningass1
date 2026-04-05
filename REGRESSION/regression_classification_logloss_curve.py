import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import log_loss
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

# %% [markdown]
# ## 1. Data Loading

folder = os.path.dirname(__file__)
train_df = pd.read_csv(os.path.join(folder, 'train.csv'))
val_df   = pd.read_csv(os.path.join(folder, 'valid.csv'))
test_df  = pd.read_csv(os.path.join(folder, 'test.csv'))

# %% [markdown]
# ## 2. Data Preparation

def _ip_first_octet(ip_str):
    try: return int(str(ip_str).split('.')[0])
    except: return 0

def prepare_classification_data(df, is_train=True, encoders=None, scaler=None):
    df = df.copy()
    
    # Target: type (multiclass)
    target_col = 'type'
    y = df[target_col].astype(str)
    
    X = df.drop(columns=[target_col, 'label'], errors='ignore')
    X = X.replace('-', np.nan)

    # IP features
    for col, prefix in [('src_ip', 'src'), ('dst_ip', 'dst')]:
        if col in X.columns:
            X[f'{prefix}_octet1'] = X[col].apply(_ip_first_octet)
            X = X.drop(columns=[col])

    # Categorical Encoding
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    if is_train:
        encoders = {}
        # Target Encoding
        le_y = LabelEncoder()
        y_encoded = le_y.fit_transform(y)
        encoders['target'] = le_y
        
        # Feature Encoding
        for col in cat_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str).fillna('unknown'))
            encoders[col] = le
    else:
        # Target Encoding
        y_encoded = encoders['target'].transform(y)
        
        # Feature Encoding
        for col in cat_cols:
            if col in encoders:
                X[col] = X[col].astype(str).fillna('unknown').apply(
                    lambda x: encoders[col].transform([x])[0] if x in encoders[col].classes_ else -1
                )
            else: X[col] = 0

    X = X.fillna(0)
    for col in X.columns:
        if X[col].dtype == 'object': X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.fillna(0)

    if is_train:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    
    return X_scaled, y_encoded, encoders, scaler

print("Preparing data for log loss analysis...")
X_train, y_train, encoders, scaler = prepare_classification_data(train_df, is_train=True)
X_val,   y_val,   _,        _      = prepare_classification_data(val_df,   is_train=False, encoders=encoders, scaler=scaler)
X_test,  y_test,  _,        _      = prepare_classification_data(test_df,  is_train=False, encoders=encoders, scaler=scaler)

# %% [markdown]
# ## 3. Model Training with Eval Metrics

print("Training XGBoost Classifier and tracking Log Loss curve...")
xgb_clf = XGBClassifier(
    n_estimators=100, 
    learning_rate=0.1, 
    max_depth=5, 
    random_state=42, 
    n_jobs=-1,
    eval_metric='mlogloss'
)

# Crucial: Eval on ALL three sets to get the curves
xgb_clf.fit(
    X_train, y_train, 
    eval_set=[(X_train, y_train), (X_val, y_val), (X_test, y_test)], 
    verbose=False
)

# %% [markdown]
# ## 4. Plotting Log Loss Curves

results = xgb_clf.evals_result()
epochs = len(results['validation_0']['mlogloss'])
x_axis = range(0, epochs)

plt.figure(figsize=(10, 6))
plt.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
plt.plot(x_axis, results['validation_1']['mlogloss'], label='Validation')
plt.plot(x_axis, results['validation_2']['mlogloss'], label='Test')

plt.title('XGBoost Multi-Class Log Loss Curve')
plt.ylabel('Log Loss')
plt.xlabel('Iterations')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(folder, 'regression_classification_logloss_curve.png'))
plt.close()

# Final metrics
train_loss = log_loss(y_train, xgb_clf.predict_proba(X_train))
val_loss   = log_loss(y_val,   xgb_clf.predict_proba(X_val))
test_loss  = log_loss(y_test,  xgb_clf.predict_proba(X_test))

print("\nFinal Log Loss:")
print(f"  Train : {train_loss:.6f}")
print(f"  Val   : {val_loss:.6f}")
print(f"  Test  : {test_loss:.6f}")
print(f"Curve plot generated: {os.path.join(folder, 'regression_classification_logloss_curve.png')}")
