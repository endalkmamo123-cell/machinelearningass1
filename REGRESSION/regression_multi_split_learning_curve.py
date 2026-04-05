import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
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

def prepare_regression_data(df, is_train=True, encoders=None, scaler=None):
    df = df.copy()
    y = np.log1p(df['duration'])
    X = df.drop(columns=['duration', 'label', 'type'], errors='ignore')
    X = X.replace('-', np.nan)

    for col, prefix in [('src_ip', 'src'), ('dst_ip', 'dst')]:
        if col in X.columns:
            X[f'{prefix}_octet1'] = X[col].apply(_ip_first_octet)
            X = X.drop(columns=[col])

    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    if is_train:
        encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str).fillna('unknown'))
            encoders[col] = le
    else:
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
    
    return X_scaled, y, encoders, scaler

print("Preparing all splits for regression analysis...")
X_train, y_train, encoders, scaler = prepare_regression_data(train_df, is_train=True)
X_val,   y_val,   _,        _      = prepare_regression_data(val_df,   is_train=False, encoders=encoders, scaler=scaler)
X_test,  y_test,  _,        _      = prepare_regression_data(test_df,  is_train=False, encoders=encoders, scaler=scaler)

# %% [markdown]
# ## 3. Triple Convergence Analysis

train_sizes = np.linspace(0.1, 1.0, 5)
train_scores = []
val_scores = []
test_scores = []

print("Simulating triple convergence curve (Train, Val, Test)...")
for size in train_sizes:
    # Sample training data
    n_samples = int(size * len(X_train))
    X_subset = X_train[:n_samples]
    y_subset = y_train[:n_samples]
    
    model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
    model.fit(X_subset, y_subset)
    
    # Evaluate
    train_scores.append(mean_squared_error(y_subset, model.predict(X_subset)))
    val_scores.append(mean_squared_error(y_val, model.predict(X_val)))
    test_scores.append(mean_squared_error(y_test, model.predict(X_test)))
    print(f" Size: {n_samples:>6} | Train MSE: {train_scores[-1]:.4f} | Val MSE: {val_scores[-1]:.4f} | Test MSE: {test_scores[-1]:.4f}")

# %% [markdown]
# ## 4. Plotting

plt.figure(figsize=(10, 6))
plt.plot(train_sizes * 100, train_scores, 'o-', color="r", label="Training Line")
plt.plot(train_sizes * 100, val_scores,   's-', color="g", label="Validation Line")
plt.plot(train_sizes * 100, test_scores,  '^-', color="b", label="Testing Line")

plt.title("Regression Triple-Split Learning Curve (Log Target)")
plt.xlabel("Training Set Size (%)")
plt.ylabel("Mean Squared Error (MSE)")
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(folder, 'regression_triple_split_learning_curve.png'))
plt.close()

print(f"Triple split graph generated: {os.path.join(folder, 'regression_triple_split_learning_curve.png')}")
