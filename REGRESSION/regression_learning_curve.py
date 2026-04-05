import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

# %% [markdown]
# ## 1. Data Loading

folder = os.path.dirname(__file__)
train_df = pd.read_csv(os.path.join(folder, 'train.csv'))

# %% [markdown]
# ## 2. Data Preparation (Log Transformation)

def _ip_first_octet(ip_str):
    try: return int(str(ip_str).split('.')[0])
    except: return 0

def prepare_regression_data(df):
    df = df.copy()
    
    # Target: Log Transformation to handle skewness and improve R2
    # We use log1p (log(1+x)) to handle potential zeros
    y = np.log1p(df['duration'])
    
    X = df.drop(columns=['duration', 'label', 'type'], errors='ignore')
    X = X.replace('-', np.nan)

    # IP features
    for col, prefix in [('src_ip', 'src'), ('dst_ip', 'dst')]:
        if col in X.columns:
            X[f'{prefix}_octet1'] = X[col].apply(_ip_first_octet)
            X = X.drop(columns=[col])

    # Categorical Encoding
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str).fillna('unknown'))

    X = X.fillna(0)
    for col in X.columns:
        if X[col].dtype == 'object': X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

print("Preparing data and applying log transformation...")
X_scaled, y = prepare_regression_data(train_df)

# To speed up learning curve, we'll use a representative subset of the data (50k samples)
if len(X_scaled) > 50000:
    indices = np.random.choice(len(X_scaled), 50000, replace=False)
    X_subset = X_scaled[indices]
    y_subset = y.iloc[indices]
else:
    X_subset = X_scaled
    y_subset = y

# %% [markdown]
# ## 3. Learning Curve Analysis

print("Calculating learning curve (this may take a minute)...")
xgb_reg = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)

train_sizes, train_scores, test_scores = learning_curve(
    xgb_reg, X_subset, y_subset, cv=5, 
    scoring='neg_mean_squared_error',
    train_sizes=np.linspace(0.1, 1.0, 5),
    n_jobs=-1
)

# Scores were negative MSE
train_mse_mean = -np.mean(train_scores, axis=1)
train_mse_std  = np.std(train_scores, axis=1)
test_mse_mean  = -np.mean(test_scores, axis=1)
test_mse_std   = np.std(test_scores, axis=1)

# %% [markdown]
# ## 4. Visualization

plt.figure(figsize=(10, 6))
plt.fill_between(train_sizes, train_mse_mean - train_mse_std, train_mse_mean + train_mse_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_mse_mean - test_mse_std, test_mse_mean + test_mse_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_mse_mean, 'o-', color="r", label="Training Score (MSE)")
plt.plot(train_sizes, test_mse_mean, 'o-', color="g", label="Cross-validation Score (MSE)")

plt.title("Regression Learning Curve (Target: Log(Duration))")
plt.xlabel("Training Examples")
plt.ylabel("Mean Squared Error (MSE)")
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(folder, 'regression_learning_curve.png'))
plt.close()

print(f"Learning curve generated: {os.path.join(folder, 'regression_learning_curve.png')}")

# %% [markdown]
# ## 5. Performance Check (Improved Metric)

# Fit on subset to check R2
xgb_reg.fit(X_subset, y_subset)
y_pred = xgb_reg.predict(X_subset)
from sklearn.metrics import r2_score
r2 = r2_score(y_subset, y_pred)
print(f"Improved R2 on log-transformed subset: {r2:.4f}")
