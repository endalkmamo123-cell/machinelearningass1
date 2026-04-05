import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

# %% [markdown]
# ## 1. Data Loading

folder = os.path.dirname(__file__)
train_df = pd.read_csv(os.path.join(folder, 'train.csv'))
test_df  = pd.read_csv(os.path.join(folder, 'test.csv'))

# %% [markdown]
# ## 2. Data Preparation for Regression

def _ip_first_octet(ip_str):
    try: return int(str(ip_str).split('.')[0])
    except: return 0

def prepare_regression_data(df, target_col='duration', is_train=True, encoders=None, scaler=None):
    df = df.copy()
    
    # Filter extreme outliers in duration for better curve fitting visualization
    if is_train:
        q_upper = df[target_col].quantile(0.95)
        df = df[df[target_col] <= q_upper]

    y = df[target_col]
    X = df.drop(columns=[target_col, 'label', 'type'], errors='ignore')
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

    return X_scaled, y, X, encoders, scaler

print("Preparing data for regression...")
X_train_scaled, y_train, X_train_raw, encoders, scaler = prepare_regression_data(train_df, is_train=True)
X_test_scaled, y_test, X_test_raw, _, _ = prepare_regression_data(test_df, is_train=False, encoders=encoders, scaler=scaler)

# %% [markdown]
# ## 3. Model Training (Curve Fitting)

print("Training XGBoost Regressor...")
xgb_reg = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
xgb_reg.fit(X_train_scaled, y_train)

print("Training Linear Regression Baseline...")
lr_reg = LinearRegression()
lr_reg.fit(X_train_scaled, y_train)

# %% [markdown]
# ## 4. Evaluation & Visualization

def evaluate_regression(model, X, y, model_name):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print(f"\n--- {model_name} Metrics ---")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R2 : {r2:.4f}")
    return y_pred

y_pred_xgb = evaluate_regression(xgb_reg, X_test_scaled, y_test, "XGBoost Regressor")

# 4.1 Actual vs Predicted Plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_xgb, alpha=0.3, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Duration')
plt.ylabel('Predicted Duration')
plt.title('Actual vs Predicted (XGBoost Regressor)')
plt.tight_layout()
plt.savefig(os.path.join(folder, 'regression_actual_vs_predicted.png'))
plt.close()

# 4.2 Regression Curve Fitting (Duration vs Source Bytes)
# We pick a feature with strong correlation, e.g., 'src_bytes' (index needs to be found)
feature_to_plot = 'src_bytes'
if feature_to_plot in X_test_raw.columns:
    plt.figure(figsize=(12, 6))
    
    # Sort by feature for smooth plotting
    sort_idx = X_test_raw[feature_to_plot].argsort()
    x_plot = X_test_raw[feature_to_plot].iloc[sort_idx]
    y_actual = y_test.iloc[sort_idx]
    y_pred_plot = pd.Series(y_pred_xgb).iloc[sort_idx]
    
    plt.scatter(x_plot, y_actual, alpha=0.2, label='Actual Data', color='gray')
    plt.plot(x_plot, y_pred_plot, color='red', lw=2, label='Fitted Regression Curve (XGBoost)')
    
    plt.xlabel('Source Bytes')
    plt.ylabel('Duration')
    plt.title(f'Regression Curve Fitting: Duration vs {feature_to_plot}')
    plt.legend()
    plt.xlim(0, x_plot.quantile(0.99)) # Clip for visualization
    plt.ylim(0, y_actual.quantile(0.99))
    plt.tight_layout()
    plt.savefig(os.path.join(folder, 'regression_curve_fitting.png'))
    plt.close()

# %% [markdown]
# ## 5. Saving Artifacts

print("\nSaving regression artifacts...")
joblib.dump(xgb_reg, os.path.join(folder, 'regression_curve_model.pkl'))
joblib.dump(encoders, os.path.join(folder, 'regression_curve_encoders.pkl'))
joblib.dump(scaler, os.path.join(folder, 'regression_curve_scaler.pkl'))
print("Regression pipeline complete.")
