import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, roc_auc_score
)
import joblib
import warnings
import os

warnings.filterwarnings('ignore')

# %% [markdown]
# ## 1. Data Loading

# Set base path to the regression folder
folder = os.path.dirname(__file__)
train_df = pd.read_csv(os.path.join(folder, 'train.csv'))
val_df   = pd.read_csv(os.path.join(folder, 'valid.csv'))
test_df  = pd.read_csv(os.path.join(folder, 'test.csv'))

print(f"Train : {train_df.shape}")
print(f"Val   : {val_df.shape}")
print(f"Test  : {test_df.shape}")

# %% [markdown]
# ## 2. Exploratory Data Analysis (EDA)

def perform_eda(df, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Class Distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='type', order=df['type'].value_counts().index)
    plt.title('Attack Type Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'attack_types_distribution.png'))
    plt.close()

    # Binary Label Distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='label')
    plt.title('Normal vs Attack Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'label_distribution.png'))
    plt.close()

    # Correlation Heatmap for numerical features
    num_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(14, 12))
    sns.heatmap(num_df.corr(), annot=False, cmap='coolwarm')
    plt.title('Numerical Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'correlation_heatmap.png'))
    plt.close()

print("\nPerforming EDA...")
perform_eda(train_df, folder)

# %% [markdown]
# ## 3. Preprocessing Pipeline

def _ip_first_octet(ip_str):
    try:
        return int(str(ip_str).split('.')[0])
    except Exception:
        return 0

def preprocess_data(df, is_train=True, encoders=None, scaler=None):
    df = df.copy()

    # Identify Target (we will predict 'duration' for regression)
    target_col = 'duration'
    y = df[target_col].astype(float)
    
    # Drop target-related columns from features
    X = df.drop(columns=[target_col, 'label', 'type'], errors='ignore')

    # Handle '-' and missing values
    X = X.replace('-', np.nan)
    
    # IP Features (Extract first octet)
    for col, prefix in [('src_ip', 'src'), ('dst_ip', 'dst')]:
        if col in X.columns:
            X[f'{prefix}_octet1'] = X[col].apply(_ip_first_octet)
            X = X.drop(columns=[col])

    # Categorical Encoding
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    if is_train:
        encoders = {}
        # Log-transform target for regression
        y = np.log1p(y)  # Log(1 + duration) to handle zeros
        
        # Feature Encoding
        for col in cat_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str).fillna('unknown'))
            encoders[col] = le
    else:
        # Log-transform target for regression
        y = np.log1p(y)
        
        # Feature Encoding
        for col in cat_cols:
            if col in encoders:
                X[col] = X[col].astype(str).fillna('unknown').apply(
                    lambda x: encoders[col].transform([x])[0]
                    if x in encoders[col].classes_ else -1
                )
            else:
                X[col] = 0 # Placeholder for unknown categories

    # Ensure all columns are numeric
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce')
    
    X = X.fillna(0)

    # Scaling
    if is_train:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    return X_scaled, y, encoders, scaler

print("\nPreprocessing data...")
X_train, y_train, encoders, scaler = preprocess_data(train_df, is_train=True)
X_val,   y_val,   _,        _      = preprocess_data(val_df,   is_train=False, encoders=encoders, scaler=scaler)
X_test,  y_test,  _,        _      = preprocess_data(test_df,  is_train=False, encoders=encoders, scaler=scaler)

print(f"Features -> Train:{X_train.shape}  Val:{X_val.shape}  Test:{X_test.shape}")

# %% [markdown]
# ## 4. Model Development & Training

print("\nTraining models...")

# XGBoost
xgb_clf = XGBRegressor(
    n_estimators=100, 
    learning_rate=0.1, 
    max_depth=5, 
    random_state=42, 
    n_jobs=-1,
    eval_metric='rmse'
)
xgb_clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# Random Forest
rf_clf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
rf_clf.fit(X_train, y_train)

# %% [markdown]
# ## 5. Evaluation

def evaluate_model(model, X, y, model_name, split_name='Test', output_folder='.'):
    print(f"\n--- {model_name} | {split_name} ---")
    y_pred = model.predict(X)
    
    # Inverse log transform for evaluation
    y_actual = np.expm1(y)
    y_pred_actual = np.expm1(y_pred)
    
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    r2 = r2_score(y_actual, y_pred_actual)
    mae = mean_absolute_error(y_actual, y_pred_actual)
    mse = mean_squared_error(y_actual, y_pred_actual)
    
    print(f"R² Score: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    
    # Actual vs Predicted plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_actual, y_pred_actual, alpha=0.5)
    plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--')
    plt.xlabel('Actual Duration')
    plt.ylabel('Predicted Duration')
    plt.title(f'Actual vs Predicted: {model_name} ({split_name})')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'{model_name}_actual_vs_predicted.png'))
    plt.close()

evaluate_model(xgb_clf, X_test, y_test, 'XGBoost', split_name='Test', output_folder=folder)
evaluate_model(rf_clf,  X_test, y_test, 'Random_Forest', split_name='Test', output_folder=folder)

# %% [markdown]
# ## 6. Saving Artifacts

print("\nSaving artifacts...")
joblib.dump(xgb_clf, os.path.join(folder, 'regression_xgb_model.pkl'))
joblib.dump(rf_clf,  os.path.join(folder, 'regression_rf_model.pkl'))
joblib.dump(encoders, os.path.join(folder, 'regression_encoders.pkl'))
joblib.dump(scaler,   os.path.join(folder, 'regression_scaler.pkl'))
print("Pipeline complete.")
