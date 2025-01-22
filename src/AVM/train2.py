from google.cloud import aiplatform
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np
import torch

# Load dataset
df = pd.read_csv('data/processed/train_processed.csv')

# Convert object columns to category
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype('category')

X = df.drop(columns=['SQM_PRICE'])
y = df['SQM_PRICE']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = XGBRegressor(n_estimators=500, learning_rate=0.05, enable_categorical=True)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'model.pkl')

y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# Calculate additional accuracy metrics
all_preds = torch.tensor(y_pred)
all_targets = torch.tensor(y_test.values)

def calculate_accuracy_margin(preds, targets, margin):
    within_margin = torch.abs(preds - targets) <= (targets * margin)
    return within_margin.float().mean().item() * 100  # Convert to percentage

acc_5 = calculate_accuracy_margin(all_preds, all_targets, 0.05)
acc_10 = calculate_accuracy_margin(all_preds, all_targets, 0.10)
acc_20 = calculate_accuracy_margin(all_preds, all_targets, 0.20)

print(f"Accuracy within ±5%: {acc_5:.2f}% (Benchmark: 30.1%)")
print(f"Accuracy within ±10%: {acc_10:.2f}% (Benchmark: 54.1%)")
print(f"Accuracy within ±20%: {acc_20:.2f}% (Benchmark: 81.4%)")

df_val = pd.read_csv('data/processed/val_processed.csv')

# Convert object columns to category
for col in df_val.select_dtypes(include=['object']).columns:
    df_val[col] = df_val[col].astype('category')
    
    
X_val = df_val.drop(columns=['SQM_PRICE'])
y_val = df_val['SQM_PRICE']

y_pred_val = model.predict(X_val)

print("MAE (Validation):", mean_absolute_error(y_val, y_pred_val))
print("RMSE (Validation):", np.sqrt(mean_squared_error(y_val, y_pred_val)))

# Calculate additional accuracy metrics
all_preds_val = torch.tensor(y_pred_val)
all_targets_val = torch.tensor(y_val.values)

acc_5_val = calculate_accuracy_margin(all_preds_val, all_targets_val, 0.05)
acc_10_val = calculate_accuracy_margin(all_preds_val, all_targets_val, 0.10)
acc_20_val = calculate_accuracy_margin(all_preds_val, all_targets_val, 0.20)

print(f"Accuracy within ±5% (Validation): {acc_5_val:.2f}% (Benchmark: 30.1%)")
print(f"Accuracy within ±10% (Validation): {acc_10_val:.2f}% (Benchmark: 54.1%)")
print(f"Accuracy within ±20% (Validation): {acc_20_val:.2f}% (Benchmark: 81.4%)")

