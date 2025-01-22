from lightgbm import LGBMRegressor
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('data/processed/train_processed.csv')

# Convert object columns to category
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    df[col] = df[col].astype('category')

X = df.drop(columns=['SQM_PRICE'])
y = df['SQM_PRICE']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LGBMRegressor(
    n_estimators=2000,        # Increase number of boosting rounds
    learning_rate=0.01,       # Lower learning rate for better generalization
    num_leaves=64,            # Increase leaves for better fit
    max_depth=7,              # Limit tree depth
    min_data_in_leaf=50,      # Prevent overfitting
    lambda_l1=0.1,            # L1 regularization
    lambda_l2=1.0,            # L2 regularization
    feature_fraction=0.8,     # Use 80% of features in each iteration
    bagging_fraction=0.8,     # Use 80% of data for bagging
    bagging_freq=5,           # Perform bagging every 5 iterations
    verbosity=-1              # Suppress warnings
)

# Fit model with early stopping
model.fit(
    X_train, y_train, 
    eval_set=[(X_test, y_test)],
    eval_metric='rmse', 
    categorical_feature=categorical_cols, 
)


# Save model
joblib.dump(model, 'model.pkl')

# Predict on test data
y_pred = model.predict(X_test)

# Evaluation metrics
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

# Load validation dataset
df_val = pd.read_csv('data/processed/test_processed.csv')

# Convert object columns to category
for col in df_val.select_dtypes(include=['object']).columns:
    df_val[col] = df_val[col].astype('category')

X_val = df_val.drop(columns=['SQM_PRICE'])
y_val = df_val['SQM_PRICE']

# Predict on validation data
y_pred_val = model.predict(X_val)

# Validation metrics
print("MAE (Validation):", mean_absolute_error(y_val, y_pred_val))
print("RMSE (Validation):", np.sqrt(mean_squared_error(y_val, y_pred_val)))

# Calculate additional accuracy metrics for validation set
all_preds_val = torch.tensor(y_pred_val)
all_targets_val = torch.tensor(y_val.values)

acc_5_val = calculate_accuracy_margin(all_preds_val, all_targets_val, 0.05)
acc_10_val = calculate_accuracy_margin(all_preds_val, all_targets_val, 0.10)
acc_20_val = calculate_accuracy_margin(all_preds_val, all_targets_val, 0.20)

print(f"Accuracy within ±5% (Validation): {acc_5_val:.2f}% (Benchmark: 30.1%)")
print(f"Accuracy within ±10% (Validation): {acc_10_val:.2f}% (Benchmark: 54.1%)")
print(f"Accuracy within ±20% (Validation): {acc_20_val:.2f}% (Benchmark: 81.4%)")
