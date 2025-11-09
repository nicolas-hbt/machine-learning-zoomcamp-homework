# train.py
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import PredefinedSplit, GridSearchCV
import pickle
from utils import DataProcessor
import warnings

warnings.filterwarnings('ignore')

print("Starting training script...")

# --- 1. Load Data ---
train_df = pd.read_csv('train.csv')
y_train_log = np.log1p(train_df['count'])

# --- 2. Create the Validation Split FIRST ---
# We must do this before fitting the processor to prevent leakage
print("Creating validation split...")
# Engineer 'day' feature just for the split
datetime_s = pd.to_datetime(train_df['datetime'])
train_days = datetime_s.dt.day
split_indices = [-1 if d <= 15 else 0 for d in train_days]
ps = PredefinedSplit(test_fold=split_indices)

# Get the indices for the training fold (days 1-15)
train_fold_indices = np.where(np.array(split_indices) == -1)[0]
train_fold_df = train_df.iloc[train_fold_indices]

print("Validation split created.")

# --- 3. Fit DataProcessor (Leak-Free) ---
# Fit the processor ONLY on the training fold (days 1-15)
processor = DataProcessor()
print("Fitting processor on training fold (days 1-15)...")
processor.fit(train_fold_df)

# --- 4. Transform ALL Training Data ---
# Now, transform the entire train_df (days 1-19)
print("Transforming entire training set (days 1-19)...")
X_train_processed = processor.transform(train_df)

# We can now drop the 'day' column, which transform() adds
X_train_processed = X_train_processed.drop(columns=['day'])

print("Data processing complete.")

# --- 5. Model Training & Tuning ---
print("Tuning Gradient Boosting Regressor...")
gbr = GradientBoostingRegressor(random_state=42)
gbr_params = {
    'n_estimators': [100, 250, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7]
}

# Pass the full X_train_processed, y_train_log, and our custom 'ps' split
grid_gbr = GridSearchCV(estimator=gbr, param_grid=gbr_params, cv=ps,
                        scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)

grid_gbr.fit(X_train_processed, y_train_log)

best_rmsle = np.sqrt(-grid_gbr.best_score_)
print(f"Best GBR RMSLE: {best_rmsle:.5f}")
print(f"Best GBR Params: {grid_gbr.best_params_}")

# --- 6. Save Artifacts (Retrain on all data) ---
# For the final model, we retrain on ALL train data
print("Retraining final model on all training data (days 1-19)...")
processor_final = DataProcessor()
processor_final.fit(train_df) # Fit on all data
X_train_final = processor_final.transform(train_df).drop(columns=['day'])

# Use the best params found by GridSearchCV
final_model = GradientBoostingRegressor(random_state=42, **grid_gbr.best_params_)
final_model.fit(X_train_final, y_train_log)

print("Saving artifacts...")
with open('processor.pkl', 'wb') as f:
    pickle.dump(processor_final, f)

with open('model.pkl', 'wb') as f:
    pickle.dump(final_model, f)

print("Training complete. 'processor.pkl' and 'model.pkl' saved.")