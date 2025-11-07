# train.py
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import PredefinedSplit, GridSearchCV
import pickle
from preprocessing import DataProcessor # Import our new class

print("Starting training script...")

train_df = pd.read_csv('train.csv')
y_train_log = np.log1p(train_df['count'])

# We use the "Impute All" strategy (which is now built into the class)
processor = DataProcessor()
X_train_processed = processor.fit_transform(train_df)

print("Data processing complete.")

train_days = X_train_processed['day']
split_indices = [-1 if d <= 15 else 0 for d in train_days]
ps = PredefinedSplit(test_fold=split_indices)

X_train_processed = X_train_processed.drop(columns=['day'])

print("Validation split created.")

print("Tuning Gradient Boosting Regressor...")
gbr = GradientBoostingRegressor(random_state=42)
gbr_params = {
    'n_estimators': [100, 250, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 10]
}

grid_gbr = GridSearchCV(estimator=gbr, param_grid=gbr_params, cv=ps,
                        scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)

grid_gbr.fit(X_train_processed, y_train_log)

best_rmsle = np.sqrt(-grid_gbr.best_score_)
print(f"Best GBR RMSLE: {best_rmsle:.5f}")
print(f"Best GBR Params: {grid_gbr.best_params_}")

# Save the *fitted processor* and the *best model*
print("Saving artifacts...")
with open('processor.pkl', 'wb') as f:
    pickle.dump(processor, f)

with open('model.pkl', 'wb') as f:
    pickle.dump(grid_gbr.best_estimator_, f)

print("Training complete. 'processor.pkl' and 'model.pkl' saved.")