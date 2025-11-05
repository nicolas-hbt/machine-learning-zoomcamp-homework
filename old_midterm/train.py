import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression


def load_data(filepath='energy_consumption.csv'):
    """Loads the dataset."""
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None


def feature_engineer(df):
    """
    Performs feature engineering based on notebook findings.

    The key feature 'high_occupancy' is derived from the synthetic data generation logic.
    """
    df['high_occupancy'] = np.where(df['occupants'] > 3, 1, 0)

    df['large_building'] = np.where(df['building_size_m2'] > 44, 1, 0)

    df['occupant_density'] = df['occupants'] / df['building_size_m2']

    df = pd.get_dummies(df, columns=['customer_type', 'regions'], drop_first=True)

    return df


def train_and_save_model(df):
    """
    Trains the selected Decision Tree model and saves it using pickle.
    """
    X = df[['high_occupancy']]
    y = df['energy_cost_brl']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    decision_tree_model = DecisionTreeRegressor(random_state=42)
    decision_tree_model.fit(X_train, y_train)

    y_preds_dt = decision_tree_model.predict(X_test)
    dt_rmse = np.sqrt(mean_squared_error(y_test, y_preds_dt))
    dt_r2 = r2_score(y_test, y_preds_dt)

    print("\n--- Final Model Performance (Decision Tree on 'high_occupancy') ---")
    print(f"Test RMSE: {dt_rmse:.2f}")
    print(f"Test R-squared: {dt_r2:.2f}")

    # --- Save Model ---
    filename = 'dt_regressor.pkl'
    try:
        with open(filename, 'wb') as file:
            pickle.dump(decision_tree_model, file)
        print(f"Model successfully saved to {filename}")
    except Exception as e:
        print(f"An error occurred while saving the model: {e}")


if __name__ == "__main__":

    raw_df = load_data()

    if raw_df is not None:
        processed_df = feature_engineer(raw_df)
        train_and_save_model(processed_df)