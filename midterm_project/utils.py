import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


class DataProcessor:
    def __init__(self):
        self.humidity_medians_ = None
        self.wind_imputer_ = None
        self.scaler_ = None
        self.one_hot_columns_ = None
        self.wind_features = ['season', 'weather', 'humidity', 'temp', 'month', 'year']
        self.categorical_features = ['season', 'holiday', 'workingday', 'weather',
                                     'hour', 'month', 'year', 'dayofweek']

    def _engineer_base_features(self, df):
        """Helper to create date/time features."""
        df_eng = df.copy()
        df_eng['datetime'] = pd.to_datetime(df_eng['datetime'])
        df_eng['hour'] = df_eng['datetime'].dt.hour
        df_eng['month'] = df_eng['datetime'].dt.month
        df_eng['year'] = df_eng['datetime'].dt.year
        df_eng['dayofweek'] = df_eng['datetime'].dt.dayofweek
        # 'day' is needed for CV split, and 'transform' needs it
        df_eng['day'] = df_eng['datetime'].dt.day
        return df_eng

    def fit(self, df):
        """
        Learns all imputation models, scalers, and column formats from the
        training data.
        """
        print("Fitting DataProcessor...")

        # --- 1. Engineer base features FIRST ---
        # This creates 'month', 'year', etc.
        df_proc = self._engineer_base_features(df)

        # --- 2. Learn Imputations (now 'month' and 'year' exist) ---
        # 1. Humidity
        self.humidity_medians_ = df_proc[df_proc['humidity'] > 0].groupby('season')['humidity'].median()

        # 2. Windspeed
        wind_train_df = df_proc[df_proc['windspeed'] > 0]
        self.wind_imputer_ = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.wind_imputer_.fit(wind_train_df[self.wind_features], wind_train_df['windspeed'])

        # --- 3. Learn Scaling & Columns ---
        # Apply all transformations to the training data to create the "final"
        # feature set that the scaler and column list will be based on.
        df_transformed = self._apply_transformations(df_proc)

        self.scaler_ = StandardScaler()
        # Don't scale 'day'
        self.scaler_.fit(df_transformed.drop(columns=['day']))

        # Store all final column names to ensure consistency
        self.one_hot_columns_ = df_transformed.drop(columns=['day']).columns
        return self

    def transform(self, df, scaled=False):
        """
        Applies all learned transformations to new data.
        """
        if self.humidity_medians_ is None or self.wind_imputer_ is None:
            raise Exception("Processor has not been fitted. Call .fit() first.")

        # 1. Create base features
        df_transformed = self._engineer_base_features(df)

        # 2. Apply all learned transformations
        df_final = self._apply_transformations(df_transformed)

        # 3. Align columns with training data
        # Check if 'day' exists (it will for train, not for test from predict.py)
        if 'day' in df_final.columns:
            df_day = df_final['day']
            df_final = df_final.drop(columns=['day'])
        else:
            df_day = None

        df_final = df_final.reindex(columns=self.one_hot_columns_, fill_value=0)

        # 4. Apply scaling if requested
        if scaled:
            df_final = self.scaler_.transform(df_final)

        # Add 'day' back if it existed (for the CV split in train.py)
        if df_day is not None:
            df_final['day'] = df_day

        return df_final

    def _apply_transformations(self, df):
        """Internal helper to apply imputations, drops, and encoding."""

        # 1. Bin 'weather'
        df['weather'] = df['weather'].replace(4, 3)

        # 2. Apply Imputations
        # Humidity
        for season in self.humidity_medians_.index:
            df.loc[
                (df['humidity'] == 0) & (df['season'] == season), 'humidity'
            ] = self.humidity_medians_[season]

        # Windspeed
        wind_test_df = df[df['windspeed'] == 0]
        if not wind_test_df.empty:
            predicted_windspeed = self.wind_imputer_.predict(wind_test_df[self.wind_features])
            df.loc[df['windspeed'] == 0, 'windspeed'] = predicted_windspeed

        # 3. Drop original/unused columns
        cols_to_drop = ['datetime', 'atemp']
        df = df.drop(columns=cols_to_drop, errors='ignore')

        # 4. One-hot encoding
        df = pd.get_dummies(df, columns=self.categorical_features, drop_first=False)

        return df