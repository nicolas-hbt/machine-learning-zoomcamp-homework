# preprocessing.py
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

    def fit(self, df):
        """
        Learns all imputation models, scalers, and column formats from the
        training data.
        """
        df_proc = df.copy()

        self.humidity_medians_ = df_proc[df_proc['humidity'] > 0].groupby('season')['humidity'].median()

        wind_train_df = df_proc[df_proc['windspeed'] > 0]
        self.wind_imputer_ = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.wind_imputer_.fit(wind_train_df[self.wind_features], wind_train_df['windspeed'])

        df_transformed = self._transform_features(df_proc, 'fit')

        self.scaler_ = StandardScaler()
        self.scaler_.fit(df_transformed)

        self.one_hot_columns_ = df_transformed.columns
        return self

    def transform(self, df, scaled=False):
        """
        Applies all learned transformations to new data.
        """
        if self.humidity_medians_ is None or self.wind_imputer_ is None:
            raise Exception("Processor has not been fitted. Call .fit() first.")

        df_transformed = self._transform_features(df.copy(), 'transform')

        # Ensure all one-hot columns are present
        df_transformed = df_transformed.reindex(columns=self.one_hot_columns_, fill_value=0)

        if scaled:
            df_transformed = self.scaler_.transform(df_transformed)

        return df_transformed

    def _transform_features(self, df, mode):
        """Helper function to apply all feature engineering steps."""

        # 1. Date features
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['hour'] = df['datetime'].dt.hour
        df['month'] = df['datetime'].dt.month
        df['year'] = df['datetime'].dt.year
        df['dayofweek'] = df['datetime'].dt.dayofweek
        if mode == 'fit':
            df['day'] = df['datetime'].dt.day  # Keep 'day' for CV splitting

        # 2. Bin 'weather'
        df['weather'] = df['weather'].replace(4, 3)

        # 3. Apply Imputations
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

        # 4. Drop original/unused columns
        cols_to_drop = ['datetime', 'atemp']
        if 'day' not in df.columns:
            cols_to_drop.append('day')
        df = df.drop(columns=cols_to_drop, errors='ignore')

        # 5. One-hot encoding
        categorical_features = ['season', 'holiday', 'workingday', 'weather',
                                'hour', 'month', 'year', 'dayofweek']
        df = pd.get_dummies(df, columns=categorical_features, drop_first=False)

        return df