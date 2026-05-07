import pandas as pd
from src.config import TARGET_COLUMN, DATE_COLUMN, LAG_PERIODS, ROLLING_WINDOW

class FeatureEngineer:
    @staticmethod
    def create_features(df):
        df = df.copy()
        
        # Ensure date is datetime
        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
        
        # Lag Features
        for lag in LAG_PERIODS:
            df[f'lag_{lag}'] = df[TARGET_COLUMN].shift(lag)
            
        # Rolling Features
        df[f'rolling_mean_{ROLLING_WINDOW}'] = (
            df[TARGET_COLUMN]
            .shift(1)
            .rolling(ROLLING_WINDOW)
            .mean()
        )
        
        df[f'rolling_std_{ROLLING_WINDOW}'] = (
            df[TARGET_COLUMN]
            .shift(1)
            .rolling(ROLLING_WINDOW)
            .std()
        )
        
        # Date Features
        df['day_of_week'] = df[DATE_COLUMN].dt.dayofweek
        df['month'] = df[DATE_COLUMN].dt.month
        
        return df.dropna().reset_index(drop=True)

    @staticmethod
    def get_feature_names():
        features = [f'lag_{lag}' for lag in LAG_PERIODS]
        features += [f'rolling_mean_{ROLLING_WINDOW}', f'rolling_std_{ROLLING_WINDOW}']
        features += ['day_of_week', 'month']
        return features
