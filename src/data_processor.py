import pandas as pd
import numpy as np
from src.config import DATA_PATH, TARGET_COLUMN, DATE_COLUMN, STATE_COLUMN

class DataProcessor:
    def __init__(self, file_path=DATA_PATH):
        self.file_path = file_path

    def load_data(self):
        df = pd.read_excel(self.file_path)
        return self.preprocess(df)

    def preprocess(self, df):
        # Rename columns to match config
        df = df.rename(columns={
            'State': STATE_COLUMN,
            'Date': DATE_COLUMN,
            'Total': TARGET_COLUMN,
            'Category': 'category'
        })
        
        # Convert date column
        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], format='mixed', dayfirst=True)
        
        # Sort and drop duplicates
        df = df.sort_values([STATE_COLUMN, DATE_COLUMN])
        df = df.drop_duplicates()
        
        return df

    def get_state_data(self, df, state_name=None):
        # Group by state and date to get total sales per state per day
        state_df = df.groupby([STATE_COLUMN, DATE_COLUMN])[TARGET_COLUMN].sum().reset_index()
        
        if state_name:
            state_df = state_df[state_df[STATE_COLUMN] == state_name]
            
        return state_df

    def fill_missing_dates(self, df):
        def _fill_group(group):
            group = group.set_index(DATE_COLUMN).asfreq('D')
            group[TARGET_COLUMN] = group[TARGET_COLUMN].ffill()
            group[STATE_COLUMN] = group[STATE_COLUMN].iloc[0]
            return group.reset_index()

        return df.groupby(STATE_COLUMN).apply(_fill_group).reset_index(drop=True)

    def split_data(self, df, test_size=0.2):
        split_index = int(len(df) * (1 - test_size))
        train = df.iloc[:split_index]
        test = df.iloc[split_index:]
        return train, test
