import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "Forecasting Case- Study.xlsx")
MODELS_DIR = os.path.join(BASE_DIR, "models_storage")

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

# Data processing
TARGET_COLUMN = "sales"
DATE_COLUMN = "date"
STATE_COLUMN = "state"

# Feature Engineering
LAG_PERIODS = [1, 7, 30]
ROLLING_WINDOW = 7

# Model Hyperparameters
XGB_PARAMS = {
    'n_estimators': 200,
    'learning_rate': 0.05,
    'max_depth': 5,
    'random_state': 42
}

LSTM_PARAMS = {
    'window_size': 10,
    'epochs': 50,
    'batch_size': 32,
    'early_stopping_patience': 5
}

SARIMA_ORDER = (1, 1, 1)
SARIMA_SEASONAL_ORDER = (1, 1, 1, 7)
