import numpy as np
import os
import pandas as pd
try:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_TF = True
except ImportError:
    HAS_TF = False
    
from sklearn.preprocessing import MinMaxScaler
import joblib
from src.models.base_model import BaseModel
from src.config import LSTM_PARAMS, TARGET_COLUMN, MODELS_DIR

class LstmModel(BaseModel):
    def __init__(self):
        super().__init__("lstm")
        self.scaler = MinMaxScaler()
        self.window_size = LSTM_PARAMS['window_size']
        if not HAS_TF:
            print("[WARNING] TensorFlow not installed. LSTM model will not be available.")

    def _create_sequences(self, data):
        X, y = [], []
        for i in range(self.window_size, len(data)):
            X.append(data[i-self.window_size:i])
            y.append(data[i])
        return np.array(X), np.array(y)

    def train(self, train_data, **kwargs):
        if not HAS_TF:
            raise ImportError("TensorFlow is required for LSTM model training.")
        scaled_data = self.scaler.fit_transform(train_data[[TARGET_COLUMN]])
        X, y = self._create_sequences(scaled_data)
        
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        self.model = Sequential([
            LSTM(64, activation='relu', return_sequences=True, input_shape=(self.window_size, 1)),
            LSTM(32, activation='relu'),
            Dense(1)
        ])
        
        self.model.compile(optimizer='adam', loss='mse')
        
        early_stop = EarlyStopping(
            monitor='val_loss', 
            patience=LSTM_PARAMS['early_stopping_patience'], 
            restore_best_weights=True
        )
        
        self.model.fit(
            X, y, 
            validation_split=0.2,
            epochs=LSTM_PARAMS['epochs'], 
            batch_size=LSTM_PARAMS['batch_size'], 
            callbacks=[early_stop], 
            verbose=0
        )

    def predict(self, test_data, history=None, **kwargs):
        if not HAS_TF:
            print("LSTM prediction skipped (TensorFlow missing)")
            return np.zeros(len(test_data))
            
        # To predict the first value of test_data, we need the last window_size values from history
        if history is not None:
            combined = pd.concat([history[[TARGET_COLUMN]], test_data[[TARGET_COLUMN]]])
            scaled_combined = self.scaler.transform(combined)
            X_test = []
            for i in range(self.window_size, len(scaled_combined)):
                X_test.append(scaled_combined[i-self.window_size:i])
            X_test = np.array(X_test)
        else:
            scaled_test = self.scaler.transform(test_data[[TARGET_COLUMN]])
            if len(scaled_test) == self.window_size:
                # Handle single window case for recursive forecasting
                X_test = np.array([scaled_test])
            else:
                X_test, _ = self._create_sequences(scaled_test)
            
        if len(X_test) == 0:
            return np.zeros(len(test_data))
        
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        preds = self.model.predict(X_test, verbose=0)
        
        preds_unscaled = self.scaler.inverse_transform(preds).flatten()
        
        # Ensure we return a result of the same length as test_data
        result = np.zeros(len(test_data))
        n_preds = min(len(preds_unscaled), len(test_data))
        result[-n_preds:] = preds_unscaled[-n_preds:]
        
        # If we didn't have enough history, fill the beginning with the first prediction
        if n_preds < len(test_data):
            result[:len(test_data)-n_preds] = preds_unscaled[0]
            
        return result

    def save(self):
        model_path = os.path.join(MODELS_DIR, f"{self.model_name}.h5")
        scaler_path = os.path.join(MODELS_DIR, f"{self.model_name}_scaler.joblib")
        self.model.save(model_path)
        joblib.dump(self.scaler, scaler_path)
        print(f"LSTM model saved to {model_path}")

    def load(self):
        if not HAS_TF:
            return
        model_path = os.path.join(MODELS_DIR, f"{self.model_name}.h5")
        scaler_path = os.path.join(MODELS_DIR, f"{self.model_name}_scaler.joblib")
        if os.path.exists(model_path):
            self.model = load_model(model_path, compile=False)
            self.scaler = joblib.load(scaler_path)
            print(f"LSTM model loaded from {model_path}")
