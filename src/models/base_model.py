from abc import ABC, abstractmethod
import joblib
import os
from src.config import MODELS_DIR

class BaseModel(ABC):
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None

    @abstractmethod
    def train(self, train_data, **kwargs):
        pass

    @abstractmethod
    def predict(self, test_data, **kwargs):
        pass

    def save(self):
        path = os.path.join(MODELS_DIR, f"{self.model_name}.joblib")
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")

    def load(self):
        path = os.path.join(MODELS_DIR, f"{self.model_name}.joblib")
        if os.path.exists(path):
            self.model = joblib.load(path)
            print(f"Model loaded from {path}")
        else:
            print(f"No model found at {path}")
