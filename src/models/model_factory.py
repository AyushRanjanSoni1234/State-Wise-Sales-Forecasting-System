from src.models.sarima_model import SarimaModel
from src.models.prophet_model import ProphetModel
from src.models.xgboost_model import XgbModel
from src.models.lstm_model import LstmModel

class ModelFactory:
    @staticmethod
    def get_model(model_type):
        models = {
            "sarima": SarimaModel,
            "prophet": ProphetModel,
            "xgboost": XgbModel,
            "lstm": LstmModel
        }
        if model_type not in models:
            raise ValueError(f"Unknown model type: {model_type}")
        return models[model_type]()
