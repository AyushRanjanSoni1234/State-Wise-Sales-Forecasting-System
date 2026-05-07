from xgboost import XGBRegressor
from src.models.base_model import BaseModel
from src.config import XGB_PARAMS, TARGET_COLUMN
from src.features import FeatureEngineer

class XgbModel(BaseModel):
    def __init__(self):
        super().__init__("xgboost")
        self.features = FeatureEngineer.get_feature_names()

    def train(self, train_data, **kwargs):
        self.model = XGBRegressor(**XGB_PARAMS)
        self.model.fit(train_data[self.features], train_data[TARGET_COLUMN])

    def predict(self, test_data, **kwargs):
        return self.model.predict(test_data[self.features])
