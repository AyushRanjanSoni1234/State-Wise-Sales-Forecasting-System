from statsmodels.tsa.statespace.sarimax import SARIMAX
from src.models.base_model import BaseModel
from src.config import SARIMA_ORDER, SARIMA_SEASONAL_ORDER, TARGET_COLUMN

class SarimaModel(BaseModel):
    def __init__(self):
        super().__init__("sarima")

    def train(self, train_data, **kwargs):
        self.model = SARIMAX(
            train_data[TARGET_COLUMN],
            order=SARIMA_ORDER,
            seasonal_order=SARIMA_SEASONAL_ORDER
        ).fit(disp=False)

    def predict(self, test_data, **kwargs):
        steps = len(test_data)
        return self.model.forecast(steps=steps)
