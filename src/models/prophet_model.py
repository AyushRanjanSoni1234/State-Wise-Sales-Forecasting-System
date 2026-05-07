from prophet import Prophet
from src.models.base_model import BaseModel
from src.config import TARGET_COLUMN, DATE_COLUMN

class ProphetModel(BaseModel):
    def __init__(self):
        super().__init__("prophet")

    def train(self, train_data, **kwargs):
        df = train_data[[DATE_COLUMN, TARGET_COLUMN]].rename(columns={DATE_COLUMN: 'ds', TARGET_COLUMN: 'y'})
        self.model = Prophet()
        self.model.fit(df)

    def predict(self, test_data, **kwargs):
        periods = len(test_data)
        future = self.model.make_future_dataframe(periods=periods, freq='D')
        forecast = self.model.predict(future)
        return forecast['yhat'].iloc[-periods:].values
