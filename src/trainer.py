import pandas as pd
from sklearn.metrics import mean_absolute_error
from src.models.model_factory import ModelFactory
from src.config import TARGET_COLUMN
import json
import os
from src.config import MODELS_DIR

class ModelTrainer:
    def __init__(self, model_types=None):
        if model_types is None:
            self.model_types = ["sarima", "prophet", "xgboost", "lstm"]
        else:
            self.model_types = model_types
        self.results = {}

    def train_and_compare(self, train_data, test_data):
        best_model_name = None
        min_mae = float('inf')
        
        for model_type in self.model_types:
            print(f"Training {model_type}...")
            model = ModelFactory.get_model(model_type)
            try:
                # Train
                model.train(train_data)
                
                # Predict
                # LSTM needs history (last window of train) to predict the start of test
                preds = model.predict(test_data, history=train_data)
                
                # Align lengths for metrics (LSTM might have shorter predictions)
                actuals = test_data[TARGET_COLUMN].values
                if len(preds) != len(actuals):
                     # Simple alignment for comparison
                     common_len = min(len(preds), len(actuals))
                     preds = preds[-common_len:]
                     actuals = actuals[-common_len:]
                
                mae = mean_absolute_error(actuals, preds)
                self.results[model_type] = mae
                print(f"{model_type} MAE: {mae}")
                
                if mae < min_mae:
                    min_mae = mae
                    best_model_name = model_type
                    
            except Exception as e:
                print(f"Error training {model_type}: {e}")

        print(f"Best model: {best_model_name} with MAE: {min_mae}")
        
        # Save results
        with open(os.path.join(MODELS_DIR, "training_results.json"), "w") as f:
            json.dump(self.results, f)
            
        # Save the best model info
        with open(os.path.join(MODELS_DIR, "best_model.json"), "w") as f:
            json.dump({"best_model": best_model_name, "mae": min_mae}, f)
            
        # Actually save the best model artifact
        best_model = ModelFactory.get_model(best_model_name)
        best_model.train(pd.concat([train_data, test_data])) # Retrain on full data
        
        # Save with its specific name
        best_model.save()
        
        # Also save a generic copy for easy access
        import shutil
        if best_model_name == "lstm":
            shutil.copy2(os.path.join(MODELS_DIR, "lstm.h5"), os.path.join(MODELS_DIR, "best_model.h5"))
            shutil.copy2(os.path.join(MODELS_DIR, "lstm_scaler.joblib"), os.path.join(MODELS_DIR, "best_model_scaler.joblib"))
        else:
            shutil.copy2(os.path.join(MODELS_DIR, f"{best_model_name}.joblib"), os.path.join(MODELS_DIR, "best_model.joblib"))
            
        print(f"Final best model ({best_model_name}) trained and saved as 'best_model'.")
        
        return best_model_name, min_mae
