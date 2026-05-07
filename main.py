import argparse
from src.data_processor import DataProcessor
from src.features import FeatureEngineer
from src.trainer import ModelTrainer

def run_training_pipeline(state_name=None):
    print("Starting Production Forecasting Pipeline...")
    
    # 1. Load and preprocess data
    processor = DataProcessor()
    df = processor.load_data()
    
    # 2. Get state specific data and fill missing dates
    state_df = processor.get_state_data(df, state_name)
    state_df = processor.fill_missing_dates(state_df)
    
    # 3. Feature Engineering
    fe = FeatureEngineer()
    data_with_features = fe.create_features(state_df)
    
    # 4. Split data
    train, test = processor.split_data(data_with_features)
    
    # 5. Train and compare models
    trainer = ModelTrainer()
    best_model, mae = trainer.train_and_compare(train, test)
    
    print(f"Pipeline completed! Best Model: {best_model} (MAE: {mae:,.2f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Production Forecasting System")
    parser.add_argument("--train", action="store_true", help="Run the training pipeline")
    parser.add_argument("--state", type=str, default="Alabama", help="State to train for")
    
    args = parser.parse_args()
    
    if args.train:
        run_training_pipeline(args.state)
    else:
        print("Use --train to run the training pipeline.")
