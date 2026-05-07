# State-Wise-Sales-Forecasting-System

A robust, modular, and production-ready time series forecasting pipeline designed to compare multiple algorithms (SARIMA, Prophet, XGBoost, and LSTM) and deploy the best-performing model via a REST API and a Streamlit dashboard.

**Python Version:** 3.10.x

---

## 🚀 Project Overview

This project transitions a monolithic forecasting notebook into a professional software architecture. It automates data preprocessing, feature engineering, model training, and evaluation to identify the most accurate forecasting algorithm for state-wise sales data.

### Key Implementation Details
*   **Modular Architecture**: Logic is split into `src/` (core logic), `api/` (FastAPI), and `app/` (Streamlit).
*   **Model Factory Pattern**: A flexible factory pattern allows for easy comparison and integration of different algorithms.
*   **Recursive Forecasting**: Implemented a recursive multi-day forecasting logic to generate realistic future sales trends.
*   **Automated Best Model Selection**: The pipeline evaluates all models using Mean Absolute Error (MAE) and automatically saves the champion model for production use.
*   **Advanced LSTM Integration**: A deep learning approach using TensorFlow/Keras with sliding window sequences and early stopping.

---

## 📁 Project Structure

```text
TimeSeriesModel/
├── api/                    # FastAPI REST Service
│   └── main.py             # API entry point
├── app/                    # Streamlit Dashboard
│   └── streamlit_app.py    # Visual analytics & Forecasting
├── data/                   # Data storage
│   └── sales_data.xlsx     # Source dataset
├── models_storage/         # Trained model artifacts (.joblib, .h5)
├── src/                    # Core source code
│   ├── models/             # Model wrappers (XGBoost, LSTM, etc.)
│   ├── config.py           # Project configuration & hyperparameters
│   ├── data_processor.py   # Data cleaning & state-wise filtering
│   ├── features.py         # Lag & rolling window feature engineering
│   └── trainer.py          # Model comparison & training pipeline
├── main.py                 # CLI entry point for training
└── requirements.txt        # Project dependencies
```

---

## 🛠️ Setup & Installation

### 1. Environment Setup
It is recommended to use **Python 3.10**. Create a virtual environment and install dependencies:

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows)
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🏃 How to Run

### 1. Training the Pipeline
Run the training script to evaluate all models for a specific state and save the best one:

```bash
python main.py --train --state "Alabama"
```
*Artifacts will be saved in `models_storage/`.*

### 2. Launching the REST API
Expose the best model's predictive power via a high-performance FastAPI service:

```bash
python api/main.py
```
*API will be available at `http://localhost:8000`. Access `/docs` for interactive documentation.*

### 3. Starting the Visual Dashboard
View historical data, model comparisons, and future forecasts in a premium UI:

```bash
streamlit run app/streamlit_app.py
```

---

## 📊 Features Included
*   **State-wise Analysis**: Filter data and train models for specific geographic regions.
*   **Algorithm Comparison**: Real-time MAE benchmarking between SARIMA, Prophet, XGBoost, and LSTM.
*   **Interactive Forecasting**: Adjust the forecast horizon (up to 90 days) directly from the dashboard.
*   **Robust Preprocessing**: Automatic handling of missing values and datetime alignment.
*   **Production Artifacts**: Saves model weights and scalers for immediate deployment.

## Author 

**Name : Ayush Ranjan Soni**
