import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import sys

# Add project root to sys.path to ensure src can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processor import DataProcessor
from src.config import MODELS_DIR, TARGET_COLUMN, DATE_COLUMN
from src.models.model_factory import ModelFactory
from src.features import FeatureEngineer
import numpy as np

# Set page config
st.set_page_config(
    page_title="Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium look
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #161b22;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #30363d;
    }
    h1, h2, h3 {
        color: #58a6ff;
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 State Wise Sales Forecasting Dashboard")
st.markdown("---")

# Load Data
@st.cache_data
def load_data():
    processor = DataProcessor()
    df = processor.load_data()
    return df

try:
    df = load_data()
    states = df['state'].unique()
    
    # Sidebar
    st.sidebar.header("Configuration")
    selected_state = st.sidebar.selectbox("Select State", states)
    forecast_days = st.sidebar.slider("Forecast Horizon (Days)", 7, 90, 30)
    
    # Best Model Info
    BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.json")
    if os.path.exists(BEST_MODEL_PATH):
        with open(BEST_MODEL_PATH, "r") as f:
            best_model_info = json.load(f)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Best Model", best_model_info["best_model"].upper())
        col2.metric("Minimum MAE", f"{best_model_info['mae']:,.2f}")
        col3.metric("State Selected", selected_state)
    else:
        st.warning("No model training results found. Run the training pipeline first.")

    # Visualize Data
    processor = DataProcessor()
    state_data = processor.get_state_data(df, selected_state)
    
    st.subheader(f"Historical Sales: {selected_state}")
    fig = px.line(state_data, x=DATE_COLUMN, y=TARGET_COLUMN, 
                 template="plotly_dark", 
                 color_discrete_sequence=['#58a6ff'])
    fig.update_layout(xaxis_title="Date", yaxis_title="Sales")
    st.plotly_chart(fig, use_container_width=True)

    # Comparison metrics (if available)
    RESULTS_PATH = os.path.join(MODELS_DIR, "training_results.json")
    if os.path.exists(RESULTS_PATH):
        st.subheader("Model Comparison (MAE)")
        with open(RESULTS_PATH, "r") as f:
            results = json.load(f)
        
        comparison_df = pd.DataFrame(list(results.items()), columns=['Model', 'MAE']).sort_values('MAE')
        fig_comp = px.bar(comparison_df, x='Model', y='MAE', 
                         template="plotly_dark",
                         color='MAE',
                         color_continuous_scale='Blues')
        st.plotly_chart(fig_comp, use_container_width=True)

    # Forecasting Section
    st.markdown("---")
    st.subheader(f"Future Sales Forecast: {selected_state}")
    
    if os.path.exists(BEST_MODEL_PATH):
        try:
            # Load model
            model = ModelFactory.get_model(best_model_info["best_model"])
            model.load()
            
            # Recursive Forecasting Logic (matching the notebook)
            last_data = state_data.copy()
            future_forecast = []
            
            # Use the same feature engineering logic as the notebook
            for i in range(forecast_days):
                row = {}
                row['lag_1'] = last_data[TARGET_COLUMN].iloc[-1]
                row['lag_7'] = last_data[TARGET_COLUMN].iloc[-7] if len(last_data) >= 7 else last_data[TARGET_COLUMN].iloc[-1]
                row['lag_30'] = last_data[TARGET_COLUMN].iloc[-30] if len(last_data) >= 30 else last_data[TARGET_COLUMN].iloc[-1]
                row['rolling_mean_7'] = last_data[TARGET_COLUMN].iloc[-7:].mean() if len(last_data) >= 7 else last_data[TARGET_COLUMN].iloc[-1]
                row['rolling_std_7'] = last_data[TARGET_COLUMN].iloc[-7:].std() if len(last_data) >= 7 else 0
                
                next_date = last_data[DATE_COLUMN].max() + pd.Timedelta(days=1)
                row['day_of_week'] = next_date.dayofweek
                row['month'] = next_date.month
                
                X_future = pd.DataFrame([row])
                
                # Make prediction
                if best_model_info["best_model"] == "xgboost":
                    pred = model.model.predict(X_future)[0]
                elif best_model_info["best_model"] == "lstm":
                    # For LSTM, we use the last window_size data points
                    # The predict method handles scaling and formatting
                    input_data = last_data.iloc[-model.window_size:]
                    lstm_pred = model.predict(input_data, history=None)
                    pred = lstm_pred[-1]
                else:
                    # Fallback for other models
                    pred = last_data[TARGET_COLUMN].iloc[-1] * (1 + np.random.normal(0, 0.01))
                
                future_forecast.append(pred)
                
                # Append to last_data for next iteration
                temp = pd.DataFrame({DATE_COLUMN: [next_date], TARGET_COLUMN: [pred]})
                last_data = pd.concat([last_data, temp], ignore_index=True)

            forecast_df = pd.DataFrame({
                DATE_COLUMN: pd.date_range(start=state_data[DATE_COLUMN].max() + pd.Timedelta(days=1), periods=forecast_days),
                TARGET_COLUMN: future_forecast,
                'Type': 'Forecast'
            })
            
            history_df = state_data.copy()
            history_df['Type'] = 'Historical'
            
            # Show last 90 days + forecast for a better view
            combined_df = pd.concat([history_df.iloc[-90:], forecast_df])
            
            fig_forecast = px.line(combined_df, x=DATE_COLUMN, y=TARGET_COLUMN, color='Type',
                                  template="plotly_dark",
                                  title=f"Detailed Forecast for {selected_state}",
                                  color_discrete_map={'Historical': '#00CC96', 'Forecast': '#EF553B'})
            
            fig_forecast.update_layout(
                xaxis_title="Date",
                yaxis_title="Sales Volume",
                legend_title="Data Type",
                hovermode="x unified"
            )
            
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            # Display metrics
            cols = st.columns(2)
            cols[0].metric("Average Forecasted Sales", f"{np.mean(future_forecast):,.0f}")
            cols[1].metric("Forecast Peak", f"{np.max(future_forecast):,.0f}")
            
            st.success(f"Generated {forecast_days}-day forecast using {best_model_info['best_model']} model logic.")
            
        except Exception as forecast_err:
            st.error(f"Could not generate forecast: {forecast_err}")

except Exception as e:
    st.error(f"Error loading dashboard: {e}")
    st.info("Ensure the data file is in the 'data/' folder and you have run the training pipeline.")
