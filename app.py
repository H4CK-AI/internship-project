import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Bike Rental Demand Prediction",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'models' not in st.session_state:
        st.session_state.models = {}
    if 'predictions' not in st.session_state:
        st.session_state.predictions = {}
    if 'model_metrics' not in st.session_state:
        st.session_state.model_metrics = {}

    # Main header
    st.markdown('<h1 class="main-header">🚲 Bike Rental Demand Prediction System</h1>', unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    ## Welcome to the Bike Rental Demand Prediction System
    
    This comprehensive application helps you predict bike rental demand using advanced time-series forecasting models including:
    - **ARIMA** (AutoRegressive Integrated Moving Average)
    - **SARIMA** (Seasonal ARIMA)

    
    ### How to Use This Application:
    1. **📊 Data Exploration**: Upload your bike rental dataset and explore the data
    2. **🔮 Model Training**: Train different forecasting models with your data
    3. **📈 Predictions**: Generate real-time predictions and forecasts
    4. **📋 Model Comparison**: Compare model performance and select the best one
    """)
    
    # Sidebar
    st.sidebar.title("Navigation")
    st.sidebar.markdown("""
    Use the pages in the sidebar to navigate through different sections:
    - **Data Exploration**: Upload and analyze your dataset
    - **Model Training**: Train ARIMA, SARIMA, and LSTM models
    - **Predictions**: Generate forecasts and predictions
    - **Model Comparison**: Compare model performance
    """)
    
    # Data upload section
    st.header("📁 Data Upload")
    
    uploaded_file = st.file_uploader(
        "Upload your bike rental dataset (CSV format)",
        type=['csv'],
        help="Upload a CSV file containing bike rental data with datetime and count columns"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            data = pd.read_csv(uploaded_file)
            st.session_state.data = data
            
            st.success(f"✅ Dataset uploaded successfully! Shape: {data.shape}")
            
            # Display basic information
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Records", len(data))
            with col2:
                st.metric("Columns", len(data.columns))
            with col3:
                st.metric("Missing Values", data.isnull().sum().sum())
            with col4:
                st.metric("Duplicate Records", data.duplicated().sum())
            
            # Display sample data
            st.subheader("📋 Sample Data")
            st.dataframe(data.head(10), use_container_width=True)
            
            # Data info
            st.subheader("📊 Dataset Information")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Column Types:**")
                st.write(data.dtypes)
            
            with col2:
                st.write("**Statistical Summary:**")
                st.write(data.describe())
            
        except Exception as e:
            st.error(f"❌ Error loading dataset: {str(e)}")
            st.info("Please ensure your CSV file has the correct format with datetime and count columns.")
    else:
        # If no file is uploaded, load hour.csv by default
        try:
            data = pd.read_csv("hour.csv")
            st.session_state.data = data
            st.info("No file uploaded. Using default dataset: hour.csv")
            
            # Display basic information
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", len(data))
            with col2:
                st.metric("Columns", len(data.columns))
            with col3:
                st.metric("Missing Values", data.isnull().sum().sum())
            with col4:
                st.metric("Duplicate Records", data.duplicated().sum())
            
            # Display sample data
            st.subheader("📋 Sample Data")
            st.dataframe(data.head(10), use_container_width=True)
            
            # Data info
            st.subheader("📊 Dataset Information")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Column Types:**")
                st.write(data.dtypes)
            with col2:
                st.write("**Statistical Summary:**")
                st.write(data.describe())
        except Exception as e:
            st.session_state.data = None
            st.error(f"❌ Error loading default dataset hour.csv: {str(e)}")
            st.info("Please upload a CSV file to get started.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; margin-top: 2rem;'>
        <p>Built with ❤️ using Streamlit | Bike Rental Demand Prediction System</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
