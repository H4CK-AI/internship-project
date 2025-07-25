import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

try:
    from utils.models_simple import ARIMAModel, SARIMAModel
    from utils.metrics import ModelEvaluator
    from utils.visualization import ModelVisualizer
    MODELS_AVAILABLE = True
except Exception as e:
    try:
        from utils.models import ARIMAModel, SARIMAModel
        from utils.metrics import ModelEvaluator
        from utils.visualization import ModelVisualizer
        MODELS_AVAILABLE = True
    except Exception as e2:
        ARIMAModel, SARIMAModel = None, None
        ModelEvaluator, ModelVisualizer = None, None
        MODELS_AVAILABLE = False

st.set_page_config(
    page_title="Model Training - Bike Rental Prediction",
    page_icon="🔮",
    layout="wide"
)

def main():
    st.title("🔮 Model Training and Hyperparameter Tuning")
    
    # Check if models are available
    if not MODELS_AVAILABLE:
        st.error("⚠️ Model training is currently unavailable due to library compatibility issues. Please try using only ARIMA/SARIMA models.")
        st.info("This is likely due to TensorFlow compatibility issues. The application will work with statistical models only.")
        return
    
    # Check if data is available
    if not hasattr(st.session_state, 'data') or st.session_state.data is None:
        st.warning("⚠️ No data available. Please upload a dataset on the main page first.")
        st.stop()
    
    # Use processed data if available, otherwise use original data
    if hasattr(st.session_state, 'processed_data') and st.session_state.processed_data is not None:
        data = st.session_state.processed_data.copy()
    else:
        data = st.session_state.data.copy()
    
    # Ensure datetime column is properly formatted
    if 'datetime' in data.columns:
        data['datetime'] = pd.to_datetime(data['datetime'])
        data = data.sort_values('datetime').reset_index(drop=True)
    
    # Data preparation
    st.header("🎯 Data Preparation for Training")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Training Configuration")
        
        # Train-test split
        train_size = st.slider("Training Data Size (%)", 60, 90, 80)
        train_split = train_size / 100
        
        # Target column
        target_column = st.selectbox(
            "Target Column",
            [col for col in data.columns if col != 'datetime'],
            index=0 if 'count' not in data.columns else list(data.columns).index('count') - 1
        )
        
        # Additional feature columns
        feature_columns = st.multiselect(
            "Additional Feature Columns",
            [col for col in data.columns if col not in ['datetime', target_column]],
            default=[]
        )
    
    with col2:
        st.subheader("Data Split Information")
        
        n_total = len(data)
        n_train = int(n_total * train_split)
        n_test = n_total - n_train
        
        st.metric("Total Records", n_total)
        st.metric("Training Records", n_train)
        st.metric("Testing Records", n_test)
        
        if 'datetime' in data.columns:
            st.write(f"**Training Period:** {data['datetime'].iloc[0]} to {data['datetime'].iloc[n_train-1]}")
            st.write(f"**Testing Period:** {data['datetime'].iloc[n_train]} to {data['datetime'].iloc[-1]}")
    
    # Split data
    train_data = data.iloc[:n_train].copy()
    test_data = data.iloc[n_train:].copy()
    
    # Model selection and training
    st.header("🤖 Model Selection and Training")
    
    # Model tabs
    tab1, tab2 = st.tabs(["ARIMA", "SARIMA"])
    
    with tab1:
        st.subheader("ARIMA Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**ARIMA Parameters (p, d, q):**")
            arima_p = st.slider("p (AR order)", 0, 5, 1, key="arima_p")
            arima_d = st.slider("d (Differencing)", 0, 2, 1, key="arima_d")
            arima_q = st.slider("q (MA order)", 0, 5, 1, key="arima_q")
            
            auto_arima = st.checkbox("Auto-select ARIMA parameters", value=True)
        
        with col2:
            st.write("**Training Options:**")
            arima_seasonal = st.checkbox("Consider seasonality", value=False)
            arima_trend = st.selectbox("Trend component", ["add", "mul", None], index=0)
        
        if st.button("🚀 Train ARIMA Model", key="train_arima"):
            try:
                with st.spinner("Training ARIMA model..."):
                    arima_model = ARIMAModel()
                    
                    if auto_arima:
                        model, fitted_values, forecast = arima_model.fit_auto(
                            train_data[target_column],
                            test_steps=len(test_data)
                        )
                    else:
                        model, fitted_values, forecast = arima_model.fit(
                            train_data[target_column],
                            order=(arima_p, arima_d, arima_q),
                            test_steps=len(test_data)
                        )
                    
                    # Store model and predictions
                    st.session_state.models['ARIMA'] = {
                        'model': model,
                        'fitted_values': fitted_values,
                        'forecast': forecast,
                        'train_data': train_data[target_column],
                        'test_data': test_data[target_column]
                    }
                    
                    # Calculate metrics
                    evaluator = ModelEvaluator()
                    metrics = evaluator.calculate_metrics(
                        test_data[target_column].values,
                        forecast[:len(test_data)]
                    )
                    st.session_state.model_metrics['ARIMA'] = metrics
                    
                    st.success("✅ ARIMA model trained successfully!")
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("MAE", f"{metrics['mae']:.2f}")
                    with col2:
                        st.metric("RMSE", f"{metrics['rmse']:.2f}")
                    with col3:
                        st.metric("MAPE", f"{metrics['mape']:.2f}%")
                    with col4:
                        st.metric("R²", f"{metrics['r2']:.3f}")
                    
                    # Plot results
                    visualizer = ModelVisualizer()
                    fig = visualizer.plot_forecast_results(
                        train_data['datetime'] if 'datetime' in train_data.columns else None,
                        test_data['datetime'] if 'datetime' in test_data.columns else None,
                        train_data[target_column],
                        test_data[target_column],
                        forecast[:len(test_data)],
                        "ARIMA Model Results"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.error(f"❌ Error training ARIMA model: {str(e)}")
    
    with tab2:
        st.subheader("SARIMA Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**SARIMA Parameters (p, d, q):**")
            sarima_p = st.slider("p (AR order)", 0, 3, 1, key="sarima_p")
            sarima_d = st.slider("d (Differencing)", 0, 2, 1, key="sarima_d")
            sarima_q = st.slider("q (MA order)", 0, 3, 1, key="sarima_q")
            
            st.write("**Seasonal Parameters (P, D, Q, s):**")
            sarima_P = st.slider("P (Seasonal AR)", 0, 2, 1, key="sarima_P")
            sarima_D = st.slider("D (Seasonal Diff)", 0, 1, 1, key="sarima_D")
            sarima_Q = st.slider("Q (Seasonal MA)", 0, 2, 1, key="sarima_Q")
            sarima_s = st.slider("s (Seasonal period)", 4, 24, 12, key="sarima_s")
        
        with col2:
            st.write("**Training Options:**")
            auto_sarima = st.checkbox("Auto-select SARIMA parameters", value=True, key="auto_sarima")
            sarima_enforce_stationarity = st.checkbox("Enforce stationarity", value=True)
            sarima_enforce_invertibility = st.checkbox("Enforce invertibility", value=True)
        
        if st.button("🚀 Train SARIMA Model", key="train_sarima"):
            try:
                with st.spinner("Training SARIMA model..."):
                    sarima_model = SARIMAModel()
                    
                    if auto_sarima:
                        model, fitted_values, forecast = sarima_model.fit_auto(
                            train_data[target_column],
                            seasonal_period=sarima_s,
                            test_steps=len(test_data)
                        )
                    else:
                        model, fitted_values, forecast = sarima_model.fit(
                            train_data[target_column],
                            order=(sarima_p, sarima_d, sarima_q),
                            seasonal_order=(sarima_P, sarima_D, sarima_Q, sarima_s),
                            test_steps=len(test_data)
                        )
                    
                    # Store model and predictions
                    st.session_state.models['SARIMA'] = {
                        'model': model,
                        'fitted_values': fitted_values,
                        'forecast': forecast,
                        'train_data': train_data[target_column],
                        'test_data': test_data[target_column]
                    }
                    
                    # Calculate metrics
                    evaluator = ModelEvaluator()
                    metrics = evaluator.calculate_metrics(
                        test_data[target_column].values,
                        forecast[:len(test_data)]
                    )
                    st.session_state.model_metrics['SARIMA'] = metrics
                    
                    st.success("✅ SARIMA model trained successfully!")
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("MAE", f"{metrics['mae']:.2f}")
                    with col2:
                        st.metric("RMSE", f"{metrics['rmse']:.2f}")
                    with col3:
                        st.metric("MAPE", f"{metrics['mape']:.2f}%")
                    with col4:
                        st.metric("R²", f"{metrics['r2']:.3f}")
                    
                    # Plot results
                    visualizer = ModelVisualizer()
                    fig = visualizer.plot_forecast_results(
                        train_data['datetime'] if 'datetime' in train_data.columns else None,
                        test_data['datetime'] if 'datetime' in test_data.columns else None,
                        train_data[target_column],
                        test_data[target_column],
                        forecast[:len(test_data)],
                        "SARIMA Model Results"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.error(f"❌ Error training SARIMA model: {str(e)}")
    
    if st.session_state.models:
        st.header("📊 Training Summary")
        
        # Create comparison table
        comparison_data = []
        for model_name, metrics in st.session_state.model_metrics.items():
            comparison_data.append({
                'Model': model_name,
                'MAE': f"{metrics['mae']:.2f}",
                'RMSE': f"{metrics['rmse']:.2f}",
                'MAPE': f"{metrics['mape']:.2f}%",
                'R²': f"{metrics['r2']:.3f}",
                'Status': '✅ Trained'
            })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Best model recommendation
            best_model = min(st.session_state.model_metrics.items(), key=lambda x: x[1]['rmse'])
            st.success(f"🏆 Best performing model: **{best_model[0]}** (RMSE: {best_model[1]['rmse']:.2f})")

if __name__ == "__main__":
    main()
