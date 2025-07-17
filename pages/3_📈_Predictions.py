import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from utils.models import ARIMAModel, SARIMAModel, LSTMModel
from utils.visualization import ModelVisualizer

st.set_page_config(
    page_title="Predictions - Bike Rental Prediction",
    page_icon="📈",
    layout="wide"
)

def main():
    st.title("📈 Real-time Predictions and Forecasting")
    
    # Check if models are trained
    if not st.session_state.models:
        st.warning("⚠️ No trained models available. Please train models first.")
        st.stop()
    
    # Model selection
    st.header("🎯 Model Selection")
    
    available_models = list(st.session_state.models.keys())
    selected_model = st.selectbox("Select Model for Predictions", available_models)
    
    if selected_model in st.session_state.model_metrics:
        metrics = st.session_state.model_metrics[selected_model]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("MAE", f"{metrics['mae']:.2f}")
        with col2:
            st.metric("RMSE", f"{metrics['rmse']:.2f}")
        with col3:
            st.metric("MAPE", f"{metrics['mape']:.2f}%")
        with col4:
            st.metric("R²", f"{metrics['r2']:.3f}")
    
    # Prediction options
    st.header("⚙️ Prediction Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Forecast Settings")
        
        # Forecast horizon
        forecast_horizon = st.slider("Forecast Horizon (hours)", 1, 168, 24)  # Up to 1 week
        
        # Prediction interval
        prediction_interval = st.slider("Prediction Interval (%)", 80, 99, 95)
        
        # Start date for prediction
        if st.session_state.processed_data is not None:
            max_date = pd.to_datetime(st.session_state.processed_data['datetime']).max()
            start_date = st.date_input(
                "Start Date for Prediction",
                value=max_date.date() + timedelta(days=1),
                min_value=max_date.date()
            )
            start_hour = st.slider("Start Hour", 0, 23, 0)
        else:
            start_date = st.date_input("Start Date for Prediction", value=datetime.now().date())
            start_hour = st.slider("Start Hour", 0, 23, 0)
    
    with col2:
        st.subheader("External Factors")
        
        # Weather conditions (if applicable)
        weather_condition = st.selectbox(
            "Weather Condition",
            ["Clear", "Cloudy", "Light Rain", "Heavy Rain", "Snow"],
            index=0
        )
        
        # Temperature
        temperature = st.slider("Temperature (°C)", -10, 40, 20)
        
        # Holiday indicator
        is_holiday = st.checkbox("Holiday")
        
        # Working day indicator
        is_working_day = st.checkbox("Working Day", value=True)
        
        # Special event
        special_event = st.checkbox("Special Event")
    
    # Generate predictions
    if st.button("🔮 Generate Predictions"):
        try:
            with st.spinner(f"Generating predictions using {selected_model} model..."):
                
                # Create datetime range for predictions
                start_datetime = datetime.combine(start_date, datetime.min.time().replace(hour=start_hour))
                prediction_dates = pd.date_range(
                    start=start_datetime,
                    periods=forecast_horizon,
                    freq='h'
                )
                
                model_data = st.session_state.models[selected_model]
                
                if selected_model in ['ARIMA', 'SARIMA']:
                    # For ARIMA/SARIMA models
                    model = model_data['model']
                    
                    # Generate forecast
                    forecast_result = model.forecast(steps=forecast_horizon)
                    
                    if hasattr(forecast_result, 'predicted_mean'):
                        predictions = forecast_result.predicted_mean
                        if hasattr(forecast_result, 'conf_int'):
                            conf_int = forecast_result.conf_int()
                            lower_bound = conf_int.iloc[:, 0]
                            upper_bound = conf_int.iloc[:, 1]
                        else:
                            lower_bound = predictions * 0.9
                            upper_bound = predictions * 1.1
                    else:
                        predictions = forecast_result
                        lower_bound = predictions * 0.9
                        upper_bound = predictions * 1.1
                    
                elif selected_model == 'LSTM':
                    # For LSTM model
                    lstm_model = LSTMModel()
                    model = model_data['model']
                    
                    # Get recent data for prediction
                    recent_data = st.session_state.processed_data.tail(20)  # Last 20 hours
                    
                    # Generate predictions
                    predictions = lstm_model.predict(model, recent_data, forecast_horizon)
                    
                    # Create confidence intervals (approximated)
                    predictions_std = np.std(predictions) if len(predictions) > 1 else predictions[0] * 0.1
                    lower_bound = predictions - 1.96 * predictions_std
                    upper_bound = predictions + 1.96 * predictions_std
                
                # Create prediction DataFrame
                prediction_df = pd.DataFrame({
                    'datetime': prediction_dates,
                    'predicted_count': predictions,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'weather': weather_condition,
                    'temperature': temperature,
                    'is_holiday': is_holiday,
                    'is_working_day': is_working_day,
                    'special_event': special_event
                })
                
                # Store predictions
                st.session_state.predictions[selected_model] = prediction_df
                
                st.success(f"✅ Generated {forecast_horizon} hour predictions using {selected_model} model!")
                
        except Exception as e:
            st.error(f"❌ Error generating predictions: {str(e)}")
    
    # Display predictions
    if selected_model in st.session_state.predictions:
        st.header("📊 Prediction Results")
        
        prediction_df = st.session_state.predictions[selected_model]
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Average Predicted Count", f"{prediction_df['predicted_count'].mean():.1f}")
        with col2:
            st.metric("Peak Predicted Count", f"{prediction_df['predicted_count'].max():.1f}")
        with col3:
            st.metric("Minimum Predicted Count", f"{prediction_df['predicted_count'].min():.1f}")
        with col4:
            st.metric("Total Predicted Demand", f"{prediction_df['predicted_count'].sum():.0f}")
        
        # Visualization
        visualizer = ModelVisualizer()
        
        # Main prediction chart
        fig = go.Figure()
        
        # Add prediction line
        fig.add_trace(go.Scatter(
            x=prediction_df['datetime'],
            y=prediction_df['predicted_count'],
            mode='lines+markers',
            name='Predicted Count',
            line=dict(color='blue', width=3)
        ))
        
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=prediction_df['datetime'],
            y=prediction_df['upper_bound'],
            mode='lines',
            line=dict(color='lightblue', width=0),
            name='Upper Bound',
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=prediction_df['datetime'],
            y=prediction_df['lower_bound'],
            mode='lines',
            line=dict(color='lightblue', width=0),
            fill='tonexty',
            fillcolor='rgba(173, 216, 230, 0.3)',
            name=f'{prediction_interval}% Confidence Interval'
        ))
        
        fig.update_layout(
            title=f"{selected_model} Bike Rental Predictions",
            xaxis_title="Date and Time",
            yaxis_title="Predicted Bike Count",
            hovermode='x unified',
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Hourly pattern analysis
        st.subheader("📈 Hourly Pattern Analysis")
        
        prediction_df['hour'] = prediction_df['datetime'].dt.hour
        hourly_pattern = prediction_df.groupby('hour')['predicted_count'].mean().reset_index()
        
        hourly_fig = px.bar(
            hourly_pattern,
            x='hour',
            y='predicted_count',
            title='Average Predicted Demand by Hour',
            labels={'hour': 'Hour of Day', 'predicted_count': 'Average Predicted Count'}
        )
        
        st.plotly_chart(hourly_fig, use_container_width=True)
        
        # Daily pattern analysis
        if len(prediction_df) > 24:
            st.subheader("📅 Daily Pattern Analysis")
            
            prediction_df['date'] = prediction_df['datetime'].dt.date
            daily_pattern = prediction_df.groupby('date')['predicted_count'].sum().reset_index()
            
            daily_fig = px.line(
                daily_pattern,
                x='date',
                y='predicted_count',
                title='Total Predicted Demand by Day',
                labels={'date': 'Date', 'predicted_count': 'Total Predicted Count'}
            )
            
            st.plotly_chart(daily_fig, use_container_width=True)
        
        # Prediction table
        st.subheader("📋 Detailed Predictions")
        
        # Format datetime for display
        display_df = prediction_df.copy()
        display_df['datetime'] = display_df['datetime'].dt.strftime('%Y-%m-%d %H:%00')
        display_df['predicted_count'] = display_df['predicted_count'].round(1)
        display_df['lower_bound'] = display_df['lower_bound'].round(1)
        display_df['upper_bound'] = display_df['upper_bound'].round(1)
        
        st.dataframe(
            display_df[['datetime', 'predicted_count', 'lower_bound', 'upper_bound']].rename(columns={
                'datetime': 'Date & Time',
                'predicted_count': 'Predicted Count',
                'lower_bound': 'Lower Bound',
                'upper_bound': 'Upper Bound'
            }),
            use_container_width=True
        )
        
        # Export predictions
        st.header("💾 Export Predictions")
        
        csv_data = prediction_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Predictions CSV",
            data=csv_data,
            file_name=f"{selected_model}_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        # Business insights
        st.header("💡 Business Insights")
        
        # Peak hours
        peak_hours = prediction_df.nlargest(3, 'predicted_count')
        st.subheader("🔝 Peak Demand Hours")
        
        for idx, row in peak_hours.iterrows():
            st.write(f"**{row['datetime'].strftime('%Y-%m-%d %H:%00')}**: {row['predicted_count']:.1f} bikes")
        
        # Low demand hours
        low_hours = prediction_df.nsmallest(3, 'predicted_count')
        st.subheader("📉 Low Demand Hours")
        
        for idx, row in low_hours.iterrows():
            st.write(f"**{row['datetime'].strftime('%Y-%m-%d %H:%00')}**: {row['predicted_count']:.1f} bikes")
        
        # Recommendations
        st.subheader("🎯 Operational Recommendations")
        
        avg_demand = prediction_df['predicted_count'].mean()
        peak_demand = prediction_df['predicted_count'].max()
        
        if peak_demand > avg_demand * 1.5:
            st.warning(f"⚠️ High demand spike expected: {peak_demand:.1f} bikes (vs average {avg_demand:.1f})")
            st.write("**Recommendation**: Increase bike availability during peak hours")
        
        if prediction_df['predicted_count'].min() < avg_demand * 0.5:
            st.info("💡 **Recommendation**: Consider bike redistribution during low demand periods")
        
        # Weather impact
        if weather_condition != "Clear":
            st.write(f"🌦️ **Weather Impact**: {weather_condition} conditions may affect actual demand")

if __name__ == "__main__":
    main()
