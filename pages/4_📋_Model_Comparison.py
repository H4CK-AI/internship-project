import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

try:
    from utils.metrics import ModelEvaluator
    from utils.visualization import ModelVisualizer
    MODELS_AVAILABLE = True
except Exception as e:
    ModelEvaluator, ModelVisualizer = None, None
    MODELS_AVAILABLE = False

st.set_page_config(
    page_title="Model Comparison - Bike Rental Prediction",
    page_icon="📋",
    layout="wide"
)

def main():
    st.title("📋 Model Comparison and Performance Analysis")
    
    # Check if models are available
    if not MODELS_AVAILABLE:
        st.error("⚠️ Model comparison is currently unavailable due to library compatibility issues.")
        st.info("This is likely due to TensorFlow compatibility issues. Please try restarting the application.")
        return
    
    # Check if models are trained
    if not st.session_state.models or not st.session_state.model_metrics:
        st.warning("⚠️ No trained models available for comparison. Please train models first.")
        st.stop()
    
    # Model comparison overview
    st.header("📊 Model Performance Overview")
    
    # Create comparison DataFrame
    comparison_data = []
    for model_name, metrics in st.session_state.model_metrics.items():
        comparison_data.append({
            'Model': model_name,
            'MAE': metrics['mae'],
            'RMSE': metrics['rmse'],
            'MAPE': metrics['mape'],
            'R²': metrics['r2'],
            'Status': '✅ Trained'
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Display metrics table
    st.subheader("🎯 Performance Metrics Comparison")
    
    # Format the display DataFrame
    display_df = comparison_df.copy()
    display_df['MAE'] = display_df['MAE'].round(2)
    display_df['RMSE'] = display_df['RMSE'].round(2)
    display_df['MAPE'] = display_df['MAPE'].round(2)
    display_df['R²'] = display_df['R²'].round(3)
    
    st.dataframe(display_df, use_container_width=True)
    
    # Best model identification
    best_mae = comparison_df.loc[comparison_df['MAE'].idxmin()]
    best_rmse = comparison_df.loc[comparison_df['RMSE'].idxmin()]
    best_mape = comparison_df.loc[comparison_df['MAPE'].idxmin()]
    best_r2 = comparison_df.loc[comparison_df['R²'].idxmax()]
    
    st.subheader("🏆 Best Performing Models")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Best MAE",
            f"{best_mae['MAE']:.2f}",
            delta=f"{best_mae['Model']}"
        )
    
    with col2:
        st.metric(
            "Best RMSE",
            f"{best_rmse['RMSE']:.2f}",
            delta=f"{best_rmse['Model']}"
        )
    
    with col3:
        st.metric(
            "Best MAPE",
            f"{best_mape['MAPE']:.2f}%",
            delta=f"{best_mape['Model']}"
        )
    
    with col4:
        st.metric(
            "Best R²",
            f"{best_r2['R²']:.3f}",
            delta=f"{best_r2['Model']}"
        )
    
    # Overall best model
    # Weight the metrics (RMSE gets higher weight)
    comparison_df['weighted_score'] = (
        comparison_df['RMSE'] * 0.4 +
        comparison_df['MAE'] * 0.3 +
        comparison_df['MAPE'] * 0.2 +
        (1 - comparison_df['R²']) * 0.1  # Lower is better for R²
    )
    
    best_overall = comparison_df.loc[comparison_df['weighted_score'].idxmin()]
    
    st.success(f"🎉 **Overall Best Model**: {best_overall['Model']} (Weighted Score: {best_overall['weighted_score']:.2f})")
    
    # Visualization section
    st.header("📈 Performance Visualization")
    
    # Metrics comparison charts
    col1, col2 = st.columns(2)
    
    with col1:
        # MAE and RMSE comparison
        fig_error = go.Figure()
        
        fig_error.add_trace(go.Bar(
            name='MAE',
            x=comparison_df['Model'],
            y=comparison_df['MAE'],
            yaxis='y1',
            marker_color='lightblue'
        ))
        
        fig_error.add_trace(go.Bar(
            name='RMSE',
            x=comparison_df['Model'],
            y=comparison_df['RMSE'],
            yaxis='y1',
            marker_color='darkblue'
        ))
        
        fig_error.update_layout(
            title="MAE and RMSE Comparison",
            xaxis_title="Model",
            yaxis_title="Error Value",
            barmode='group'
        )
        
        st.plotly_chart(fig_error, use_container_width=True)
    
    with col2:
        # MAPE and R² comparison
        fig_metrics = make_subplots(
            rows=2, cols=1,
            subplot_titles=('MAPE (%)', 'R² Score'),
            vertical_spacing=0.1
        )
        
        # MAPE
        fig_metrics.add_trace(
            go.Bar(
                x=comparison_df['Model'],
                y=comparison_df['MAPE'],
                name='MAPE',
                marker_color='orange'
            ),
            row=1, col=1
        )
        
        # R²
        fig_metrics.add_trace(
            go.Bar(
                x=comparison_df['Model'],
                y=comparison_df['R²'],
                name='R²',
                marker_color='green'
            ),
            row=2, col=1
        )
        
        fig_metrics.update_layout(
            title="MAPE and R² Comparison",
            showlegend=False,
            height=600
        )
        
        st.plotly_chart(fig_metrics, use_container_width=True)
    
    # Radar chart for comprehensive comparison
    st.subheader("🎯 Comprehensive Performance Radar")
    
    if len(comparison_df) > 1:
        # Normalize metrics for radar chart (0-1 scale)
        normalized_df = comparison_df.copy()
        
        # For MAE, RMSE, MAPE: lower is better, so we invert them
        normalized_df['MAE_norm'] = 1 - (normalized_df['MAE'] - normalized_df['MAE'].min()) / (normalized_df['MAE'].max() - normalized_df['MAE'].min())
        normalized_df['RMSE_norm'] = 1 - (normalized_df['RMSE'] - normalized_df['RMSE'].min()) / (normalized_df['RMSE'].max() - normalized_df['RMSE'].min())
        normalized_df['MAPE_norm'] = 1 - (normalized_df['MAPE'] - normalized_df['MAPE'].min()) / (normalized_df['MAPE'].max() - normalized_df['MAPE'].min())
        
        # For R²: higher is better, so we keep it as is
        normalized_df['R2_norm'] = (normalized_df['R²'] - normalized_df['R²'].min()) / (normalized_df['R²'].max() - normalized_df['R²'].min())
        
        # Create radar chart
        fig_radar = go.Figure()
        
        colors = ['blue', 'red', 'green', 'purple', 'orange']
        
        for i, (idx, row) in enumerate(normalized_df.iterrows()):
            fig_radar.add_trace(go.Scatterpolar(
                r=[row['MAE_norm'], row['RMSE_norm'], row['MAPE_norm'], row['R2_norm']],
                theta=['MAE', 'RMSE', 'MAPE', 'R²'],
                fill='toself',
                name=row['Model'],
                line_color=colors[i % len(colors)]
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title="Model Performance Radar Chart (Normalized)"
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
    
    # Prediction comparison
    st.header("🔄 Prediction Comparison")
    
    # Select models to compare
    models_to_compare = st.multiselect(
        "Select models to compare predictions",
        list(st.session_state.models.keys()),
        default=list(st.session_state.models.keys())
    )
    
    if len(models_to_compare) >= 2:
        # Create combined prediction plot
        visualizer = ModelVisualizer()
        
        fig_combined = go.Figure()
        
        colors = ['blue', 'red', 'green', 'purple', 'orange']
        
        for i, model_name in enumerate(models_to_compare):
            model_data = st.session_state.models[model_name]
            
            if model_name in ['ARIMA', 'SARIMA']:
                # For ARIMA/SARIMA
                test_data = model_data['test_data']
                forecast = model_data['forecast'][:len(test_data)]
                
                # Create x-axis (assuming hourly data)
                x_data = list(range(len(test_data)))
                
                fig_combined.add_trace(go.Scatter(
                    x=x_data,
                    y=forecast,
                    mode='lines',
                    name=f'{model_name} Predictions',
                    line=dict(color=colors[i % len(colors)], width=2)
                ))
            
            elif model_name == 'LSTM':
                # For LSTM
                predictions = model_data['predictions']
                
                # Create x-axis
                x_data = list(range(len(predictions)))
                
                fig_combined.add_trace(go.Scatter(
                    x=x_data,
                    y=predictions,
                    mode='lines',
                    name=f'{model_name} Predictions',
                    line=dict(color=colors[i % len(colors)], width=2)
                ))
        
        # Add actual values
        if models_to_compare:
            first_model = models_to_compare[0]
            if first_model in st.session_state.models:
                test_data = st.session_state.models[first_model]['test_data']
                if first_model == 'LSTM':
                    test_data = test_data.values[-len(st.session_state.models[first_model]['predictions']):]
                
                x_data = list(range(len(test_data)))
                
                fig_combined.add_trace(go.Scatter(
                    x=x_data,
                    y=test_data,
                    mode='lines',
                    name='Actual Values',
                    line=dict(color='black', width=3, dash='dash')
                ))
        
        fig_combined.update_layout(
            title="Model Predictions Comparison",
            xaxis_title="Time Index",
            yaxis_title="Bike Count",
            hovermode='x unified',
            showlegend=True
        )
        
        st.plotly_chart(fig_combined, use_container_width=True)
    
    # Error analysis
    st.header("🔍 Error Analysis")
    
    if len(models_to_compare) >= 1:
        selected_model_error = st.selectbox(
            "Select model for detailed error analysis",
            models_to_compare
        )
        
        if selected_model_error in st.session_state.models:
            model_data = st.session_state.models[selected_model_error]
            
            # Calculate residuals
            if selected_model_error in ['ARIMA', 'SARIMA']:
                actual = model_data['test_data'].values
                predicted = model_data['forecast'][:len(actual)]
            else:  # LSTM
                actual = model_data['test_data'].values[-len(model_data['predictions']):]
                predicted = model_data['predictions']
            
            residuals = actual - predicted
            
            # Error distribution
            col1, col2 = st.columns(2)
            
            with col1:
                # Residuals plot
                fig_residuals = px.histogram(
                    x=residuals,
                    nbins=30,
                    title=f"{selected_model_error} - Residuals Distribution"
                )
                st.plotly_chart(fig_residuals, use_container_width=True)
            
            with col2:
                # Actual vs Predicted scatter plot
                fig_scatter = px.scatter(
                    x=actual,
                    y=predicted,
                    title=f"{selected_model_error} - Actual vs Predicted",
                    labels={'x': 'Actual', 'y': 'Predicted'}
                )
                
                # Add perfect prediction line
                min_val = min(actual.min(), predicted.min())
                max_val = max(actual.max(), predicted.max())
                fig_scatter.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color='red', dash='dash')
                ))
                
                st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Model recommendations
    st.header("💡 Model Recommendations")
    
    st.subheader("🎯 When to Use Each Model")
    
    for model_name in st.session_state.models.keys():
        metrics = st.session_state.model_metrics[model_name]
        
        with st.expander(f"{model_name} Model Analysis"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Performance Metrics:**")
                st.write(f"- MAE: {metrics['mae']:.2f}")
                st.write(f"- RMSE: {metrics['rmse']:.2f}")
                st.write(f"- MAPE: {metrics['mape']:.2f}%")
                st.write(f"- R²: {metrics['r2']:.3f}")
            
            with col2:
                st.write("**Recommendations:**")
                
                if model_name == 'ARIMA':
                    st.write("✅ **Best for**: Simple, linear trends")
                    st.write("✅ **Pros**: Fast training, interpretable")
                    st.write("❌ **Cons**: Limited with complex patterns")
                
                elif model_name == 'SARIMA':
                    st.write("✅ **Best for**: Seasonal patterns")
                    st.write("✅ **Pros**: Handles seasonality well")
                    st.write("❌ **Cons**: Requires parameter tuning")
                
                elif model_name == 'LSTM':
                    st.write("✅ **Best for**: Complex, non-linear patterns")
                    st.write("✅ **Pros**: Captures long-term dependencies")
                    st.write("❌ **Cons**: Requires more data, longer training")
    
    # Final recommendation
    st.subheader("🏆 Final Recommendation")
    
    if metrics['r2'] > 0.8:
        confidence = "High"
        color = "success"
    elif metrics['r2'] > 0.6:
        confidence = "Medium"
        color = "warning"
    else:
        confidence = "Low"
        color = "error"
    
    st.write(f"**Recommended Model**: {best_overall['Model']}")
    st.write(f"**Confidence Level**: {confidence}")
    st.write(f"**Expected Performance**: RMSE ≈ {best_overall['RMSE']:.2f}")
    
    if confidence == "High":
        st.success("🎉 The model shows excellent performance and is ready for production use!")
    elif confidence == "Medium":
        st.warning("⚠️ The model shows good performance but may need further tuning or more data.")
    else:
        st.error("❌ The model needs significant improvement before production use.")

if __name__ == "__main__":
    main()
