import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from utils.data_preprocessing import DataPreprocessor, ensure_arrow_compatibility
from utils.visualization import DataVisualizer

st.set_page_config(
    page_title="Data Exploration - Bike Rental Prediction",
    page_icon="📊",
    layout="wide"
)

def main():
    st.title("📊 Data Exploration and Analysis")
    
    # Check if data is loaded
    if st.session_state.data is None:
        st.warning("⚠️ No data loaded. Please upload a dataset on the main page first.")
        st.stop()
    
    data = st.session_state.data.copy()
    
    # Data preprocessing section
    st.header("🔧 Data Preprocessing")
    
    preprocessor = DataPreprocessor()
    visualizer = DataVisualizer()
    
    # Data cleaning options
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Cleaning Options")
        remove_duplicates = st.checkbox("Remove duplicate records", value=True)
        handle_missing = st.selectbox(
            "Handle missing values",
            ["Keep as is", "Forward fill", "Backward fill", "Interpolate", "Drop rows"]
        )
        
    with col2:
        st.subheader("Feature Engineering")
        extract_time_features = st.checkbox("Extract time-based features", value=True)
        detect_outliers = st.checkbox("Detect and highlight outliers", value=True)
    
    if st.button("🔄 Apply Preprocessing"):
        try:
            # Apply preprocessing
            processed_data = preprocessor.preprocess_data(
                data,
                remove_duplicates=remove_duplicates,
                missing_strategy=handle_missing.lower().replace(" ", "_"),
                extract_time_features=extract_time_features
            )
            
            st.session_state.processed_data = processed_data
            st.success("✅ Data preprocessing completed successfully!")
            
            # Show preprocessing summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original Records", len(data))
            with col2:
                st.metric("Processed Records", len(processed_data))
            with col3:
                st.metric("New Features", len(processed_data.columns) - len(data.columns))
                
        except Exception as e:
            st.error(f"❌ Error during preprocessing: {str(e)}")
    
    # Use processed data if available, otherwise use original
    working_data = st.session_state.processed_data if st.session_state.processed_data is not None else data
    
    # Ensure Arrow compatibility
    working_data = ensure_arrow_compatibility(working_data)
    
    # Display available columns for debugging
    st.expander("🔍 Available Columns").write(f"Columns in dataset: {list(working_data.columns)}")
    
    # Data overview
    st.header("📋 Data Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Summary")
        st.write(f"**Shape:** {working_data.shape}")
        
        # Check if datetime column exists
        if 'datetime' in working_data.columns:
            # Convert datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(working_data['datetime']):
                working_data['datetime'] = pd.to_datetime(working_data['datetime'])
            st.write(f"**Date Range:** {working_data['datetime'].min()} to {working_data['datetime'].max()}")
        else:
            st.write("**Date Range:** Not available (no datetime column)")
            
        st.write(f"**Total Records:** {len(working_data):,}")
        
        # Missing values
        missing_data = working_data.isnull().sum()
        if missing_data.sum() > 0:
            st.write("**Missing Values:**")
            st.write(missing_data[missing_data > 0])
        else:
            st.write("**Missing Values:** None")
    
    with col2:
        st.subheader("Statistical Summary")
        numeric_columns = working_data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            st.write(working_data[numeric_columns].describe())
    
    # Data visualization
    st.header("📈 Data Visualization")
    
    # Time series plot
    st.subheader("Time Series Analysis")
    
    # Try to find datetime column
    datetime_col = None
    for col in working_data.columns:
        if 'datetime' in col.lower() or 'date' in col.lower() or 'time' in col.lower():
            datetime_col = col
            break
    
    # Try to find target column
    target_col = None
    for col in working_data.columns:
        if col.lower() in ['count', 'cnt', 'target', 'y', 'demand', 'rental', 'bike']:
            target_col = col
            break
    
    # If still not found, use first numeric column
    if target_col is None:
        numeric_cols = working_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            target_col = numeric_cols[0]
    
    if datetime_col and target_col:
        # Convert datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(working_data[datetime_col]):
            working_data[datetime_col] = pd.to_datetime(working_data[datetime_col])
        
        # Main time series plot
        fig = visualizer.create_time_series_plot(working_data, datetime_col, target_col)
        st.plotly_chart(fig, use_container_width=True)
        
        # Seasonal decomposition
        st.subheader("Seasonal Decomposition")
        decomp_fig = visualizer.create_seasonal_decomposition(working_data, datetime_col, target_col)
        st.plotly_chart(decomp_fig, use_container_width=True)
        
        # Distribution analysis
        st.subheader("Distribution Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram
            hist_fig = px.histogram(
                working_data, 
                x=target_col, 
                nbins=50,
                title=f'Distribution of {target_col}',
                labels={target_col: target_col.title(), f'{target_col}_count': 'Frequency'}
            )
            st.plotly_chart(hist_fig, use_container_width=True)
        
        with col2:
            # Box plot
            box_fig = px.box(
                working_data, 
                y=target_col,
                title=f'Box Plot of {target_col}'
            )
            st.plotly_chart(box_fig, use_container_width=True)
        
        # Correlation analysis
        st.subheader("Correlation Analysis")
        numeric_data = working_data.select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) > 1:
            corr_fig = visualizer.create_correlation_heatmap(numeric_data)
            st.plotly_chart(corr_fig, use_container_width=True)
        
        # Hourly and daily patterns
        if extract_time_features and 'hour' in working_data.columns:
            st.subheader("Temporal Patterns")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Hourly pattern
                hourly_avg = working_data.groupby('hour')[target_col].mean().reset_index()
                hourly_fig = px.bar(
                    hourly_avg, 
                    x='hour', 
                    y=target_col,
                    title=f'Average {target_col} by Hour'
                )
                st.plotly_chart(hourly_fig, use_container_width=True)
            
            with col2:
                # Daily pattern
                if 'day_of_week' in working_data.columns:
                    daily_avg = working_data.groupby('day_of_week')[target_col].mean().reset_index()
                    daily_fig = px.bar(
                        daily_avg, 
                        x='day_of_week', 
                        y=target_col,
                        title=f'Average {target_col} by Day of Week'
                    )
                    st.plotly_chart(daily_fig, use_container_width=True)
        
        # Weather impact analysis
        if 'weather' in working_data.columns:
            st.subheader("Weather Impact Analysis")
            weather_fig = px.box(
                working_data, 
                x='weather', 
                y=target_col,
                title=f'{target_col} by Weather Condition'
            )
            st.plotly_chart(weather_fig, use_container_width=True)
        
        # Outlier detection
        if detect_outliers:
            st.subheader("Outlier Detection")
            outliers = preprocessor.detect_outliers(working_data, target_col)
            
            if len(outliers) > 0:
                st.warning(f"⚠️ Detected {len(outliers)} potential outliers")
                
                # Scatter plot with outliers highlighted
                working_data['is_outlier'] = working_data.index.isin(outliers)
                scatter_fig = px.scatter(
                    working_data, 
                    x=datetime_col, 
                    y=target_col,
                    color='is_outlier',
                    title='Time Series with Outliers Highlighted',
                    color_discrete_map={True: 'red', False: 'blue'}
                )
                st.plotly_chart(scatter_fig, use_container_width=True)
                
                # Show outlier details
                st.write("**Outlier Details:**")
                st.dataframe(working_data[working_data['is_outlier']][[datetime_col, target_col]])
            else:
                st.success("✅ No outliers detected")
    
    else:
        st.error("❌ Could not find appropriate datetime and target columns")
        st.info(f"Available columns: {list(working_data.columns)}")
        st.info("Please ensure your dataset has datetime and numeric target columns, or apply preprocessing to create them.")
    
    # Data export
    st.header("💾 Export Processed Data")
    
    if st.session_state.processed_data is not None:
        csv_data = st.session_state.processed_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Processed Data",
            data=csv_data,
            file_name=f"processed_bike_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
