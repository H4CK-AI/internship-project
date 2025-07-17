# Bike Rental Demand Prediction System

## Overview

This is a comprehensive Streamlit-based web application for bike rental demand prediction using time series forecasting. The system provides an end-to-end solution for data exploration, model training, prediction generation, and performance comparison using statistical machine learning approaches including ARIMA and SARIMA models. The application has been optimized for deployment compatibility by removing TensorFlow dependencies.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit with multi-page architecture
- **Layout**: Wide layout with sidebar navigation
- **Styling**: Custom CSS with Plotly visualizations
- **State Management**: Session state for data persistence across pages

### Backend Architecture
- **Core Application**: Single-threaded Streamlit app (app.py)
- **Modular Design**: Utility modules for data processing, models, metrics, and visualization
- **Page Structure**: 
  - Main page: Data upload and overview
  - Data Exploration: EDA and preprocessing
  - Model Training: ML model development
  - Predictions: Forecasting interface
  - Model Comparison: Performance analysis

## Key Components

### 1. Data Processing Pipeline (`utils/data_preprocessing.py`)
- **Purpose**: Handles data cleaning, feature engineering, and preprocessing
- **Features**: 
  - Duplicate removal
  - Missing value handling strategies
  - Time-based feature extraction
  - Data validation and formatting

### 2. Model Framework (`utils/models.py`)
- **ARIMA Model**: Traditional statistical time series forecasting
- **SARIMA Model**: Seasonal ARIMA for seasonal patterns
- **LSTM Model**: Deep learning approach for complex patterns
- **Architecture**: Object-oriented design with consistent interfaces

### 3. Evaluation System (`utils/metrics.py`)
- **Purpose**: Comprehensive model performance evaluation
- **Metrics**: MAE, RMSE, MAPE, R², SMAPE, WAPE, MASE
- **Features**: Error handling, data validation, historical tracking

### 4. Visualization Engine (`utils/visualization.py`)
- **Framework**: Plotly for interactive charts
- **Components**: Time series plots, correlation heatmaps, distribution charts
- **Design**: Consistent color scheme and styling

## Data Flow

1. **Data Upload**: Users upload CSV files through the main page
2. **Data Storage**: Raw data stored in session state
3. **Preprocessing**: Data cleaned and features engineered in exploration page
4. **Model Training**: Processed data used to train multiple models
5. **Prediction**: Trained models generate forecasts
6. **Comparison**: Performance metrics calculated and visualized

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework
- **Pandas/NumPy**: Data manipulation and numerical computing
- **Plotly**: Interactive visualizations

### Machine Learning Stack
- **Statsmodels**: Statistical time series models (ARIMA, SARIMA)
- **pmdarima**: Automated ARIMA parameter selection
- **Scikit-learn**: Model evaluation and preprocessing

### Optional Dependencies
- **Seaborn/Matplotlib**: Additional visualization options
- Graceful fallback when advanced ML libraries unavailable

## Deployment Strategy

### Local Development
- **Environment**: Python virtual environment
- **Dependencies**: Requirements managed through pip
- **Configuration**: Streamlit config for page settings

### Production Considerations
- **Scalability**: Single-user application design
- **State Management**: Session-based data persistence
- **Resource Management**: Memory-efficient data handling
- **Error Handling**: Graceful degradation for missing dependencies

### Architecture Decisions

1. **Multi-page Structure**: Chosen for clear workflow separation and better user experience
2. **Session State**: Used for data persistence across pages without database dependency
3. **Modular Utilities**: Separates concerns and enables code reusability
4. **Optional Dependencies**: Allows graceful degradation when advanced ML libraries unavailable
5. **Plotly Visualization**: Provides interactive charts suitable for time series analysis
6. **Object-Oriented Models**: Consistent interface for different forecasting approaches

The system is designed to be user-friendly while providing professional-grade forecasting capabilities, with clear separation between data processing, model development, and visualization components.