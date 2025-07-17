import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Statistical models
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.stats.diagnostic import acorr_ljungbox
    STATSMODELS_AVAILABLE = True
except ImportError:
    ARIMA = None
    SARIMAX = None
    STATSMODELS_AVAILABLE = False

# Try to import pmdarima but handle version conflicts
try:
    import pmdarima as pm
    PMDARIMA_AVAILABLE = True
except (ImportError, ValueError) as e:
    pm = None
    PMDARIMA_AVAILABLE = False

class ARIMAModel:
    def __init__(self):
        self.model = None
        self.fitted_model = None
        self.order = None
    
    def fit(self, data, order=(1, 1, 1), test_steps=24):
        """
        Fit ARIMA model
        
        Args:
            data (pd.Series): Time series data
            order (tuple): ARIMA order (p, d, q)
            test_steps (int): Number of steps to forecast
        
        Returns:
            tuple: (fitted_model, fitted_values, forecast)
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for ARIMA modeling")
        
        try:
            self.order = order
            self.model = ARIMA(data, order=order)
            self.fitted_model = self.model.fit()
            
            # Get fitted values
            fitted_values = self.fitted_model.fittedvalues
            
            # Generate forecast
            forecast = self.fitted_model.forecast(steps=test_steps)
            
            return self.fitted_model, fitted_values, forecast
            
        except Exception as e:
            raise ValueError(f"Error fitting ARIMA model: {str(e)}")
    
    def predict(self, steps=1):
        """
        Make predictions using fitted model
        
        Args:
            steps (int): Number of steps to predict
        
        Returns:
            np.array: Predictions
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            return self.fitted_model.forecast(steps=steps)
        except Exception as e:
            raise ValueError(f"Error making predictions: {str(e)}")

class SARIMAModel:
    def __init__(self):
        self.model = None
        self.fitted_model = None
        self.order = None
        self.seasonal_order = None
    
    def fit(self, data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), test_steps=24):
        """
        Fit SARIMA model
        
        Args:
            data (pd.Series): Time series data
            order (tuple): ARIMA order (p, d, q)
            seasonal_order (tuple): Seasonal order (P, D, Q, s)
            test_steps (int): Number of steps to forecast
        
        Returns:
            tuple: (fitted_model, fitted_values, forecast)
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for SARIMA modeling")
        
        try:
            self.order = order
            self.seasonal_order = seasonal_order
            self.model = SARIMAX(data, order=order, seasonal_order=seasonal_order)
            self.fitted_model = self.model.fit(disp=False)
            
            # Get fitted values
            fitted_values = self.fitted_model.fittedvalues
            
            # Generate forecast
            forecast = self.fitted_model.forecast(steps=test_steps)
            
            return self.fitted_model, fitted_values, forecast
            
        except Exception as e:
            raise ValueError(f"Error fitting SARIMA model: {str(e)}")
    
    def fit_auto(self, data, seasonal_period=12, test_steps=24):
        """
        Automatically fit SARIMA model using auto_arima
        
        Args:
            data (pd.Series): Time series data
            seasonal_period (int): Seasonal period
            test_steps (int): Number of steps to forecast
        
        Returns:
            tuple: (fitted_model, fitted_values, forecast)
        """
        if not PMDARIMA_AVAILABLE:
            # Fallback to basic SARIMA with default parameters
            return self.fit(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, seasonal_period), test_steps=test_steps)
        
        try:
            # Use auto_arima with seasonal components
            auto_model = pm.auto_arima(
                data,
                start_p=0, start_q=0,
                max_p=2, max_q=2,
                start_P=0, start_Q=0,
                max_P=2, max_Q=2,
                seasonal=True,
                m=seasonal_period,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore'
            )
            
            self.fitted_model = auto_model
            self.order = auto_model.order
            self.seasonal_order = auto_model.seasonal_order
            
            # Get fitted values
            fitted_values = auto_model.fittedvalues()
            
            # Generate forecast
            forecast = auto_model.predict(n_periods=test_steps)
            
            return self.fitted_model, fitted_values, forecast
            
        except Exception as e:
            # Fallback to basic SARIMA if auto_arima fails
            return self.fit(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, seasonal_period), test_steps=test_steps)
    
    def predict(self, steps=1):
        """
        Make predictions using fitted model
        
        Args:
            steps (int): Number of steps to predict
        
        Returns:
            np.array: Predictions
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            if hasattr(self.fitted_model, 'predict'):
                return self.fitted_model.predict(n_periods=steps)
            else:
                return self.fitted_model.forecast(steps=steps)
        except Exception as e:
            raise ValueError(f"Error making predictions: {str(e)}")

class LSTMModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.lookback_window = None
        self.feature_columns = None
    
    def fit(self, data, target_column, lookback_window=20, lstm_units=64, 
            num_layers=2, dropout_rate=0.2, epochs=50, batch_size=32,
            learning_rate=0.001, validation_split=0.2, test_data=None):
        """
        LSTM model is not available in this simplified version
        """
        raise ImportError("LSTM model is not available due to TensorFlow compatibility issues. Please use ARIMA or SARIMA models instead.")
    
    def predict(self, model, data, target_column=None, steps=None):
        """
        LSTM prediction is not available in this simplified version
        """
        raise ImportError("LSTM prediction is not available due to TensorFlow compatibility issues. Please use ARIMA or SARIMA models instead.")