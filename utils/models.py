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
    # Handle numpy dtype compatibility issues
    pm = None
    PMDARIMA_AVAILABLE = False

# Deep learning models
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.preprocessing import MinMaxScaler
    TENSORFLOW_AVAILABLE = True
except (ImportError, TypeError, Exception) as e:
    # Handle TensorFlow import errors including dtype conversion issues
    tf = None
    Sequential = None
    LSTM = None
    Dense = None
    Dropout = None
    Adam = None
    EarlyStopping = None
    MinMaxScaler = None
    TENSORFLOW_AVAILABLE = False

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
    
    def fit_auto(self, data, test_steps=24, seasonal=False, stepwise=True):
        """
        Automatically fit ARIMA model using auto_arima
        
        Args:
            data (pd.Series): Time series data
            test_steps (int): Number of steps to forecast
            seasonal (bool): Whether to consider seasonal components
            stepwise (bool): Whether to use stepwise selection
        
        Returns:
            tuple: (fitted_model, fitted_values, forecast)
        """
        if not PMDARIMA_AVAILABLE:
            # Fallback to basic ARIMA with default parameters
            return self.fit(data, order=(1, 1, 1), test_steps=test_steps)
        
        try:
            # Use auto_arima to find best parameters
            auto_model = pm.auto_arima(
                data,
                start_p=0, start_q=0,
                max_p=3, max_q=3,
                seasonal=seasonal,
                stepwise=stepwise,
                suppress_warnings=True,
                error_action='ignore'
            )
            
            self.fitted_model = auto_model
            self.order = auto_model.order
            
            # Get fitted values
            fitted_values = auto_model.fittedvalues()
            
            # Generate forecast
            forecast = auto_model.predict(n_periods=test_steps)
            
            return self.fitted_model, fitted_values, forecast
            
        except Exception as e:
            # Fallback to basic ARIMA if auto_arima fails
            return self.fit(data, order=(1, 1, 1), test_steps=test_steps)
    
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
            
            self.model = SARIMAX(
                data,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
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
    
    def _prepare_data(self, data, target_column, lookback_window=20):
        """
        Prepare data for LSTM training
        
        Args:
            data (pd.DataFrame): Input data
            target_column (str): Target column name
            lookback_window (int): Number of previous time steps to use
        
        Returns:
            tuple: (X, y, scaler)
        """
        # Select numeric columns
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        feature_data = data[numeric_columns].copy()
        
        # Scale the data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(feature_data)
        
        # Create sequences
        X, y = [], []
        target_idx = list(numeric_columns).index(target_column)
        
        for i in range(lookback_window, len(scaled_data)):
            X.append(scaled_data[i-lookback_window:i])
            y.append(scaled_data[i, target_idx])
        
        return np.array(X), np.array(y), scaler
    
    def fit(self, data, target_column, lookback_window=20, lstm_units=64, 
            num_layers=2, dropout_rate=0.2, epochs=50, batch_size=32,
            learning_rate=0.001, validation_split=0.2, test_data=None):
        """
        Fit LSTM model
        
        Args:
            data (pd.DataFrame): Training data
            target_column (str): Target column name
            lookback_window (int): Number of previous time steps to use
            lstm_units (int): Number of LSTM units
            num_layers (int): Number of LSTM layers
            dropout_rate (float): Dropout rate
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            learning_rate (float): Learning rate
            validation_split (float): Validation split ratio
            test_data (pd.DataFrame): Test data for evaluation
        
        Returns:
            tuple: (model, history, predictions)
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM modeling but is not available")
        
        try:
            self.lookback_window = lookback_window
            
            # Prepare training data
            X_train, y_train, self.scaler = self._prepare_data(
                data, target_column, lookback_window
            )
            
            # Build LSTM model
            self.model = Sequential()
            
            # First LSTM layer
            if num_layers > 1:
                self.model.add(LSTM(
                    lstm_units,
                    return_sequences=True,
                    input_shape=(lookback_window, X_train.shape[2])
                ))
            else:
                self.model.add(LSTM(
                    lstm_units,
                    input_shape=(lookback_window, X_train.shape[2])
                ))
            
            self.model.add(Dropout(dropout_rate))
            
            # Additional LSTM layers
            for i in range(1, num_layers):
                return_sequences = i < num_layers - 1
                self.model.add(LSTM(lstm_units, return_sequences=return_sequences))
                self.model.add(Dropout(dropout_rate))
            
            # Output layer
            self.model.add(Dense(1))
            
            # Compile model
            optimizer = Adam(learning_rate=learning_rate)
            self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
            
            # Early stopping
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Make predictions on test data if provided
            predictions = None
            if test_data is not None:
                predictions = self.predict(self.model, test_data, target_column)
            
            return self.model, history, predictions
            
        except Exception as e:
            raise ValueError(f"Error fitting LSTM model: {str(e)}")
    
    def predict(self, model, data, target_column=None, steps=None):
        """
        Make predictions using trained LSTM model
        
        Args:
            model: Trained LSTM model
            data (pd.DataFrame): Data for prediction
            target_column (str): Target column name
            steps (int): Number of future steps to predict
        
        Returns:
            np.array: Predictions
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM prediction but is not available")
        
        if model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        if self.scaler is None:
            raise ValueError("Scaler not available. Model must be fitted first.")
        
        try:
            # If steps is specified, generate future predictions
            if steps is not None:
                # Use the last lookback_window points for prediction
                numeric_columns = data.select_dtypes(include=[np.number]).columns
                feature_data = data[numeric_columns].tail(self.lookback_window)
                
                # Scale the data
                scaled_data = self.scaler.transform(feature_data)
                
                predictions = []
                current_batch = scaled_data.copy()
                
                for i in range(steps):
                    # Reshape for prediction
                    current_batch_reshaped = current_batch.reshape(1, self.lookback_window, -1)
                    
                    # Make prediction
                    pred = model.predict(current_batch_reshaped, verbose=0)[0, 0]
                    predictions.append(pred)
                    
                    # Update batch for next prediction
                    # Create new row with prediction
                    new_row = current_batch[-1].copy()
                    if target_column and target_column in numeric_columns:
                        target_idx = list(numeric_columns).index(target_column)
                        new_row[target_idx] = pred
                    
                    # Roll the batch
                    current_batch = np.roll(current_batch, -1, axis=0)
                    current_batch[-1] = new_row
                
                # Denormalize predictions
                if target_column and target_column in numeric_columns:
                    target_idx = list(numeric_columns).index(target_column)
                    
                    # Create dummy array for inverse transform
                    dummy_array = np.zeros((len(predictions), len(numeric_columns)))
                    dummy_array[:, target_idx] = predictions
                    
                    # Inverse transform
                    denormalized = self.scaler.inverse_transform(dummy_array)
                    return denormalized[:, target_idx]
                else:
                    return np.array(predictions)
            
            else:
                # Regular prediction on provided data
                numeric_columns = data.select_dtypes(include=[np.number]).columns
                feature_data = data[numeric_columns]
                
                # Scale the data
                scaled_data = self.scaler.transform(feature_data)
                
                # Create sequences
                X = []
                for i in range(self.lookback_window, len(scaled_data)):
                    X.append(scaled_data[i-self.lookback_window:i])
                
                if len(X) == 0:
                    raise ValueError("Not enough data for prediction")
                
                X = np.array(X)
                
                # Make predictions
                predictions = model.predict(X, verbose=0)
                
                # Denormalize predictions
                if target_column and target_column in numeric_columns:
                    target_idx = list(numeric_columns).index(target_column)
                    
                    # Create dummy array for inverse transform
                    dummy_array = np.zeros((len(predictions), len(numeric_columns)))
                    dummy_array[:, target_idx] = predictions.flatten()
                    
                    # Inverse transform
                    denormalized = self.scaler.inverse_transform(dummy_array)
                    return denormalized[:, target_idx]
                else:
                    return predictions.flatten()
                    
        except Exception as e:
            raise ValueError(f"Error making predictions: {str(e)}")
    
    def evaluate(self, data, target_column):
        """
        Evaluate model performance
        
        Args:
            data (pd.DataFrame): Test data
            target_column (str): Target column name
        
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be fitted before evaluation")
        
        try:
            # Prepare test data
            X_test, y_test, _ = self._prepare_data(data, target_column, self.lookback_window)
            
            # Make predictions
            predictions = self.model.predict(X_test, verbose=0)
            
            # Calculate metrics
            mse = np.mean((y_test - predictions.flatten())**2)
            mae = np.mean(np.abs(y_test - predictions.flatten()))
            rmse = np.sqrt(mse)
            
            return {
                'mse': mse,
                'mae': mae,
                'rmse': rmse
            }
            
        except Exception as e:
            raise ValueError(f"Error evaluating model: {str(e)}")
