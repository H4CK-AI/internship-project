import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self):
        self.scaler = None
        self.feature_columns = []
    
    def preprocess_data(self, data, remove_duplicates=True, missing_strategy='forward_fill', 
                       extract_time_features=True):
        """
        Comprehensive data preprocessing pipeline
        
        Args:
            data (pd.DataFrame): Raw data
            remove_duplicates (bool): Whether to remove duplicate records
            missing_strategy (str): Strategy for handling missing values
            extract_time_features (bool): Whether to extract time-based features
        
        Returns:
            pd.DataFrame: Preprocessed data
        """
        processed_data = data.copy()
        
        # Ensure datetime column exists and is properly formatted
        if 'datetime' in processed_data.columns:
            processed_data['datetime'] = pd.to_datetime(processed_data['datetime'])
        elif 'date' in processed_data.columns and 'time' in processed_data.columns:
            processed_data['datetime'] = pd.to_datetime(
                processed_data['date'].astype(str) + ' ' + processed_data['time'].astype(str)
            )
        else:
            # Create a datetime column if none exists
            processed_data['datetime'] = pd.date_range(
                start='2023-01-01', periods=len(processed_data), freq='H'
            )
        
        # Sort by datetime
        processed_data = processed_data.sort_values('datetime').reset_index(drop=True)
        
        # Remove duplicates
        if remove_duplicates:
            processed_data = processed_data.drop_duplicates(subset=['datetime'], keep='first')
        
        # Handle missing values
        processed_data = self._handle_missing_values(processed_data, missing_strategy)
        
        # Extract time-based features
        if extract_time_features:
            processed_data = self._extract_time_features(processed_data)
        
        # Clean and validate data
        processed_data = self._clean_data(processed_data)
        
        return processed_data
    
    def _handle_missing_values(self, data, strategy):
        """Handle missing values based on strategy"""
        if strategy == 'forward_fill':
            return data.fillna(method='ffill')
        elif strategy == 'backward_fill':
            return data.fillna(method='bfill')
        elif strategy == 'interpolate':
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                data[col] = data[col].interpolate()
            return data.fillna(method='ffill')  # Handle any remaining missing values
        elif strategy == 'drop_rows':
            return data.dropna()
        else:
            return data
    
    def _extract_time_features(self, data):
        """Extract time-based features from datetime column"""
        if 'datetime' in data.columns:
            data['year'] = data['datetime'].dt.year
            data['month'] = data['datetime'].dt.month
            data['day'] = data['datetime'].dt.day
            data['hour'] = data['datetime'].dt.hour
            data['day_of_week'] = data['datetime'].dt.dayofweek
            data['day_of_year'] = data['datetime'].dt.dayofyear
            data['week_of_year'] = data['datetime'].dt.isocalendar().week
            data['quarter'] = data['datetime'].dt.quarter
            
            # Is weekend
            data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
            
            # Season
            data['season'] = data['month'].apply(self._get_season)
            
            # Time of day
            data['time_of_day'] = data['hour'].apply(self._get_time_of_day)
            
            # Cyclical features
            data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
            data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
            data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
            data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
            data['day_of_week_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
            data['day_of_week_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
        
        return data
    
    def _get_season(self, month):
        """Get season from month"""
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    def _get_time_of_day(self, hour):
        """Get time of day from hour"""
        if 6 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 18:
            return 'Afternoon'
        elif 18 <= hour < 22:
            return 'Evening'
        else:
            return 'Night'
    
    def _clean_data(self, data):
        """Clean and validate data"""
        # Remove any infinite values
        data = data.replace([np.inf, -np.inf], np.nan)
        
        # Handle remaining missing values
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        # Ensure numeric columns are properly typed
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Remove any rows with all NaN values
        data = data.dropna(how='all')
        
        return data
    
    def detect_outliers(self, data, column, method='iqr', threshold=1.5):
        """
        Detect outliers in a column using IQR method
        
        Args:
            data (pd.DataFrame): Data
            column (str): Column to check for outliers
            method (str): Method for outlier detection ('iqr' or 'zscore')
            threshold (float): Threshold for outlier detection
        
        Returns:
            list: Indices of outliers
        """
        if column not in data.columns:
            return []
        
        if method == 'iqr':
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)].index.tolist()
        
        elif method == 'zscore':
            z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
            outliers = data[z_scores > threshold].index.tolist()
        
        else:
            outliers = []
        
        return outliers
    
    def create_sequences(self, data, target_column, sequence_length=24, features=None):
        """
        Create sequences for time series modeling
        
        Args:
            data (pd.DataFrame): Input data
            target_column (str): Target column name
            sequence_length (int): Length of input sequences
            features (list): List of feature columns
        
        Returns:
            tuple: (X, y) sequences
        """
        if features is None:
            features = [target_column]
        
        # Ensure all feature columns exist
        features = [col for col in features if col in data.columns]
        
        X, y = [], []
        
        for i in range(sequence_length, len(data)):
            X.append(data[features].iloc[i-sequence_length:i].values)
            y.append(data[target_column].iloc[i])
        
        return np.array(X), np.array(y)
    
    def normalize_data(self, data, columns=None):
        """
        Normalize data using min-max scaling
        
        Args:
            data (pd.DataFrame): Data to normalize
            columns (list): Columns to normalize (if None, normalize all numeric columns)
        
        Returns:
            pd.DataFrame: Normalized data
        """
        from sklearn.preprocessing import MinMaxScaler
        
        normalized_data = data.copy()
        
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns
        
        scaler = MinMaxScaler()
        normalized_data[columns] = scaler.fit_transform(data[columns])
        
        self.scaler = scaler
        self.feature_columns = columns
        
        return normalized_data
    
    def denormalize_data(self, data, columns=None):
        """
        Denormalize data using stored scaler
        
        Args:
            data (pd.DataFrame or np.array): Data to denormalize
            columns (list): Columns to denormalize
        
        Returns:
            pd.DataFrame or np.array: Denormalized data
        """
        if self.scaler is None:
            return data
        
        if columns is None:
            columns = self.feature_columns
        
        if isinstance(data, pd.DataFrame):
            denormalized_data = data.copy()
            denormalized_data[columns] = self.scaler.inverse_transform(data[columns])
            return denormalized_data
        else:
            return self.scaler.inverse_transform(data)
