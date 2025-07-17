import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    def __init__(self):
        self.metrics_history = {}
    
    def calculate_metrics(self, y_true, y_pred, model_name=None):
        """
        Calculate comprehensive evaluation metrics for time series forecasting
        
        Args:
            y_true (array-like): True values
            y_pred (array-like): Predicted values
            model_name (str): Name of the model (optional)
        
        Returns:
            dict: Dictionary containing all calculated metrics
        """
        # Convert to numpy arrays
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        # Ensure same length
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        # Remove any NaN or infinite values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        if len(y_true) == 0:
            return {
                'mae': np.nan,
                'rmse': np.nan,
                'mse': np.nan,
                'mape': np.nan,
                'r2': np.nan,
                'smape': np.nan,
                'wape': np.nan,
                'mase': np.nan,
                'accuracy': np.nan,
                'bias': np.nan,
                'variance': np.nan,
                'std_error': np.nan
            }
        
        # Basic metrics
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        # R-squared
        try:
            r2 = r2_score(y_true, y_pred)
        except:
            r2 = np.nan
        
        # Mean Absolute Percentage Error (MAPE)
        mape = self._calculate_mape(y_true, y_pred)
        
        # Symmetric Mean Absolute Percentage Error (SMAPE)
        smape = self._calculate_smape(y_true, y_pred)
        
        # Weighted Absolute Percentage Error (WAPE)
        wape = self._calculate_wape(y_true, y_pred)
        
        # Mean Absolute Scaled Error (MASE)
        mase = self._calculate_mase(y_true, y_pred)
        
        # Accuracy (for percentage terms)
        accuracy = max(0, 100 - mape) if not np.isnan(mape) else np.nan
        
        # Bias and variance
        bias = np.mean(y_pred - y_true)
        variance = np.var(y_pred - y_true)
        std_error = np.std(y_pred - y_true)
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'mse': mse,
            'mape': mape,
            'r2': r2,
            'smape': smape,
            'wape': wape,
            'mase': mase,
            'accuracy': accuracy,
            'bias': bias,
            'variance': variance,
            'std_error': std_error
        }
        
        # Store metrics history if model name is provided
        if model_name:
            self.metrics_history[model_name] = metrics
        
        return metrics
    
    def _calculate_mape(self, y_true, y_pred):
        """Calculate Mean Absolute Percentage Error"""
        try:
            # Avoid division by zero
            mask = y_true != 0
            if np.sum(mask) == 0:
                return np.nan
            
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            return mape if not np.isnan(mape) else np.nan
        except:
            return np.nan
    
    def _calculate_smape(self, y_true, y_pred):
        """Calculate Symmetric Mean Absolute Percentage Error"""
        try:
            denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
            mask = denominator != 0
            if np.sum(mask) == 0:
                return np.nan
            
            smape = np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100
            return smape if not np.isnan(smape) else np.nan
        except:
            return np.nan
    
    def _calculate_wape(self, y_true, y_pred):
        """Calculate Weighted Absolute Percentage Error"""
        try:
            if np.sum(np.abs(y_true)) == 0:
                return np.nan
            
            wape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100
            return wape if not np.isnan(wape) else np.nan
        except:
            return np.nan
    
    def _calculate_mase(self, y_true, y_pred):
        """Calculate Mean Absolute Scaled Error"""
        try:
            if len(y_true) < 2:
                return np.nan
            
            # Calculate naive forecast error (seasonal naive with period=1)
            naive_error = np.mean(np.abs(y_true[1:] - y_true[:-1]))
            
            if naive_error == 0:
                return np.nan
            
            mae = np.mean(np.abs(y_true - y_pred))
            mase = mae / naive_error
            return mase if not np.isnan(mase) else np.nan
        except:
            return np.nan
    
    def calculate_directional_accuracy(self, y_true, y_pred):
        """
        Calculate directional accuracy (how well the model predicts the direction of change)
        
        Args:
            y_true (array-like): True values
            y_pred (array-like): Predicted values
        
        Returns:
            float: Directional accuracy as percentage
        """
        try:
            y_true = np.array(y_true).flatten()
            y_pred = np.array(y_pred).flatten()
            
            if len(y_true) < 2:
                return np.nan
            
            # Calculate direction of change
            true_direction = np.sign(np.diff(y_true))
            pred_direction = np.sign(np.diff(y_pred))
            
            # Calculate accuracy
            correct_directions = np.sum(true_direction == pred_direction)
            total_directions = len(true_direction)
            
            if total_directions == 0:
                return np.nan
            
            directional_accuracy = (correct_directions / total_directions) * 100
            return directional_accuracy
        except:
            return np.nan
    
    def calculate_forecast_bias(self, y_true, y_pred):
        """
        Calculate forecast bias metrics
        
        Args:
            y_true (array-like): True values
            y_pred (array-like): Predicted values
        
        Returns:
            dict: Dictionary containing bias metrics
        """
        try:
            y_true = np.array(y_true).flatten()
            y_pred = np.array(y_pred).flatten()
            
            # Ensure same length
            min_len = min(len(y_true), len(y_pred))
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]
            
            errors = y_pred - y_true
            
            # Mean bias
            mean_bias = np.mean(errors)
            
            # Mean absolute bias
            mean_abs_bias = np.mean(np.abs(errors))
            
            # Bias percentage
            bias_percentage = (mean_bias / np.mean(y_true)) * 100 if np.mean(y_true) != 0 else np.nan
            
            # Tracking signal
            tracking_signal = np.sum(errors) / np.sum(np.abs(errors)) if np.sum(np.abs(errors)) != 0 else np.nan
            
            return {
                'mean_bias': mean_bias,
                'mean_abs_bias': mean_abs_bias,
                'bias_percentage': bias_percentage,
                'tracking_signal': tracking_signal
            }
        except:
            return {
                'mean_bias': np.nan,
                'mean_abs_bias': np.nan,
                'bias_percentage': np.nan,
                'tracking_signal': np.nan
            }
    
    def calculate_prediction_intervals(self, y_true, y_pred, confidence_level=0.95):
        """
        Calculate prediction intervals for forecasts
        
        Args:
            y_true (array-like): True values
            y_pred (array-like): Predicted values
            confidence_level (float): Confidence level for intervals
        
        Returns:
            dict: Dictionary containing interval metrics
        """
        try:
            y_true = np.array(y_true).flatten()
            y_pred = np.array(y_pred).flatten()
            
            # Calculate residuals
            residuals = y_true - y_pred
            
            # Calculate standard error
            std_error = np.std(residuals)
            
            # Calculate z-score for confidence level
            from scipy import stats
            z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)
            
            # Calculate intervals
            lower_bound = y_pred - z_score * std_error
            upper_bound = y_pred + z_score * std_error
            
            # Calculate coverage
            coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound)) * 100
            
            # Calculate interval width
            interval_width = np.mean(upper_bound - lower_bound)
            
            return {
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'coverage': coverage,
                'interval_width': interval_width,
                'std_error': std_error
            }
        except:
            return {
                'lower_bound': np.array([]),
                'upper_bound': np.array([]),
                'coverage': np.nan,
                'interval_width': np.nan,
                'std_error': np.nan
            }
    
    def calculate_seasonal_metrics(self, y_true, y_pred, seasonal_period=24):
        """
        Calculate seasonal-specific metrics
        
        Args:
            y_true (array-like): True values
            y_pred (array-like): Predicted values
            seasonal_period (int): Length of seasonal period
        
        Returns:
            dict: Dictionary containing seasonal metrics
        """
        try:
            y_true = np.array(y_true).flatten()
            y_pred = np.array(y_pred).flatten()
            
            if len(y_true) < seasonal_period:
                return {'seasonal_mae': np.nan, 'seasonal_rmse': np.nan}
            
            # Calculate metrics for each season
            seasonal_mae = []
            seasonal_rmse = []
            
            for i in range(seasonal_period):
                # Get values for this season
                season_true = y_true[i::seasonal_period]
                season_pred = y_pred[i::seasonal_period]
                
                if len(season_true) > 0:
                    mae = np.mean(np.abs(season_true - season_pred))
                    rmse = np.sqrt(np.mean((season_true - season_pred)**2))
                    
                    seasonal_mae.append(mae)
                    seasonal_rmse.append(rmse)
            
            return {
                'seasonal_mae': np.array(seasonal_mae),
                'seasonal_rmse': np.array(seasonal_rmse),
                'avg_seasonal_mae': np.mean(seasonal_mae) if seasonal_mae else np.nan,
                'avg_seasonal_rmse': np.mean(seasonal_rmse) if seasonal_rmse else np.nan
            }
        except:
            return {
                'seasonal_mae': np.array([]),
                'seasonal_rmse': np.array([]),
                'avg_seasonal_mae': np.nan,
                'avg_seasonal_rmse': np.nan
            }
    
    def compare_models(self, models_metrics, primary_metric='rmse'):
        """
        Compare multiple models and rank them
        
        Args:
            models_metrics (dict): Dictionary of model metrics
            primary_metric (str): Primary metric for ranking
        
        Returns:
            dict: Comparison results
        """
        try:
            if not models_metrics:
                return {}
            
            # Create comparison DataFrame
            comparison_df = pd.DataFrame(models_metrics).T
            
            # Rank models by primary metric
            if primary_metric in comparison_df.columns:
                if primary_metric in ['r2', 'accuracy']:
                    # Higher is better
                    comparison_df['rank'] = comparison_df[primary_metric].rank(ascending=False)
                else:
                    # Lower is better
                    comparison_df['rank'] = comparison_df[primary_metric].rank(ascending=True)
            else:
                comparison_df['rank'] = 1
            
            # Best model
            best_model = comparison_df.loc[comparison_df['rank'].idxmin()]
            
            # Calculate relative performance
            relative_performance = {}
            for model_name in comparison_df.index:
                if primary_metric in comparison_df.columns:
                    base_value = comparison_df.loc[model_name, primary_metric]
                    best_value = best_model[primary_metric]
                    
                    if primary_metric in ['r2', 'accuracy']:
                        relative_performance[model_name] = (base_value / best_value) * 100 if best_value != 0 else 100
                    else:
                        relative_performance[model_name] = (best_value / base_value) * 100 if base_value != 0 else 100
                else:
                    relative_performance[model_name] = 100
            
            return {
                'comparison_table': comparison_df,
                'best_model': best_model.name,
                'best_metrics': best_model.to_dict(),
                'relative_performance': relative_performance,
                'ranking': comparison_df['rank'].to_dict()
            }
        except:
            return {}
    
    def generate_evaluation_report(self, model_name, y_true, y_pred, include_intervals=True):
        """
        Generate comprehensive evaluation report
        
        Args:
            model_name (str): Name of the model
            y_true (array-like): True values
            y_pred (array-like): Predicted values
            include_intervals (bool): Whether to include prediction intervals
        
        Returns:
            dict: Comprehensive evaluation report
        """
        try:
            # Basic metrics
            basic_metrics = self.calculate_metrics(y_true, y_pred, model_name)
            
            # Directional accuracy
            directional_acc = self.calculate_directional_accuracy(y_true, y_pred)
            
            # Bias metrics
            bias_metrics = self.calculate_forecast_bias(y_true, y_pred)
            
            # Seasonal metrics
            seasonal_metrics = self.calculate_seasonal_metrics(y_true, y_pred)
            
            # Prediction intervals
            interval_metrics = {}
            if include_intervals:
                interval_metrics = self.calculate_prediction_intervals(y_true, y_pred)
            
            # Residual analysis
            residuals = np.array(y_true) - np.array(y_pred)
            residual_stats = {
                'residual_mean': np.mean(residuals),
                'residual_std': np.std(residuals),
                'residual_skewness': self._calculate_skewness(residuals),
                'residual_kurtosis': self._calculate_kurtosis(residuals)
            }
            
            # Compile report
            report = {
                'model_name': model_name,
                'basic_metrics': basic_metrics,
                'directional_accuracy': directional_acc,
                'bias_metrics': bias_metrics,
                'seasonal_metrics': seasonal_metrics,
                'interval_metrics': interval_metrics,
                'residual_stats': residual_stats,
                'evaluation_summary': self._create_evaluation_summary(basic_metrics, directional_acc)
            }
            
            return report
        except Exception as e:
            return {
                'model_name': model_name,
                'error': f"Failed to generate evaluation report: {str(e)}"
            }
    
    def _calculate_skewness(self, data):
        """Calculate skewness of data"""
        try:
            from scipy import stats
            return stats.skew(data)
        except:
            return np.nan
    
    def _calculate_kurtosis(self, data):
        """Calculate kurtosis of data"""
        try:
            from scipy import stats
            return stats.kurtosis(data)
        except:
            return np.nan
    
    def _create_evaluation_summary(self, metrics, directional_acc):
        """Create evaluation summary"""
        summary = {
            'performance_level': 'Poor',
            'recommendations': []
        }
        
        try:
            # Determine performance level based on R²
            r2 = metrics.get('r2', 0)
            if r2 >= 0.9:
                summary['performance_level'] = 'Excellent'
            elif r2 >= 0.8:
                summary['performance_level'] = 'Good'
            elif r2 >= 0.6:
                summary['performance_level'] = 'Fair'
            else:
                summary['performance_level'] = 'Poor'
            
            # Generate recommendations
            mape = metrics.get('mape', 100)
            if mape > 20:
                summary['recommendations'].append("High MAPE suggests need for model improvement")
            
            if directional_acc < 60:
                summary['recommendations'].append("Poor directional accuracy - consider different model approach")
            
            bias = abs(metrics.get('bias', 0))
            if bias > np.std([metrics.get('mae', 0)]):
                summary['recommendations'].append("Model shows significant bias - consider bias correction")
            
            if not summary['recommendations']:
                summary['recommendations'].append("Model performance is acceptable")
            
            return summary
        except:
            return summary
    
    def export_metrics_to_csv(self, filename='model_metrics.csv'):
        """
        Export metrics history to CSV file
        
        Args:
            filename (str): Output filename
        
        Returns:
            pd.DataFrame: Metrics DataFrame
        """
        try:
            if not self.metrics_history:
                return pd.DataFrame()
            
            df = pd.DataFrame(self.metrics_history).T
            df.index.name = 'model_name'
            df.to_csv(filename)
            return df
        except:
            return pd.DataFrame()
    
    def get_metrics_summary(self):
        """
        Get summary of all calculated metrics
        
        Returns:
            dict: Summary of metrics
        """
        if not self.metrics_history:
            return {}
        
        summary = {}
        for model_name, metrics in self.metrics_history.items():
            summary[model_name] = {
                'mae': metrics.get('mae', np.nan),
                'rmse': metrics.get('rmse', np.nan),
                'mape': metrics.get('mape', np.nan),
                'r2': metrics.get('r2', np.nan),
                'accuracy': metrics.get('accuracy', np.nan)
            }
        
        return summary
