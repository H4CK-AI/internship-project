import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class DataVisualizer:
    def __init__(self):
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8',
            'light': '#f8f9fa',
            'dark': '#343a40'
        }
    
    def create_time_series_plot(self, data, date_column, value_column, title="Time Series Plot"):
        """
        Create a time series plot
        
        Args:
            data (pd.DataFrame): Data to plot
            date_column (str): Date column name
            value_column (str): Value column name
            title (str): Plot title
        
        Returns:
            plotly.graph_objects.Figure: Time series plot
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data[date_column],
            y=data[value_column],
            mode='lines',
            name=value_column,
            line=dict(color=self.colors['primary'], width=2)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title=value_column,
            hovermode='x unified',
            showlegend=True,
            height=400
        )
        
        return fig
    
    def create_seasonal_decomposition(self, data, date_column, value_column, 
                                    period=24, title="Seasonal Decomposition"):
        """
        Create seasonal decomposition plot
        
        Args:
            data (pd.DataFrame): Data to decompose
            date_column (str): Date column name
            value_column (str): Value column name
            period (int): Seasonal period
            title (str): Plot title
        
        Returns:
            plotly.graph_objects.Figure: Seasonal decomposition plot
        """
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            # Ensure data is sorted by date
            data_sorted = data.sort_values(date_column)
            
            # Perform seasonal decomposition
            decomposition = seasonal_decompose(
                data_sorted[value_column].dropna(),
                model='additive',
                period=period
            )
            
            # Create subplots
            fig = make_subplots(
                rows=4, cols=1,
                subplot_titles=('Original', 'Trend', 'Seasonal', 'Residual'),
                vertical_spacing=0.08
            )
            
            # Original data
            fig.add_trace(
                go.Scatter(
                    x=data_sorted[date_column],
                    y=data_sorted[value_column],
                    mode='lines',
                    name='Original',
                    line=dict(color=self.colors['primary'])
                ),
                row=1, col=1
            )
            
            # Trend
            fig.add_trace(
                go.Scatter(
                    x=data_sorted[date_column],
                    y=decomposition.trend,
                    mode='lines',
                    name='Trend',
                    line=dict(color=self.colors['success'])
                ),
                row=2, col=1
            )
            
            # Seasonal
            fig.add_trace(
                go.Scatter(
                    x=data_sorted[date_column],
                    y=decomposition.seasonal,
                    mode='lines',
                    name='Seasonal',
                    line=dict(color=self.colors['warning'])
                ),
                row=3, col=1
            )
            
            # Residual
            fig.add_trace(
                go.Scatter(
                    x=data_sorted[date_column],
                    y=decomposition.resid,
                    mode='lines',
                    name='Residual',
                    line=dict(color=self.colors['danger'])
                ),
                row=4, col=1
            )
            
            fig.update_layout(
                title=title,
                height=800,
                showlegend=False
            )
            
            return fig
            
        except ImportError:
            # Fallback to simple moving average if statsmodels not available
            return self._create_simple_trend_plot(data, date_column, value_column, title)
    
    def _create_simple_trend_plot(self, data, date_column, value_column, title):
        """
        Create a simple trend plot with moving average
        """
        fig = go.Figure()
        
        # Original data
        fig.add_trace(go.Scatter(
            x=data[date_column],
            y=data[value_column],
            mode='lines',
            name='Original',
            line=dict(color=self.colors['primary'], width=1)
        ))
        
        # Moving average
        if len(data) >= 24:
            ma_24 = data[value_column].rolling(window=24).mean()
            fig.add_trace(go.Scatter(
                x=data[date_column],
                y=ma_24,
                mode='lines',
                name='24-hour Moving Average',
                line=dict(color=self.colors['danger'], width=3)
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title=value_column,
            hovermode='x unified',
            showlegend=True,
            height=400
        )
        
        return fig
    
    def create_correlation_heatmap(self, data, title="Correlation Heatmap"):
        """
        Create correlation heatmap
        
        Args:
            data (pd.DataFrame): Data for correlation analysis
            title (str): Plot title
        
        Returns:
            plotly.graph_objects.Figure: Correlation heatmap
        """
        # Calculate correlation matrix
        corr_matrix = data.corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values.round(2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate='%{x} vs %{y}<br>Correlation: %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Features",
            yaxis_title="Features",
            height=600,
            width=600
        )
        
        return fig
    
    def create_distribution_plot(self, data, column, title="Distribution Plot"):
        """
        Create distribution plot
        
        Args:
            data (pd.DataFrame): Data to plot
            column (str): Column name
            title (str): Plot title
        
        Returns:
            plotly.graph_objects.Figure: Distribution plot
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Histogram', 'Box Plot', 'Violin Plot', 'Q-Q Plot'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(
                x=data[column],
                nbinsx=30,
                name='Histogram',
                marker_color=self.colors['primary']
            ),
            row=1, col=1
        )
        
        # Box plot
        fig.add_trace(
            go.Box(
                y=data[column],
                name='Box Plot',
                marker_color=self.colors['success']
            ),
            row=1, col=2
        )
        
        # Violin plot
        fig.add_trace(
            go.Violin(
                y=data[column],
                name='Violin Plot',
                marker_color=self.colors['warning']
            ),
            row=2, col=1
        )
        
        # Q-Q plot (approximation)
        sorted_data = np.sort(data[column].dropna())
        n = len(sorted_data)
        theoretical_quantiles = np.linspace(0, 1, n)
        
        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=sorted_data,
                mode='markers',
                name='Q-Q Plot',
                marker=dict(color=self.colors['danger'])
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=title,
            height=600,
            showlegend=False
        )
        
        return fig

class ModelVisualizer:
    def __init__(self):
        self.colors = {
            'actual': '#1f77b4',
            'predicted': '#ff7f0e',
            'train': '#2ca02c',
            'test': '#d62728',
            'confidence': 'rgba(255, 127, 14, 0.2)'
        }
    
    def plot_forecast_results(self, train_dates, test_dates, train_data, 
                            test_data, predictions, title="Forecast Results"):
        """
        Plot forecast results comparing actual vs predicted values
        
        Args:
            train_dates: Training dates
            test_dates: Test dates
            train_data: Training data
            test_data: Test data
            predictions: Model predictions
            title (str): Plot title
        
        Returns:
            plotly.graph_objects.Figure: Forecast plot
        """
        fig = go.Figure()
        
        # Training data
        if train_dates is not None:
            fig.add_trace(go.Scatter(
                x=train_dates,
                y=train_data,
                mode='lines',
                name='Training Data',
                line=dict(color=self.colors['train'], width=2)
            ))
        else:
            fig.add_trace(go.Scatter(
                y=train_data,
                mode='lines',
                name='Training Data',
                line=dict(color=self.colors['train'], width=2)
            ))
        
        # Actual test data
        if test_dates is not None:
            fig.add_trace(go.Scatter(
                x=test_dates,
                y=test_data,
                mode='lines',
                name='Actual',
                line=dict(color=self.colors['actual'], width=3)
            ))
        else:
            test_x = list(range(len(train_data), len(train_data) + len(test_data)))
            fig.add_trace(go.Scatter(
                x=test_x,
                y=test_data,
                mode='lines',
                name='Actual',
                line=dict(color=self.colors['actual'], width=3)
            ))
        
        # Predictions
        if test_dates is not None:
            fig.add_trace(go.Scatter(
                x=test_dates,
                y=predictions,
                mode='lines',
                name='Predicted',
                line=dict(color=self.colors['predicted'], width=3, dash='dash')
            ))
        else:
            pred_x = list(range(len(train_data), len(train_data) + len(predictions)))
            fig.add_trace(go.Scatter(
                x=pred_x,
                y=predictions,
                mode='lines',
                name='Predicted',
                line=dict(color=self.colors['predicted'], width=3, dash='dash')
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Value",
            hovermode='x unified',
            showlegend=True,
            height=500
        )
        
        return fig
    
    def plot_model_comparison(self, models_results, title="Model Comparison"):
        """
        Plot comparison of multiple models
        
        Args:
            models_results (dict): Dictionary of model results
            title (str): Plot title
        
        Returns:
            plotly.graph_objects.Figure: Comparison plot
        """
        fig = go.Figure()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, (model_name, results) in enumerate(models_results.items()):
            fig.add_trace(go.Scatter(
                y=results['predictions'],
                mode='lines',
                name=f'{model_name} Predictions',
                line=dict(color=colors[i % len(colors)], width=2)
            ))
        
        # Add actual values if available
        if 'actual' in list(models_results.values())[0]:
            fig.add_trace(go.Scatter(
                y=list(models_results.values())[0]['actual'],
                mode='lines',
                name='Actual',
                line=dict(color='black', width=3)
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Value",
            hovermode='x unified',
            showlegend=True,
            height=500
        )
        
        return fig
    
    def plot_residuals(self, actual, predicted, title="Residual Analysis"):
        """
        Plot residual analysis
        
        Args:
            actual (array): Actual values
            predicted (array): Predicted values
            title (str): Plot title
        
        Returns:
            plotly.graph_objects.Figure: Residual plot
        """
        residuals = actual - predicted
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Residuals vs Fitted', 'Residuals Distribution', 
                          'Actual vs Predicted', 'Residuals vs Time'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Residuals vs Fitted
        fig.add_trace(
            go.Scatter(
                x=predicted,
                y=residuals,
                mode='markers',
                name='Residuals vs Fitted',
                marker=dict(color=self.colors['actual'], size=6)
            ),
            row=1, col=1
        )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
        
        # Residuals distribution
        fig.add_trace(
            go.Histogram(
                x=residuals,
                nbinsx=30,
                name='Residuals Distribution',
                marker_color=self.colors['predicted']
            ),
            row=1, col=2
        )
        
        # Actual vs Predicted
        fig.add_trace(
            go.Scatter(
                x=actual,
                y=predicted,
                mode='markers',
                name='Actual vs Predicted',
                marker=dict(color=self.colors['train'], size=6)
            ),
            row=2, col=1
        )
        
        # Add perfect prediction line
        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ),
            row=2, col=1
        )
        
        # Residuals vs Time
        fig.add_trace(
            go.Scatter(
                y=residuals,
                mode='lines+markers',
                name='Residuals vs Time',
                line=dict(color=self.colors['test']),
                marker=dict(size=4)
            ),
            row=2, col=2
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=2)
        
        fig.update_layout(
            title=title,
            height=600,
            showlegend=False
        )
        
        return fig
    
    def plot_prediction_intervals(self, dates, predictions, lower_bound, 
                                upper_bound, actual=None, title="Prediction Intervals"):
        """
        Plot predictions with confidence intervals
        
        Args:
            dates: Date values
            predictions: Predicted values
            lower_bound: Lower confidence bound
            upper_bound: Upper confidence bound
            actual: Actual values (optional)
            title (str): Plot title
        
        Returns:
            plotly.graph_objects.Figure: Prediction intervals plot
        """
        fig = go.Figure()
        
        # Add upper bound (invisible line for fill)
        fig.add_trace(go.Scatter(
            x=dates,
            y=upper_bound,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            name='Upper Bound'
        ))
        
        # Add lower bound with fill
        fig.add_trace(go.Scatter(
            x=dates,
            y=lower_bound,
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor=self.colors['confidence'],
            name='Confidence Interval'
        ))
        
        # Add predictions
        fig.add_trace(go.Scatter(
            x=dates,
            y=predictions,
            mode='lines',
            name='Predictions',
            line=dict(color=self.colors['predicted'], width=3)
        ))
        
        # Add actual values if provided
        if actual is not None:
            fig.add_trace(go.Scatter(
                x=dates,
                y=actual,
                mode='lines',
                name='Actual',
                line=dict(color=self.colors['actual'], width=2)
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Value",
            hovermode='x unified',
            showlegend=True,
            height=500
        )
        
        return fig
