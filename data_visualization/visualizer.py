# data_visualization/visualizer.py

import os
import json
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from typing import Dict, List, Any, Union, Tuple

class DataVisualizer:
    """Creates improved visualizations from extracted structured data"""
    
    def __init__(self, output_dir="./visualizations"):
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Configure visualization styles
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (10, 6)
        
        # Enhanced color palettes
        self.color_palettes = {
            "default": px.colors.qualitative.Plotly,
            "sequential": px.colors.sequential.Blues,
            "sequential_green": px.colors.sequential.Greens,
            "sequential_red": px.colors.sequential.Reds,
            "diverging": px.colors.diverging.RdBu,
            "categorical": px.colors.qualitative.Safe
        }
    
    def visualize_extraction(self, extraction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create an enhanced visualization based on data type
        
        Args:
            extraction: Data extraction object containing data and metadata
            
        Returns:
            Visualization result with paths and descriptions
        """
        data = extraction.get('data', [])
        data_type = extraction.get('data_type', '')
        description = extraction.get('description', '')
        
        # Skip if no data
        if not data:
            return {"error": "No data to visualize"}
        
        print(f"Visualizing {data_type} data: {description}")
        
        # Choose visualization based on data type
        if data_type == 'table':
            return self._visualize_table(data, description, extraction)
        elif data_type == 'financial_figures':
            return self._visualize_financial_figures(data, description, extraction)
        elif data_type == 'percentages':
            return self._visualize_percentages(data, description, extraction)
        elif data_type == 'time_series':
            return self._visualize_time_series(data, description, extraction)
        elif data_type == 'multi_time_series':
            return self._visualize_multi_time_series(data, description, extraction)
        elif data_type == 'financials':
            return self._visualize_financials(data, description, extraction)
        elif data_type == 'metrics':
            return self._visualize_metrics(data, description, extraction)
        elif data_type == 'trends':
            return self._visualize_trends(data, description, extraction)
        elif data_type == 'comparison':
            return self._visualize_comparison(data, description, extraction)
        elif data_type == 'comparisons':
            return self._visualize_comparisons(data, description, extraction)
        else:
            # Generic visualization for unknown types
            return self._visualize_generic(data, description, extraction)
    
    def _get_filename(self, prefix: str, extraction: Dict[str, Any] = None) -> str:
        """Generate a unique filename for a visualization"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        
        # If extraction has an ID, use it to make filename more identifiable
        if extraction and 'id' in extraction:
            return f"{prefix}_{extraction['id']}_{timestamp}"
        else:
            return f"{prefix}_{timestamp}"
    
    def _normalize_period_labels(self, periods):
        """Normalize time period labels for consistent visualization"""
        normalized = []
        
        for period in periods:
            period_str = str(period).strip().lower()
            
            # Convert quarters
            if re.match(r'q[1-4]', period_str):
                normalized.append(period_str.upper())
            # Convert months
            elif period_str in ['jan', 'january']:
                normalized.append('Jan')
            elif period_str in ['feb', 'february']:
                normalized.append('Feb')
            elif period_str in ['mar', 'march']:
                normalized.append('Mar')
            elif period_str in ['apr', 'april']:
                normalized.append('Apr')
            elif period_str in ['may']:
                normalized.append('May')
            elif period_str in ['jun', 'june']:
                normalized.append('Jun')
            elif period_str in ['jul', 'july']:
                normalized.append('Jul')
            elif period_str in ['aug', 'august']:
                normalized.append('Aug')
            elif period_str in ['sep', 'september']:
                normalized.append('Sep')
            elif period_str in ['oct', 'october']:
                normalized.append('Oct')
            elif period_str in ['nov', 'november']:
                normalized.append('Nov')
            elif period_str in ['dec', 'december']:
                normalized.append('Dec')
            else:
                normalized.append(period_str)
                
        return normalized
    
    def _visualize_financial_figures(self, data: List[Dict[str, Any]], description: str, extraction: Dict[str, Any]) -> Dict[str, Any]:
        """Create visualization for financial figures with labels"""
        # Handle different data formats
        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
            # Data is a list of dictionaries with 'label' and 'value'
            labels = [item.get('label', f'Value {i+1}') for i, item in enumerate(data)]
            values = [item.get('value', 0) for item in data]
        else:
            # Default handling as before
            values = data if isinstance(data, list) else [data]
            labels = [f'Value {i+1}' for i in range(len(values))]
        
        # Create DataFrame
        df = pd.DataFrame({
            'Label': labels,
            'Value': values
        })
        
        # Sort by value for better visualization
        df = df.sort_values('Value', ascending=False)
        
        # Create horizontal bar chart
        fig = px.bar(
            df,
            y='Label',
            x='Value',
            title='Financial Figures',
            orientation='h',
            color='Value',
            color_continuous_scale=self.color_palettes['sequential'],
            text='Value'
        )
        
        # Improve label formatting for currency values
        fig.update_traces(
            texttemplate='%{x:,.0f}',
            textposition='outside'
        )
        
        fig.update_layout(
            xaxis_title='Amount',
            yaxis_title='',
            yaxis={'categoryorder': 'total ascending'},
            coloraxis_showscale=False
        )
        
        # Save visualization
        filename = self._get_filename("financial_figures", extraction)
        fig_path = f"{self.output_dir}/{filename}.html"
        fig.write_html(fig_path)
        
        return {
            'type': 'financial_figures',
            'title': 'Financial Figures',
            'description': description,
            'data': {'labels': labels, 'values': values},
            'file_path': fig_path,
            'filename': f"{filename}.html"
        }
    
    def _visualize_percentages(self, data: List[Dict[str, Any]], description: str, extraction: Dict[str, Any]) -> Dict[str, Any]:
        """Create visualization for percentage values with labels"""
        # Handle different data formats
        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
            # Data is a list of dictionaries with 'label' and 'value'
            labels = [item.get('label', f'Percentage {i+1}') for i, item in enumerate(data)]
            values = [item.get('value', 0) for item in data]
        else:
            # Default handling as before
            values = data if isinstance(data, list) else [data]
            labels = [f'Percentage {i+1}' for i in range(len(values))]
        
        # Create DataFrame
        df = pd.DataFrame({
            'Label': labels,
            'Value': values
        })
        
        # Sort by value for better visualization
        df = df.sort_values('Value', ascending=False)
        
        # Create horizontal bar chart or pie chart based on number of items
        if len(df) <= 5:
            # Pie chart for fewer items
            fig = px.pie(
                df,
                names='Label',
                values='Value',
                title='Percentage Distribution',
                color_discrete_sequence=self.color_palettes['categorical']
            )
            
            fig.update_traces(textposition='inside', textinfo='label+percent')
            
        else:
            # Bar chart for more items
            fig = px.bar(
                df,
                y='Label',
                x='Value',
                title='Percentage Values',
                orientation='h',
                color='Value',
                color_continuous_scale=self.color_palettes['sequential'],
                text='Value'
            )
            
            fig.update_traces(
                texttemplate='%{x:.1f}%',
                textposition='outside'
            )
            
            fig.update_layout(
                xaxis_title='Percentage',
                yaxis_title='',
                yaxis={'categoryorder': 'total ascending'},
                coloraxis_showscale=False
            )
        
        # Save visualization
        filename = self._get_filename("percentages", extraction)
        fig_path = f"{self.output_dir}/{filename}.html"
        fig.write_html(fig_path)
        
        return {
            'type': 'percentages',
            'title': 'Percentage Analysis',
            'description': description,
            'data': {'labels': labels, 'values': values},
            'file_path': fig_path,
            'filename': f"{filename}.html"
        }
    
    def _visualize_time_series(self, data: List[Dict[str, Any]], description: str, extraction: Dict[str, Any]) -> Dict[str, Any]:
        """Create enhanced time series visualization"""
        # Handle different possible formats
        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
            # Extract periods, values, and metric
            periods = [item.get('period', f'Period {i+1}') for i, item in enumerate(data)]
            values = [item.get('value', 0) for item in data]
            metric = data[0].get('metric', 'Value') if data else 'Value'
        else:
            # Fall back to basic format
            periods = [f'Period {i+1}' for i in range(len(data))]
            values = data
            metric = 'Value'
        
        # Normalize period labels
        normalized_periods = self._normalize_period_labels(periods)
        
        # Create DataFrame
        df = pd.DataFrame({
            'Period': normalized_periods,
            metric: values
        })
        
        # Sort by period if possible
        try:
            df = df.sort_values('Period')
        except:
            pass
        
        # Create Plotly line chart with improved styling
        fig = px.line(
            df,
            x='Period',
            y=metric,
            title=f'{metric} Over Time',
            markers=True,
            labels={metric: metric, 'Period': 'Time Period'},
            color_discrete_sequence=self.color_palettes['default']
        )
        
        # Enhance with better formatting
        fig.update_layout(
            xaxis_title='Time Period',
            yaxis_title=metric,
            xaxis={'categoryorder': 'array', 'categoryarray': normalized_periods},
            yaxis={'tickformat': ',.0f'} if max(values) > 100 else {'tickformat': '.2f'}
        )
        
        # Add markers and hover info
        fig.update_traces(
            mode='lines+markers',
            marker=dict(size=8),
            hovertemplate='%{x}: %{y:,.2f}<extra></extra>'
        )
        
        # Add data points as text for clarity
        fig.update_traces(
            textposition="top center",
            texttemplate='%{y:,.1f}' if max(values) < 10000 else '%{y:,.0f}'
        )
        
        # Save visualization
        filename = self._get_filename("time_series", extraction)
        fig_path = f"{self.output_dir}/{filename}.html"
        fig.write_html(fig_path)
        
        # Calculate statistics
        stats = {
            'count': len(values),
            'min': float(min(values)),
            'max': float(max(values)),
            'mean': float(np.mean(values)),
            'trend': 'Increasing' if values[-1] > values[0] else 'Decreasing' if values[-1] < values[0] else 'Stable',
            'change_pct': float(((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0)
        }
        
        return {
            'type': 'time_series',
            'title': f'{metric} Over Time',
            'description': description,
            'metric': metric,
            'data': {'periods': normalized_periods, 'values': values},
            'stats': stats,
            'file_path': fig_path,
            'filename': f"{filename}.html"
        }
    
    def _visualize_multi_time_series(self, data: List[Dict[str, Any]], description: str, extraction: Dict[str, Any]) -> Dict[str, Any]:
        """Create visualization for multiple time series"""
        # Group by metric
        metric_groups = {}
        
        for item in data:
            metric = item.get('metric', 'Value')
            if metric not in metric_groups:
                metric_groups[metric] = []
            
            metric_groups[metric].append({
                'period': item.get('period', ''),
                'value': item.get('value', 0)
            })
        
        # Create a figure with multiple traces
        fig = go.Figure()
        
        metrics = []
        all_periods = set()
        
        # Add each metric as a separate line
        for metric, values in metric_groups.items():
            periods = [item['period'] for item in values]
            data_values = [item['value'] for item in values]
            
            # Collect all periods for x-axis ordering
            all_periods.update(periods)
            
            # Normalize period labels
            normalized_periods = self._normalize_period_labels(periods)
            
            # Add trace for this metric
            fig.add_trace(go.Scatter(
                x=normalized_periods,
                y=data_values,
                name=metric,
                mode='lines+markers',
                marker=dict(size=8),
                textposition="top center",
                texttemplate='%{y:,.1f}' if max(data_values) < 10000 else '%{y:,.0f}'
            ))
            
            metrics.append(metric)
        
        # Update layout
        fig.update_layout(
            title='Multiple Metrics Over Time',
            xaxis_title='Time Period',
            yaxis_title='Value',
            legend_title='Metric',
            hovermode="x unified"
        )
        
        # Save visualization
        filename = self._get_filename("multi_time_series", extraction)
        fig_path = f"{self.output_dir}/{filename}.html"
        fig.write_html(fig_path)
        
        return {
            'type': 'multi_time_series',
            'title': 'Multiple Metrics Over Time',
            'description': description,
            'metrics': metrics,
            'file_path': fig_path,
            'filename': f"{filename}.html"
        }
    
    def _visualize_table(self, data: List[Dict[str, Any]], description: str, extraction: Dict[str, Any]) -> Dict[str, Any]:
        """Create an enhanced table visualization"""
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Check if we have a specific table type
        table_type = extraction.get('table_type', 'Generic Data')
        
        # Sample first rows for large tables
        if len(df) > 10:
            display_df = df.head(10)
        else:
            display_df = df
        
        # Create Plotly table with improved styling
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(df.columns),
                fill_color='#6c7ae0',
                align='center',
                font=dict(color='white', size=12),
                height=40
            ),
            cells=dict(
                values=[display_df[col] for col in display_df.columns],
                fill_color=[['#f5f7ff', '#edf0ff'] * (len(display_df) // 2 + 1)],
                align=['left' if isinstance(df[col].iloc[0], str) else 'right' for col in df.columns],
                font=dict(color='#333', size=11),
                height=30,
                format=[None if isinstance(df[col].iloc[0], str) else ',.2f' for col in df.columns]
            )
        )])
        
        fig.update_layout(
            title=f'{table_type} (First 10 rows)',
            height=400 + len(display_df) * 25,
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        # Look for numeric columns for a secondary visualization
        numeric_cols = df.select_dtypes(include=['number']).columns
        time_cols = [col for col in df.columns if any(term in col.lower() for term in ['date', 'time', 'year', 'month', 'day'])]
        
        secondary_fig = None
        
        # If we have numeric data and time columns, create a line chart
        if len(numeric_cols) > 0 and len(time_cols) > 0:
            time_col = time_cols[0]
            
            for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                try:
                    secondary_fig = px.line(
                        df,
                        x=time_col,
                        y=col,
                        title=f'{col} by {time_col}',
                        markers=True
                    )
                    break
                except:
                    pass
        
        # If we have just numeric columns, create a bar chart of the first column
        elif len(numeric_cols) > 0 and len(df.columns) > 1:
            try:
                category_col = [col for col in df.columns if col not in numeric_cols][0]
                value_col = numeric_cols[0]
                
                # Limit to top 10 categories by value
                top_df = df.sort_values(value_col, ascending=False).head(10)
                
                secondary_fig = px.bar(
                    top_df,
                    x=category_col,
                    y=value_col,
                    title=f'Top {category_col} by {value_col}',
                    color=value_col,
                    color_continuous_scale=self.color_palettes['sequential']
                )
            except:
                pass
        
        # Save visualizations
        filename = self._get_filename("table", extraction)
        fig_path = f"{self.output_dir}/{filename}.html"
        fig.write_html(fig_path)
        
        secondary_path = None
        if secondary_fig:
            secondary_filename = f"{filename}_chart"
            secondary_path = f"{self.output_dir}/{secondary_filename}.html"
            secondary_fig.write_html(secondary_path)
        
        # Also save the data as CSV
        csv_path = f"{self.output_dir}/{filename}.csv"
        df.to_csv(csv_path, index=False)
        
        result = {
            'type': 'table',
            'title': table_type,
            'description': description,
            'shape': {'rows': len(df), 'columns': len(df.columns)},
            'columns': list(df.columns),
            'file_path': fig_path,
            'csv_path': csv_path,
            'filename': f"{filename}.html"
        }
        
        if secondary_path:
            result['secondary_chart'] = {
                'path': secondary_path,
                'filename': f"{filename}_chart.html"
            }
        
        return result
    
    def _visualize_financials(self, data: Union[Dict[str, Any], List[Dict[str, Any]]], description: str, extraction: Dict[str, Any]) -> Dict[str, Any]:
        """Create enhanced financial data visualization"""
        
        # Detect financial data format
        if isinstance(data, dict) and not any(isinstance(v, (list, dict)) for v in data.values()):
            # Single period financial data (dictionary of metrics)
            return self._visualize_financial_metrics(data, description, extraction)
        elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
            # Check if it contains period-based data
            if all('period' in item for item in data):
                return self._visualize_financial_time_series(data, description, extraction)
            else:
                # List of financial metrics
                metrics_data = {}
                for item in data:
                    metrics_data.update(item)
                return self._visualize_financial_metrics(metrics_data, description, extraction)
        else:
            # Try generic approach
            return self._visualize_generic(data, description, extraction)
    
    def _visualize_financial_metrics(self, data: Dict[str, Any], description: str, extraction: Dict[str, Any]) -> Dict[str, Any]:
        """Visualize financial metrics as a bar chart with improved labeling"""
        # Extract metrics and values
        metrics = []
        values = []
        
        for key, value in data.items():
            # Skip non-numeric values
            if isinstance(value, (int, float)):
                # Clean up the metric name for display
                metric_name = key.replace('_', ' ').title()
                metrics.append(metric_name)
                values.append(value)
        
        # Create DataFrame
        df = pd.DataFrame({
            'Metric': metrics,
            'Value': values
        })
        
        # Sort by value for better visualization
        df = df.sort_values('Value', ascending=False)
        
        # Create horizontal bar chart for better readability
        fig = px.bar(
            df,
            y='Metric',
            x='Value',
            title='Financial Metrics',
            orientation='h',
            color='Value',
            color_continuous_scale=self.color_palettes['sequential_green'],
            text='Value'
        )
        
        # Improve text formatting for currency values
        fig.update_traces(
            texttemplate='%{x:,.0f}',
            textposition='outside'
        )
        
        fig.update_layout(
            xaxis_title='Amount',
            yaxis_title='',
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        
        # Save visualization
        filename = self._get_filename("financial_metrics", extraction)
        fig_path = f"{self.output_dir}/{filename}.html"
        fig.write_html(fig_path)
        
        return {
            'type': 'financial_metrics',
            'title': 'Financial Metrics',
            'description': description,
            'metrics': {metric: value for metric, value in zip(metrics, values)},
            'file_path': fig_path,
            'filename': f"{filename}.html"
        }
    
    def _visualize_financial_time_series(self, data: List[Dict[str, Any]], description: str, extraction: Dict[str, Any]) -> Dict[str, Any]:
        """Visualize financial data over time with enhanced styling"""
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Extract time periods
        periods = df['period'].tolist()
        normalized_periods = self._normalize_period_labels(periods)
        
        # Get metrics (all columns except 'period')
        metrics = [col for col in df.columns if col != 'period']
        
        # Create a more meaningful title based on metrics
        if 'revenue' in metrics and 'profit' in metrics:
            title = 'Revenue and Profit Over Time'
        elif 'revenue' in metrics:
            title = 'Revenue Over Time'
        elif 'profit' in metrics:
            title = 'Profit Over Time'
        else:
            title = 'Financial Metrics Over Time'
        
        # Create interactive line chart for each metric
        fig = go.Figure()
        
        # Define a color map for common financial metrics
        color_map = {
            'revenue': '#4CAF50',    # Green
            'profit': '#2196F3',     # Blue
            'margin': '#9C27B0',     # Purple
            'costs': '#F44336',      # Red
            'expenses': '#FF9800',   # Orange
            'income': '#3F51B5',     # Indigo
            'sales': '#00BCD4'       # Cyan
        }
        
        # Add line for each metric with appropriate styling
        for i, metric in enumerate(metrics):
            metric_name = metric.replace('_', ' ').title()
            
            # Assign appropriate color
            color = color_map.get(metric.lower(), self.color_palettes['default'][i % len(self.color_palettes['default'])])
            
            # Determine line style based on metric type
            if any(term in metric.lower() for term in ['margin', 'percentage', 'ratio', 'rate']):
                # Use dashed line for percentage metrics
                line_style = dict(dash='dash', width=3)
                # And right y-axis
                yaxis = 'y2'
            else:
                # Solid line for absolute values
                line_style = dict(width=3)
                yaxis = 'y'
            
            fig.add_trace(go.Scatter(
                x=normalized_periods,
                y=df[metric],
                name=metric_name,
                mode='lines+markers',
                line=line_style,
                marker=dict(size=8, color=color),
                yaxis=yaxis
            ))
        
        # Create y-axis title based on metrics
        if all(any(term in metric.lower() for term in ['revenue', 'sales', 'cost', 'expense', 'profit', 'income']) for metric in metrics):
            y_title = "Amount"
        else:
            y_title = "Value"
            
        # Update layout with dual y-axis if needed
        if any(any(term in metric.lower() for term in ['margin', 'percentage', 'ratio', 'rate']) for metric in metrics):
            fig.update_layout(
                title=title,
                xaxis_title='Period',
                yaxis=dict(
                    title=y_title,
                    showgrid=True,
                    tickformat=',.'
                ),
                yaxis2=dict(
                    title='Percentage',
                    showgrid=False,
                    overlaying='y',
                    side='right',
                    tickformat='.1%'
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                ),
                hovermode="x unified"
            )
        else:
            fig.update_layout(
                title=title,
                xaxis_title='Period',
                yaxis_title=y_title,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                ),
                hovermode="x unified"
            )
        
        # Save visualization
        filename = self._get_filename("financial_time_series", extraction)
        fig_path = f"{self.output_dir}/{filename}.html"
        fig.write_html(fig_path)
        
        # Calculate key statistics about the data
        stats = {}
        for metric in metrics:
            if not df[metric].empty:
                stats[f"{metric}_growth"] = float(((df[metric].iloc[-1] - df[metric].iloc[0]) / df[metric].iloc[0] * 100) if df[metric].iloc[0] != 0 else 0)
                stats[f"{metric}_avg"] = float(df[metric].mean())
                stats[f"{metric}_max"] = float(df[metric].max())
        
        return {
            'type': 'financial_time_series',
            'title': title,
            'description': description,
            'periods': normalized_periods,
            'metrics': metrics,
            'stats': stats,
            'file_path': fig_path,
            'filename': f"{filename}.html"
        }
    
    def _visualize_metrics(self, data: Union[Dict[str, Any], List[Dict[str, Any]]], description: str, extraction: Dict[str, Any]) -> Dict[str, Any]:
        """Visualize metrics with enhanced formatting and charts"""
        # Handle different data formats
        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
            # List of metric objects
            if all('metric' in item and 'value' in item for item in data):
                # Create a dictionary of metrics and values
                metrics_data = {}
                for item in data:
                    metrics_data[item['metric']] = item['value']
                
                # Visualize as radar chart if we have enough metrics
                if len(metrics_data) >= 3:
                    return self._visualize_metrics_radar(metrics_data, description, extraction)
                else:
                    return self._visualize_metrics_bar(metrics_data, description, extraction)
            
            # Time series of metrics
            elif all('periods' in item and 'values' in item for item in data):
                return self._visualize_metrics_time_series(data, description, extraction)
            
            # Generic handling
            else:
                return self._visualize_generic(data, description, extraction)
        
        elif isinstance(data, dict):
            # Check if it's a time series
            if 'periods' in data and 'values' in data:
                return self._visualize_metrics_time_series([data], description, extraction)
            else:
                # Simple dictionary of metrics and values
                return self._visualize_metrics_bar(data, description, extraction)
        
        else:
            return self._visualize_generic(data, description, extraction)
    
    def _visualize_metrics_bar(self, data: Dict[str, Any], description: str, extraction: Dict[str, Any]) -> Dict[str, Any]:
        """Visualize metrics as a bar chart"""
        # Extract metrics and values
        metrics = []
        values = []
        
        for key, value in data.items():
            # Convert to numeric if possible
            try:
                if isinstance(value, str):
                    # Try to extract numeric value
                    numeric_match = re.search(r'(\d+(?:\.\d+)?)', value)
                    if numeric_match:
                        value = float(numeric_match.group(1))
                    else:
                        continue
                
                metrics.append(key)
                values.append(float(value))
            except:
                pass
        
        # Create DataFrame
        df = pd.DataFrame({
            'Metric': metrics,
            'Value': values
        })
        
        # Sort by value for better visualization
        df = df.sort_values('Value', ascending=False)
        
        # Create bar chart
        fig = px.bar(
            df,
            x='Metric',
            y='Value',
            title='Key Metrics',
            color='Value',
            color_continuous_scale=self.color_palettes['sequential'],
            text='Value'
        )
        
        # Format text values
        fig.update_traces(
            texttemplate='%{y:,.2f}',
            textposition='outside'
        )
        
        fig.update_layout(
            xaxis_title='',
            yaxis_title='Value',
            coloraxis_showscale=False
        )
        
        # Save visualization
        filename = self._get_filename("metrics_bar", extraction)
        fig_path = f"{self.output_dir}/{filename}.html"
        fig.write_html(fig_path)
        
        return {
            'type': 'metrics_bar',
            'title': 'Key Metrics',
            'description': description,
            'metrics': {metric: value for metric, value in zip(metrics, values)},
            'file_path': fig_path,
            'filename': f"{filename}.html"
        }
    
    def _visualize_metrics_radar(self, data: Dict[str, Any], description: str, extraction: Dict[str, Any]) -> Dict[str, Any]:
        """Visualize metrics as a radar chart"""
        # Extract metrics and values
        metrics = []
        values = []
        
        for key, value in data.items():
            # Convert to numeric if possible
            try:
                if isinstance(value, str):
                    # Try to extract numeric value
                    numeric_match = re.search(r'(\d+(?:\.\d+)?)', value)
                    if numeric_match:
                        value = float(numeric_match.group(1))
                    else:
                        continue
                
                metrics.append(key)
                values.append(float(value))
            except:
                pass
        
        # Create radar chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics,
            fill='toself',
            name='Metrics',
            line=dict(color='rgb(31, 119, 180)', width=3),
            fillcolor='rgba(31, 119, 180, 0.2)'
        ))
        
        fig.update_layout(
            title='Metrics Overview',
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(values) * 1.1]
                )
            ),
            showlegend=False
        )
        
        # Save visualization
        filename = self._get_filename("metrics_radar", extraction)
        fig_path = f"{self.output_dir}/{filename}.html"
        fig.write_html(fig_path)
        
        return {
            'type': 'metrics_radar',
            'title': 'Metrics Overview',
            'description': description,
            'metrics': {metric: value for metric, value in zip(metrics, values)},
            'file_path': fig_path,
            'filename': f"{filename}.html"
        }
    
    def _visualize_metrics_time_series(self, data: List[Dict[str, Any]], description: str, extraction: Dict[str, Any]) -> Dict[str, Any]:
        """Visualize metrics over time"""
        fig = go.Figure()
        
        for item in data:
            metric = item.get('metric', 'Value')
            periods = item.get('periods', [])
            values = item.get('values', [])
            unit = item.get('unit', '')
            
            # Normalize periods
            normalized_periods = self._normalize_period_labels(periods)
            
            # Add trace for this metric
            fig.add_trace(go.Scatter(
                x=normalized_periods,
                y=values,
                name=metric,
                mode='lines+markers',
                marker=dict(size=8)
            ))
        
        # Update layout
        y_title = f"Value ({unit})" if unit else "Value"
        
        fig.update_layout(
            title='Metrics Over Time',
            xaxis_title='Period',
            yaxis_title=y_title,
            legend_title='Metric',
            hovermode="x unified"
        )
        
        # Save visualization
        filename = self._get_filename("metrics_time_series", extraction)
        fig_path = f"{self.output_dir}/{filename}.html"
        fig.write_html(fig_path)
        
        return {
            'type': 'metrics_time_series',
            'title': 'Metrics Over Time',
            'description': description,
            'file_path': fig_path,
            'filename': f"{filename}.html"
        }
    
    def _visualize_trends(self, data: List[Dict[str, Any]], description: str, extraction: Dict[str, Any]) -> Dict[str, Any]:
        """Visualize trend data with enhanced styling"""
        # Extract relevant data
        metrics = []
        directions = []
        magnitudes = []
        periods = []
        
        for item in data:
            metrics.append(item.get('metric', 'Unknown Metric'))
            directions.append(item.get('direction', 'stable').lower())
            
            # Get magnitude with default value of 1
            magnitude = item.get('magnitude', 1)
            if isinstance(magnitude, str):
                try:
                    magnitude = float(re.search(r'(\d+(?:\.\d+)?)', magnitude).group(1))
                except:
                    magnitude = 1
            
            # Convert to signed value based on direction
            if directions[-1] == 'increasing':
                magnitudes.append(abs(magnitude))
            elif directions[-1] == 'decreasing':
                magnitudes.append(-abs(magnitude))
            else:  # stable
                magnitudes.append(0)
                
            # Capture period if available
            periods.append(item.get('period', ''))
        
        # Create a DataFrame
        df = pd.DataFrame({
            'Metric': metrics,
            'Direction': directions,
            'Magnitude': magnitudes,
            'Period': periods
        })
        
        # Sort by magnitude for better visualization
        df = df.sort_values('Magnitude')
        
        # Define colors based on direction (red for decreasing, green for increasing)
        colors = ['red' if val < 0 else 'green' if val > 0 else 'gray' for val in df['Magnitude']]
        
        # Create horizontal bar chart
        fig = go.Figure()
        
        # Add trace for the bars
        fig.add_trace(go.Bar(
            y=df['Metric'],
            x=df['Magnitude'],
            orientation='h',
            marker_color=colors,
            text=[f"{abs(val):.1f}" for val in df['Magnitude']],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Change: %{x:+.1f}<br>Direction: %{customdata}<extra></extra>',
            customdata=df['Direction']
        ))
        
        # Add reference line for zero
        fig.add_shape(
            type="line",
            x0=0, y0=-0.5,
            x1=0, y1=len(df)-0.5,
            line=dict(color="black", width=2, dash="dot")
        )
        
        # Add annotations to indicate direction
        for i, (metric, direction, magnitude) in enumerate(zip(df['Metric'], df['Direction'], df['Magnitude'])):
            icon = "↑" if direction == "increasing" else "↓" if direction == "decreasing" else "→"
            fig.add_annotation(
                x=0,
                y=metric,
                text=icon,
                showarrow=False,
                font=dict(
                    size=20,
                    color="green" if direction == "increasing" else "red" if direction == "decreasing" else "gray"
                ),
                xanchor="center"
            )
        
        fig.update_layout(
            title='Trends Analysis',
            xaxis_title='Change Magnitude (+ increasing, - decreasing)',
            yaxis_title='',
            xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black')
        )
        
        # Save visualization
        filename = self._get_filename("trends", extraction)
        fig_path = f"{self.output_dir}/{filename}.html"
        fig.write_html(fig_path)
        
        return {
            'type': 'trends',
            'title': 'Trends Analysis',
            'description': description,
            'data': [{'metric': metric, 'direction': direction, 'magnitude': mag} 
                     for metric, direction, mag in zip(df['Metric'], df['Direction'], df['Magnitude'])],
            'file_path': fig_path,
            'filename': f"{filename}.html"
        }
    
    def _visualize_comparison(self, data: Dict[str, Any], description: str, extraction: Dict[str, Any]) -> Dict[str, Any]:
        """Visualize a single comparison"""
        # Check for expected keys
        items = data.get('items', [])
        values = data.get('values', [])
        metric = data.get('metric', 'Comparison')
        unit = data.get('unit', '')
        
        # Make sure we have items and values to compare
        if not items or not values:
            return self._visualize_generic(data, description, extraction)
        
        # Make sure lengths match
        if len(items) != len(values):
            # Try to extract items and values from other fields
            if isinstance(data, dict):
                # Look for possible items and values
                possible_items = []
                possible_values = []
                
                for key, value in data.items():
                    if isinstance(value, (int, float)):
                        possible_items.append(key)
                        possible_values.append(value)
                
                if possible_items and possible_values:
                    items = possible_items
                    values = possible_values
        
        # Check again
        if not items or not values or len(items) != len(values):
            return self._visualize_generic(data, description, extraction)
        
        # Create DataFrame
        df = pd.DataFrame({
            'Item': items,
            'Value': values
        })
        
        # Sort by value for better visualization
        df = df.sort_values('Value')
        
        # Format the title based on metric
        title = f'{metric} Comparison' if metric else 'Comparison'
        
        # Create bar chart
        if len(items) <= 2:
            # Horizontal orientation for 2 items
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                y=df['Item'],
                x=df['Value'],
                orientation='h',
                marker=dict(
                    color=df['Value'],
                    colorscale='Viridis'
                ),
                text=[f"{val} {unit}" for val in df['Value']],
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Value: %{x:,.2f} %{text}<extra></extra>',
                texttemplate='%{x:,.2f}'
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title=f'Value ({unit})' if unit else 'Value',
                yaxis_title='',
                xaxis=dict(tickformat=',.2f'),
                showlegend=False
            )
            
        else:
            # Vertical orientation for more items
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=df['Item'],
                y=df['Value'],
                marker=dict(
                    color=df['Value'],
                    colorscale='Viridis'
                ),
                text=[f"{val} {unit}" for val in df['Value']],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Value: %{y:,.2f} %{text}<extra></extra>',
                texttemplate='%{y:,.2f}'
            ))
            
            fig.update_layout(
                title=title,
                yaxis_title=f'Value ({unit})' if unit else 'Value',
                xaxis_title='',
                yaxis=dict(tickformat=',.2f'),
                showlegend=False
            )
        
        # Save visualization
        filename = self._get_filename("comparison", extraction)
        fig_path = f"{self.output_dir}/{filename}.html"
        fig.write_html(fig_path)
        
        # Calculate statistics
        stats = {
            'min': float(min(values)),
            'max': float(max(values)),
            'range': float(max(values) - min(values)),
            'avg': float(np.mean(values)),
            'diff_percent': float(((max(values) - min(values)) / min(values) * 100) if min(values) != 0 else 0)
        }
        
        return {
            'type': 'comparison',
            'title': title,
            'description': description,
            'items': items,
            'values': values,
            'metric': metric,
            'unit': unit,
            'stats': stats,
            'file_path': fig_path,
            'filename': f"{filename}.html"
        }
    
    def _visualize_comparisons(self, data: List[Dict[str, Any]], description: str, extraction: Dict[str, Any]) -> Dict[str, Any]:
        """Visualize multiple comparisons"""
        # If it's a single comparison object, use the single comparison visualization
        if len(data) == 1:
            return self._visualize_comparison(data[0], description, extraction)
        
        # Create a figure with multiple subplots
        fig = go.Figure()
        
        # Extract and visualize each comparison
        valid_comparisons = []
        
        for i, comparison in enumerate(data):
            items = comparison.get('items', [])
            values = comparison.get('values', [])
            metric = comparison.get('metric', f'Comparison {i+1}')
            
            if items and values and len(items) == len(values):
                valid_comparisons.append({
                    'items': items,
                    'values': values,
                    'metric': metric
                })
        
        # If no valid comparisons, try generic visualization
        if not valid_comparisons:
            return self._visualize_generic(data, description, extraction)
        
        # If only one valid comparison, use single comparison visualization
        if len(valid_comparisons) == 1:
            return self._visualize_comparison(valid_comparisons[0], description, extraction)
        
        # Create a visualization that combines multiple comparisons
        # Use a grouped bar chart if comparing the same items
        all_items = [set(comp['items']) for comp in valid_comparisons]
        if all(items == all_items[0] for items in all_items):
            # Same items, create grouped bar chart
            items = list(all_items[0])
            
            for comparison in valid_comparisons:
                # Map values to the common item list
                item_to_value = {item: value for item, value in zip(comparison['items'], comparison['values'])}
                values = [item_to_value.get(item, 0) for item in items]
                
                fig.add_trace(go.Bar(
                    x=items,
                    y=values,
                    name=comparison['metric']
                ))
            
            fig.update_layout(
                title='Multiple Comparisons',
                xaxis_title='Item',
                yaxis_title='Value',
                barmode='group'
            )
        else:
            # Different items, create separate bar charts in a grid
            num_comparisons = len(valid_comparisons)
            rows = (num_comparisons + 1) // 2  # Calculate number of rows (2 columns)
            
            fig = make_subplots(rows=rows, cols=2, subplot_titles=[comp['metric'] for comp in valid_comparisons])
            
            for i, comparison in enumerate(valid_comparisons):
                row = i // 2 + 1
                col = i % 2 + 1
                
                fig.add_trace(
                    go.Bar(
                        x=comparison['items'],
                        y=comparison['values'],
                        showlegend=False
                    ),
                    row=row, col=col
                )
            
            fig.update_layout(
                title='Multiple Comparisons',
                height=300 * rows
            )
        
        # Save visualization
        filename = self._get_filename("multiple_comparisons", extraction)
        fig_path = f"{self.output_dir}/{filename}.html"
        fig.write_html(fig_path)
        
        return {
            'type': 'multiple_comparisons',
            'title': 'Multiple Comparisons',
            'description': description,
            'comparisons': valid_comparisons,
            'file_path': fig_path,
            'filename': f"{filename}.html"
        }
    
    def _visualize_generic(self, data: Any, description: str, extraction: Dict[str, Any]) -> Dict[str, Any]:
        """Generic visualization for any data format"""
        # Try to convert to DataFrame or appropriate format
        try:
            if isinstance(data, dict):
                # For simple dictionaries
                labels = list(data.keys())
                values = []
                
                for v in data.values():
                    if isinstance(v, (int, float)):
                        values.append(float(v))
                    elif isinstance(v, str):
                        # Try to extract a number
                        try:
                            values.append(float(re.search(r'(\d+(?:\.\d+)?)', v).group(1)))
                        except:
                            values.append(1)  # Default value
                    else:
                        values.append(1)  # Default value
                
                # Create pie chart
                fig = px.pie(
                    names=labels,
                    values=values,
                    title='Data Distribution',
                    color_discrete_sequence=self.color_palettes['categorical']
                )
                
                fig.update_traces(
                    textposition='inside',
                    textinfo='label+percent',
                    hoverinfo='label+percent+value'
                )
                
                # Save visualization
                filename = self._get_filename("generic_pie", extraction)
                fig_path = f"{self.output_dir}/{filename}.html"
                fig.write_html(fig_path)
                
                return {
                    'type': 'generic_pie',
                    'title': 'Data Distribution',
                    'description': description,
                    'file_path': fig_path,
                    'filename': f"{filename}.html"
                }
                
            elif isinstance(data, list):
                # For lists, try to visualize as a table or bar chart
                if len(data) > 0 and isinstance(data[0], dict):
                    # Convert to DataFrame
                    df = pd.DataFrame(data)
                    
                    # If we have numeric columns, create a bar chart
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    
                    if len(numeric_cols) > 0 and len(df.columns) > 1:
                        non_numeric_cols = [col for col in df.columns if col not in numeric_cols]
                        
                        if non_numeric_cols:
                            category_col = non_numeric_cols[0]
                            value_col = numeric_cols[0]
                            
                            # Limit to top 10 categories by value
                            top_df = df.sort_values(value_col, ascending=False).head(10)
                            
                            fig = px.bar(
                                top_df,
                                x=category_col,
                                y=value_col,
                                title=f'Top {category_col} by {value_col}',
                                color=value_col,
                                color_continuous_scale=self.color_palettes['sequential']
                            )
                            
                            # Save visualization
                            filename = self._get_filename("generic_bar", extraction)
                            fig_path = f"{self.output_dir}/{filename}.html"
                            fig.write_html(fig_path)
                            
                            return {
                                'type': 'generic_bar',
                                'title': f'Top {category_col} by {value_col}',
                                'description': description,
                                'file_path': fig_path,
                                'filename': f"{filename}.html"
                            }
                
                # Default to saving as JSON
                filename = self._get_filename("generic_data", extraction)
                json_path = f"{self.output_dir}/{filename}.json"
                
                with open(json_path, 'w') as f:
                    json.dump(data, f, indent=2)
                
                return {
                    'type': 'generic_data',
                    'title': 'Generic Data',
                    'description': description,
                    'file_path': json_path,
                    'filename': f"{filename}.json"
                }
                
        except Exception as e:
            print(f"Error in generic visualization: {str(e)}")
        
        # Fallback - save as JSON
        filename = self._get_filename("raw_data", extraction)
        json_path = f"{self.output_dir}/{filename}.json"
        
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return {
            'type': 'raw_data',
            'title': 'Raw Data',
            'description': description,
            'file_path': json_path,
            'filename': f"{filename}.json"
        }
    
    def generate_visualizations(self, extractions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate visualizations for a list of extractions
        
        Args:
            extractions: List of data extraction objects
            
        Returns:
            List of visualization results
        """
        results = []
        
        for extraction in extractions:
            print(f"Processing visualization for {extraction.get('data_type', 'unknown')} data...")
            visualization = self.visualize_extraction(extraction)
            results.append(visualization)
        
        return results