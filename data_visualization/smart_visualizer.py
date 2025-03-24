# data_visualization/smart_visualizer.py

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import re
from typing import Dict, List, Any, Union
from datetime import datetime
from openai import OpenAI

# Set Matplotlib style for better aesthetics
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12

class SmartVisualizer:
    """Creates intelligent, LLM-driven visualizations from document data"""
    
    def __init__(self, config: Dict, output_dir: str = "./outputs/visualizations"):
        self.output_dir = output_dir
        self.config = config
        self.openai_client = OpenAI(api_key=config.get('openai', {}).get('api_key', ''))
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Color schemes for different chart types
        self.color_schemes = {
            'categorical': px.colors.qualitative.Bold,
            'sequential': px.colors.sequential.Viridis,
            'diverging': px.colors.diverging.RdBu,
            'matplotlib': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        }
    
    def analyze_document_data(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Use LLM to analyze document data and suggest visualizations
        
        Args:
            chunks: Document chunks from vector search
            
        Returns:
            Analysis with visualization suggestions
        """
        # Extract text from chunks
        all_text = "\n\n".join([chunk.get('text', '') for chunk in chunks])
        
        # If text is too long, limit to 15,000 characters
        if len(all_text) > 15000:
            all_text = all_text[:15000] + "...[truncated]"
        
        # Prompt for the LLM
        prompt = f"""Analyze the following document text and identify the best data visualization opportunities.
For each visualization you recommend, suggest:
1. Chart type (bar, line, pie, scatter, heatmap, etc.)
2. What data should be visualized
3. Which visualization library to use (Matplotlib or Plotly)
4. A detailed title and description for the chart

TEXT FROM DOCUMENTS:
{all_text}

Provide your recommendations in JSON format as follows:
```json
{{
  "visualizations": [
    {{
      "chart_type": "bar",
      "title": "Revenue by Quarter 2023",
      "description": "Comparison of quarterly revenue for fiscal year 2023",
      "data_points": [
        {{ "label": "Q1 2023", "value": 1500000 }},
        {{ "label": "Q2 2023", "value": 1750000 }},
        {{ "label": "Q3 2023", "value": 1900000 }},
        {{ "label": "Q4 2023", "value": 2150000 }}
      ],
      "x_axis_label": "Quarter",
      "y_axis_label": "Revenue (USD)",
      "library": "plotly",
      "color_scheme": "sequential"
    }}
  ]
}}
```

IMPORTANT:
- Recommend only 2-3 high-value visualizations
- Focus on numerical data that would benefit from visualization
- If the text doesn't contain appropriate data for visualization, suggest at most 1 chart
- For time series, ensure data points are in chronological order
- Make sure all values are reasonable based on the context
- Use the specific data from the text, not made-up values
"""
        
        try:
            # Call LLM for analysis
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are a data visualization expert who can identify the best visualization opportunities in document text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            # Extract JSON from response
            response_text = response.choices[0].message.content
            json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to extract any JSON-like structure
                json_match = re.search(r'(\{[\s\S]*\})', response_text)
                json_str = json_match.group(1) if json_match else "{}"
            
            # Parse visualization suggestions
            try:
                analysis = json.loads(json_str)
                return analysis
            except json.JSONDecodeError:
                print(f"Error parsing JSON response: {json_str[:100]}...")
                return {"visualizations": []}
            
        except Exception as e:
            print(f"Error analyzing document data: {str(e)}")
            return {"visualizations": []}
    
    def create_visualizations(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create visualizations based on LLM analysis
        
        Args:
            analysis: Analysis with visualization suggestions
            
        Returns:
            List of created visualizations
        """
        visualizations = []
        
        for viz_data in analysis.get('visualizations', []):
            try:
                # Extract visualization parameters
                chart_type = viz_data.get('chart_type', '').lower()
                title = viz_data.get('title', 'Untitled Chart')
                description = viz_data.get('description', '')
                data_points = viz_data.get('data_points', [])
                x_axis_label = viz_data.get('x_axis_label', '')
                y_axis_label = viz_data.get('y_axis_label', '')
                library = viz_data.get('library', 'plotly').lower()
                color_scheme = viz_data.get('color_scheme', 'categorical')
                
                # Skip if no data points
                if not data_points:
                    continue
                
                # Create a DataFrame from the data points
                if isinstance(data_points, list) and all(isinstance(item, dict) for item in data_points):
                    df = pd.DataFrame(data_points)
                else:
                    continue
                
                # Create the visualization
                if library == 'matplotlib':
                    viz_result = self._create_matplotlib_chart(
                        df, chart_type, title, description, x_axis_label, y_axis_label, color_scheme
                    )
                else:  # Default to Plotly
                    viz_result = self._create_plotly_chart(
                        df, chart_type, title, description, x_axis_label, y_axis_label, color_scheme
                    )
                
                if viz_result:
                    visualizations.append(viz_result)
                
            except Exception as e:
                print(f"Error creating visualization: {str(e)}")
                continue
        
        return visualizations
    
    def _create_plotly_chart(self, 
                           df: pd.DataFrame, 
                           chart_type: str, 
                           title: str, 
                           description: str, 
                           x_axis_label: str,
                           y_axis_label: str, 
                           color_scheme: str) -> Dict[str, Any]:
        """Create a Plotly chart"""
        # Generate a unique filename
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"plotly_{chart_type}_{timestamp}"
        file_path = f"{self.output_dir}/{filename}.html"
        
        # Figure placeholder
        fig = None
        
        # Select color palette
        colors = self.color_schemes.get(color_scheme, self.color_schemes['categorical'])
        
        # Extract key columns for charting
        label_col = df.columns[0] if 'label' not in df else 'label'
        value_col = df.columns[1] if 'value' not in df else 'value'
        
        # Create the appropriate chart type
        if chart_type == 'bar':
            fig = px.bar(
                df, 
                x=label_col, 
                y=value_col,
                title=title,
                labels={label_col: x_axis_label or label_col, value_col: y_axis_label or value_col},
                color_discrete_sequence=colors
            )
            
        elif chart_type == 'line':
            fig = px.line(
                df, 
                x=label_col, 
                y=value_col,
                title=title,
                labels={label_col: x_axis_label or label_col, value_col: y_axis_label or value_col},
                color_discrete_sequence=colors,
                markers=True
            )
            
        elif chart_type == 'pie':
            fig = px.pie(
                df, 
                names=label_col, 
                values=value_col,
                title=title,
                color_discrete_sequence=colors
            )
            
        elif chart_type == 'scatter':
            fig = px.scatter(
                df, 
                x=label_col, 
                y=value_col,
                title=title,
                labels={label_col: x_axis_label or label_col, value_col: y_axis_label or value_col},
                color_discrete_sequence=colors
            )
            
        elif chart_type == 'area':
            fig = px.area(
                df, 
                x=label_col, 
                y=value_col,
                title=title,
                labels={label_col: x_axis_label or label_col, value_col: y_axis_label or value_col},
                color_discrete_sequence=colors
            )
            
        else:  # Default to bar chart
            fig = px.bar(
                df, 
                x=label_col, 
                y=value_col,
                title=title,
                labels={label_col: x_axis_label or label_col, value_col: y_axis_label or value_col},
                color_discrete_sequence=colors
            )
        
        # Enhance the figure
        fig.update_layout(
            title={
                'text': title,
                'font': {'size': 20, 'family': 'Arial, sans-serif', 'color': '#333333'},
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title=x_axis_label or label_col,
            yaxis_title=y_axis_label or value_col,
            legend_title_text='Legend',
            font=dict(
                family="Arial, sans-serif",
                size=14,
                color="#333333"
            ),
            plot_bgcolor='white',
            hoverlabel=dict(
                bgcolor="white",
                font_size=14,
                font_family="Arial, sans-serif"
            )
        )
        
        # Show all labels
        if chart_type in ['bar', 'line', 'scatter']:
            fig.update_xaxes(tickangle=-45 if len(df) > 5 else 0)
        
        # Add grid lines
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        # Save the figure
        fig.write_html(file_path)
        
        return {
            'type': f'plotly_{chart_type}',
            'title': title,
            'description': description,
            'library': 'plotly',
            'file_path': file_path,
            'filename': f"{filename}.html",
            'data': df.to_dict(orient='records')
        }
    
    def _create_matplotlib_chart(self, 
                              df: pd.DataFrame, 
                              chart_type: str, 
                              title: str, 
                              description: str,
                              x_axis_label: str, 
                              y_axis_label: str, 
                              color_scheme: str) -> Dict[str, Any]:
        """Create a Matplotlib chart"""
        # Generate a unique filename
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"matplotlib_{chart_type}_{timestamp}"
        file_path = f"{self.output_dir}/{filename}.html"
        
        # Create a new figure with high resolution
        plt.figure(figsize=(12, 8), dpi=100)
        
        # Extract key columns for charting
        label_col = df.columns[0] if 'label' not in df else 'label'
        value_col = df.columns[1] if 'value' not in df else 'value'
        
        # Select color palette
        colors = self.color_schemes['matplotlib']
        
        # Create the appropriate chart type
        if chart_type == 'bar':
            ax = df.plot(
                kind='bar',
                x=label_col,
                y=value_col,
                color=colors,
                legend=False
            )
            plt.xticks(rotation=45 if len(df) > 5 else 0)
            
        elif chart_type == 'line':
            ax = df.plot(
                kind='line',
                x=label_col,
                y=value_col,
                color=colors[0],
                marker='o',
                legend=False
            )
            
        elif chart_type == 'pie':
            ax = df.plot(
                kind='pie',
                y=value_col,
                labels=df[label_col],
                colors=colors,
                autopct='%1.1f%%',
                legend=False
            )
            ax.set_ylabel('')
            
        elif chart_type == 'scatter':
            ax = df.plot(
                kind='scatter',
                x=label_col,
                y=value_col,
                color=colors[0],
                legend=False
            )
            
        else:  # Default to bar chart
            ax = df.plot(
                kind='bar',
                x=label_col,
                y=value_col,
                color=colors,
                legend=False
            )
        
        # Set title and labels
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.xlabel(x_axis_label or label_col, fontsize=14, labelpad=10)
        plt.ylabel(y_axis_label or value_col, fontsize=14, labelpad=10)
        
        # Add a grid for better readability
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Tight layout
        plt.tight_layout()
        
        # Add value labels on bars for bar charts
        if chart_type == 'bar':
            for i, v in enumerate(df[value_col]):
                ax.text(i, v + 0.1, str(round(v, 2)), ha='center')
        
        # Save as HTML
        import mpld3
        html_content = mpld3.fig_to_html(plt.gcf())
        with open(file_path, 'w') as f:
            f.write(html_content)
        
        # Close the figure to free memory
        plt.close()
        
        return {
            'type': f'matplotlib_{chart_type}',
            'title': title,
            'description': description,
            'library': 'matplotlib',
            'file_path': file_path,
            'filename': f"{filename}.html",
            'data': df.to_dict(orient='records')
        }
    
    def generate_smart_visualizations(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate intelligent visualizations from document chunks
        
        Args:
            chunks: Document chunks from vector search
            
        Returns:
            List of visualization results
        """
        # Analyze document data to identify visualization opportunities
        analysis = self.analyze_document_data(chunks)
        
        # Create visualizations based on analysis
        visualizations = self.create_visualizations(analysis)
        
        # Save visualization index
        visualization_index = []
        for viz in visualizations:
            index_entry = {
                'title': viz.get('title', ''),
                'type': viz.get('type', ''),
                'description': viz.get('description', ''),
                'filename': viz.get('filename', '')
            }
            visualization_index.append(index_entry)
        
        with open(f"{self.output_dir}/index.json", 'w') as f:
            json.dump(visualization_index, f, indent=2)
        
        return visualizations