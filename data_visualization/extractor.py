# data_visualization/extractor.py

import re
import json
import pandas as pd
import numpy as np
import ast
from typing import Dict, List, Any, Union
from openai import OpenAI
from datetime import datetime
import uuid

class DataExtractor:
    """Extracts structured data from document chunks with improved pattern recognition"""
    
    def __init__(self, openai_api_key=None):
        self.openai_api_key = openai_api_key
        if openai_api_key:
            self.openai_client = OpenAI(api_key=openai_api_key)
    
    def extract_numerical_data(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract numerical data from document chunks with enhanced pattern recognition
        
        Args:
            chunks: List of document chunks containing text data
            
        Returns:
            List of structured data extractions with metadata
        """
        extracted_data = []
        
        # Process each chunk
        for chunk in chunks:
            text = chunk.get('text', '')
            metadata = chunk.get('metadata', {})
            
            # Check if the chunk already has tabular data
            if 'table_data' in chunk:
                # Use the existing table data if available
                try:
                    if isinstance(chunk['table_data'], str):
                        table_data = json.loads(chunk['table_data'])
                    else:
                        table_data = chunk['table_data']
                    
                    # Try to identify what kind of table this is
                    table_type = self._classify_table_data(table_data)
                    
                    extraction = {
                        'data': table_data,
                        'data_type': 'table',
                        'table_type': table_type,
                        'source': metadata,
                        'description': f"{table_type} table from {metadata.get('filename', 'unknown file')}"
                    }
                    extracted_data.append(extraction)
                except Exception as e:
                    print(f"Error processing table data: {str(e)}")
            
            # Extract time series data (enhanced pattern)
            time_series = self._extract_time_series(text, metadata)
            if time_series:
                extracted_data.append(time_series)
            
            # Extract financial figures (enhanced)
            financial = self._extract_financial_figures(text, metadata)
            if financial:
                extracted_data.append(financial)
            
            # Extract percentages with context
            percentages = self._extract_percentages_with_context(text, metadata)
            if percentages:
                extracted_data.append(percentages)
            
            # Extract comparison data
            comparisons = self._extract_comparisons(text, metadata)
            if comparisons:
                extracted_data.extend(comparisons)
        
        return extracted_data
    
    def _classify_table_data(self, table_data: List[Dict]) -> str:
        """Identify the type of table data"""
        if not table_data or not isinstance(table_data, list) or not table_data[0]:
            return "Generic"
            
        # Get column names
        columns = list(table_data[0].keys())
        
        # Look for time-related columns
        time_columns = [col for col in columns if any(term in col.lower() for term in 
                       ['date', 'year', 'month', 'quarter', 'period', 'time', 'day', 'week'])]
        
        # Look for financial columns
        financial_columns = [col for col in columns if any(term in col.lower() for term in 
                           ['revenue', 'sales', 'profit', 'income', 'expense', 'cost', 
                            'margin', 'price', 'amount', 'budget', 'forecast'])]
        
        # Look for comparison columns
        comparison_columns = [col for col in columns if any(term in col.lower() for term in 
                             ['vs', 'versus', 'comparison', 'diff', 'change', 'growth', 'actual', 'target'])]
        
        # Look for percentage columns
        percentage_columns = [col for col in columns if any(term in col.lower() for term in 
                             ['percent', 'percentage', '%', 'rate', 'ratio'])]
        
        # Determine table type based on column patterns
        if time_columns and financial_columns:
            return "Financial Time Series"
        elif time_columns:
            return "Time Series"
        elif financial_columns:
            return "Financial"
        elif comparison_columns:
            return "Comparison"
        elif percentage_columns:
            return "Percentage Analysis"
        else:
            # Generic classification based on what's available
            return "Generic Data"
    
    def _extract_time_series(self, text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract time series data with enhanced pattern recognition"""
        # Look for patterns like "In Q1 2023, revenue was $1.2M"
        # or "January: $5.2M", or "Revenue in 2022: $10M"
        time_periods = []
        values = []
        metrics = []
        
        # Match different date formats and associated values
        time_pattern = r'(?:in|during|for)?\s*(?:Q[1-4]|January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|20\d{2})'
        time_matches = re.finditer(time_pattern, text, re.IGNORECASE)
        
        for match in time_matches:
            period = match.group(0).strip()
            
            # Find context around the date (up to 100 chars after)
            context_end = min(match.end() + 100, len(text))
            context = text[match.start():context_end]
            
            # Try to extract a metric name
            metric_match = re.search(r'(\w+(?:\s+\w+)?)', context)
            metric = "Value" if not metric_match else metric_match.group(1)
            
            # Try to extract a numeric value
            value_match = re.search(r'(?:USD|\$|€|£)?(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:USD|\$|€|£|[Mm]illion|[Bb]illion|[Tt]housand|[Kk])?', context)
            
            if value_match:
                # Clean the value
                value_str = value_match.group(1).replace(',', '')
                
                try:
                    value = float(value_str)
                    
                    # Apply scale if mentioned
                    if re.search(r'[Mm]illion', context):
                        value *= 1000000
                    elif re.search(r'[Bb]illion', context):
                        value *= 1000000000
                    elif re.search(r'[Tt]housand|[Kk]', context):
                        value *= 1000
                    
                    time_periods.append(period)
                    values.append(value)
                    metrics.append(metric)
                except ValueError:
                    pass
        
        if time_periods and values:
            # Convert to structured data with matching periods and values
            time_series_data = []
            for i in range(len(time_periods)):
                time_series_data.append({
                    'period': time_periods[i],
                    'value': values[i],
                    'metric': metrics[i] if i < len(metrics) else "Value"
                })
            
            # Group by metrics
            metric_groups = {}
            for item in time_series_data:
                metric = item['metric']
                if metric not in metric_groups:
                    metric_groups[metric] = []
                metric_groups[metric].append(item)
            
            # Create a data extraction for each metric
            if len(metric_groups) > 1:
                # Multiple metrics found, create a multi-series time series
                return {
                    'data': time_series_data,
                    'data_type': 'multi_time_series',
                    'source': metadata,
                    'metrics': list(metric_groups.keys()),
                    'description': f"Multi-metric time series data from {metadata.get('filename', 'unknown file')}"
                }
            else:
                # Single metric, create a simple time series
                return {
                    'data': time_series_data,
                    'data_type': 'time_series',
                    'source': metadata,
                    'description': f"Time series data from {metadata.get('filename', 'unknown file')}"
                }
        
        return None
    
    def _extract_financial_figures(self, text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract financial figures with context"""
        # Look for currency and large numbers with context
        financial_pattern = r'(?:(?:USD|\$|€|£)\s*(\d+(?:,\d{3})*(?:\.\d+)?)|(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:USD|\$|€|£))(?:\s*(?:[Mm]illion|[Bb]illion|[Tt]housand|[Kk]))?'
        financial_matches = re.finditer(financial_pattern, text)
        
        financial_data = []
        labels = []
        
        for match in financial_matches:
            # Get full match
            match_text = match.group(0)
            
            # Find context before the figure (up to 50 chars)
            context_start = max(0, match.start() - 50)
            prefix_context = text[context_start:match.start()]
            
            # Try to extract a label for this figure
            label_match = re.search(r'(\w+(?:\s+\w+){0,3}):?\s*$', prefix_context)
            label = label_match.group(1) if label_match else "Value"
            
            # Extract and clean the numeric value
            value_str = re.search(r'(\d+(?:,\d{3})*(?:\.\d+)?)', match_text).group(1).replace(',', '')
            
            try:
                value = float(value_str)
                
                # Apply scale if mentioned
                if re.search(r'[Mm]illion', match_text):
                    value *= 1000000
                elif re.search(r'[Bb]illion', match_text):
                    value *= 1000000000
                elif re.search(r'[Tt]housand|[Kk]', match_text):
                    value *= 1000
                
                financial_data.append(value)
                labels.append(label)
            except ValueError:
                pass
        
        if financial_data:
            # Create labeled financial data
            labeled_data = []
            for i in range(len(financial_data)):
                labeled_data.append({
                    'label': labels[i],
                    'value': financial_data[i]
                })
            
            return {
                'data': labeled_data,
                'data_type': 'financial_figures',
                'source': metadata,
                'description': f"Financial figures from {metadata.get('filename', 'unknown file')}"
            }
        
        return None
    
    def _extract_percentages_with_context(self, text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract percentages with their labels/context"""
        percentage_pattern = r'(\d+(?:\.\d+)?)\s*%'
        percentage_matches = re.finditer(percentage_pattern, text)
        
        percentage_data = []
        labels = []
        
        for match in percentage_matches:
            # Extract percentage value
            percentage_value = float(match.group(1))
            
            # Find context before the percentage (up to 50 chars)
            context_start = max(0, match.start() - 50)
            prefix_context = text[context_start:match.start()]
            
            # Try to extract a label for this percentage
            label_match = re.search(r'(\w+(?:\s+\w+){0,3}):?\s*$', prefix_context)
            label = label_match.group(1) if label_match else "Percentage"
            
            percentage_data.append(percentage_value)
            labels.append(label)
        
        if percentage_data:
            # Create labeled percentage data
            labeled_data = []
            for i in range(len(percentage_data)):
                labeled_data.append({
                    'label': labels[i],
                    'value': percentage_data[i]
                })
            
            return {
                'data': labeled_data,
                'data_type': 'percentages',
                'source': metadata,
                'description': f"Percentage values from {metadata.get('filename', 'unknown file')}"
            }
        
        return None
    
    def _extract_comparisons(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract comparison data between entities"""
        # Look for comparison patterns like "X is higher than Y" or "X vs Y"
        comparison_patterns = [
            r'(\w+(?:\s+\w+){0,3})\s+(?:is|was)\s+(?:higher|lower|more|less|better|worse|greater|smaller)\s+than\s+(\w+(?:\s+\w+){0,3})',
            r'(\w+(?:\s+\w+){0,3})\s+vs\.?\s+(\w+(?:\s+\w+){0,3})',
            r'(?:comparing|comparison\s+(?:of|between))\s+(\w+(?:\s+\w+){0,3})\s+(?:and|&|to)\s+(\w+(?:\s+\w+){0,3})',
            r'(\w+(?:\s+\w+){0,3})(?::|,)?\s+(\d+(?:\.\d+)?)[^\d]+(\w+(?:\s+\w+){0,3})(?::|,)?\s+(\d+(?:\.\d+)?)'
        ]
        
        extractions = []
        
        for pattern in comparison_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                # Different patterns require different handling
                if len(match.groups()) == 2:
                    # Simple entity comparison without values
                    item1 = match.group(1).strip()
                    item2 = match.group(2).strip()
                    
                    # Look for values near these items
                    context = text[max(0, match.start() - 100):min(len(text), match.end() + 100)]
                    value_pattern = r'(\d+(?:\.\d+)?)'
                    value_matches = list(re.finditer(value_pattern, context))
                    
                    # Take only the first two values found
                    values = []
                    if len(value_matches) >= 2:
                        values = [float(value_matches[0].group(0)), float(value_matches[1].group(0))]
                    
                    comparison_data = {
                        'data': {
                            'items': [item1, item2],
                            'values': values if values else None,
                            'metric': 'Value',
                            'unit': ''
                        },
                        'data_type': 'comparison',
                        'source': metadata,
                        'description': f"Comparison between {item1} and {item2} from {metadata.get('filename', 'unknown file')}"
                    }
                    extractions.append(comparison_data)
                    
                elif len(match.groups()) == 4:
                    # Pattern with embedded values
                    item1 = match.group(1).strip()
                    value1 = float(match.group(2))
                    item2 = match.group(3).strip()
                    value2 = float(match.group(4))
                    
                    comparison_data = {
                        'data': {
                            'items': [item1, item2],
                            'values': [value1, value2],
                            'metric': 'Value',
                            'unit': ''
                        },
                        'data_type': 'comparison',
                        'source': metadata,
                        'description': f"Comparison between {item1} ({value1}) and {item2} ({value2}) from {metadata.get('filename', 'unknown file')}"
                    }
                    extractions.append(comparison_data)
        
        return extractions
    
    def extract_with_llm(self, chunks: List[Dict[str, Any]], extraction_type: str = "auto") -> List[Dict[str, Any]]:
        """
        Use LLM to extract structured data with improved categorization
        
        Args:
            chunks: List of document chunks
            extraction_type: Type of data to extract ("auto", "financials", "metrics", "trends", "comparisons")
            
        Returns:
            List of structured data extractions
        """
        if not self.openai_api_key:
            return [{"error": "OpenAI API key required for LLM extraction"}]
        
        extracted_data = []
        
        # If extraction_type is "auto", use the LLM to determine the best type for each chunk
        if extraction_type == "auto":
            for chunk in chunks:
                text = chunk.get('text', '')
                metadata = chunk.get('metadata', {})
                
                # Skip short texts
                if len(text) < 100:
                    continue
                
                try:
                    # First, determine what type of data is in this chunk
                    data_type = self._determine_data_type(text)
                    
                    if data_type:
                        # Now extract that specific type
                        extractions = self._extract_specific_type(text, data_type, metadata)
                        if extractions:
                            extracted_data.extend(extractions)
                    
                except Exception as e:
                    print(f"Error in auto-extraction with LLM: {str(e)}")
        else:
            # Extract the specified type from all chunks
            for chunk in chunks:
                text = chunk.get('text', '')
                metadata = chunk.get('metadata', {})
                
                # Skip short texts
                if len(text) < 100:
                    continue
                
                try:
                    extractions = self._extract_specific_type(text, extraction_type, metadata)
                    if extractions:
                        extracted_data.extend(extractions)
                        
                except Exception as e:
                    print(f"Error extracting {extraction_type} with LLM: {str(e)}")
        
        return extracted_data
    
    def _determine_data_type(self, text: str) -> str:
        """Use LLM to determine what type of data is present in the text"""
        prompt = f"""Analyze the following text and determine what type of structured data it contains.
Choose from: financials, metrics, trends, comparisons, or none.
Only respond with one of these five categories, nothing else.

TEXT:
{text[:3000]}  # Limit text length for token efficiency
"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are a data analysis assistant that identifies data types in text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=10
            )
            
            result = response.choices[0].message.content.strip().lower()
            
            # Map to valid extraction types
            if "financial" in result:
                return "financials"
            elif "metric" in result:
                return "metrics"
            elif "trend" in result:
                return "trends"
            elif "comparison" in result:
                return "comparisons"
            else:
                return None
            
        except Exception as e:
            print(f"Error determining data type: {str(e)}")
            return None
    
    def _extract_specific_type(self, text: str, data_type: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract a specific data type from text"""
        prompt = self._generate_extraction_prompt(text, data_type)
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are a data extraction assistant that extracts structured data from text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1000
            )
            
            # Get the generated text
            result_text = response.choices[0].message.content
            
            # Extract the JSON part (between ```)
            json_match = re.search(r'```(?:json)?\n(.*?)\n```', result_text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find any JSON-like structure
                json_match = re.search(r'(\{.*\}|\[.*\])', result_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = result_text
            
            # Clean the string and parse the JSON
            try:
                extracted = json.loads(json_str)
                
                # Skip empty results
                if not extracted or (isinstance(extracted, list) and len(extracted) == 0) or (isinstance(extracted, dict) and len(extracted) == 0):
                    return []
                
                # Generate a suitable description based on the data type
                if data_type == "financials":
                    description = f"Financial data from {metadata.get('filename', 'unknown file')}"
                elif data_type == "metrics":
                    description = f"Performance metrics from {metadata.get('filename', 'unknown file')}"
                elif data_type == "trends":
                    description = f"Trend analysis from {metadata.get('filename', 'unknown file')}"
                elif data_type == "comparisons":
                    description = f"Comparative data from {metadata.get('filename', 'unknown file')}"
                else:
                    description = f"Structured data from {metadata.get('filename', 'unknown file')}"
                
                # Add metadata
                extraction = {
                    'data': extracted,
                    'data_type': data_type,
                    'source': metadata,
                    'description': description,
                    'id': str(uuid.uuid4())[:8]  # Add a unique ID for reference
                }
                
                return [extraction]
                
            except json.JSONDecodeError:
                # Try to use ast.literal_eval for simpler structures
                try:
                    extracted = ast.literal_eval(json_str)
                    
                    # Skip empty results
                    if not extracted:
                        return []
                    
                    # Add metadata
                    extraction = {
                        'data': extracted,
                        'data_type': data_type,
                        'source': metadata,
                        'description': f"{data_type.title()} data from {metadata.get('filename', 'unknown file')}",
                        'id': str(uuid.uuid4())[:8]
                    }
                    
                    return [extraction]
                except:
                    print(f"Failed to parse LLM output as JSON or literal: {json_str[:100]}...")
                    return []
            
        except Exception as e:
            print(f"Error extracting {data_type} with LLM: {str(e)}")
            return []
    
    def _generate_extraction_prompt(self, text: str, extraction_type: str) -> str:
        """Generate an improved prompt for LLM-based data extraction"""
        base_prompt = f"""Extract {extraction_type} data from the following text. Format your response as a valid JSON object or array.

TEXT:
{text[:4000]}  # Limit text length for token efficiency

"""
        
        if extraction_type == "financials":
            base_prompt += """
Extract all financial information mentioned in the text. Include revenues, costs, profits, margins, growth rates, etc.

If there are multiple time periods, format the data as an array of objects with this structure:
[
  {
    "period": "Q1 2023",  // Time period
    "revenue": 1200000,   // Revenue value in full numbers (not abbreviated)
    "costs": 800000,      // Costs value
    "profit": 400000,     // Profit value
    "margin": 33.3        // Margin as percentage
    // Include any other financial metrics found
  },
  // Additional periods...
]

If there's no time series, but just various financial metrics, use this format:
{
  "revenue": 5000000,
  "costs": 3000000,
  "profit": 2000000,
  "margin": 40,
  // Other metrics as appropriate
}

Be sure to standardize all values (e.g., convert "5.2M" to 5200000) and include units where needed.
"""
        elif extraction_type == "metrics":
            base_prompt += """
Extract all metrics and KPIs mentioned in the text. Include any quantitative measurements, statistics, or performance indicators.

Format your answer as an array of metric objects, each containing:
[
  {
    "metric": "Customer Satisfaction",  // Name of the metric
    "value": 4.8,                       // Numeric value
    "unit": "out of 5",                 // Unit of measurement
    "period": "Q2 2023"                 // Time period (if available)
  },
  // Additional metrics...
]

If there are multiple time periods for the same metrics, use this format instead:
{
  "metric": "Customer Satisfaction",
  "periods": ["Q1 2023", "Q2 2023", "Q3 2023"],
  "values": [4.7, 4.8, 4.9],
  "unit": "out of 5"
}

Be precise about units and time periods. Standardize all values to their numeric form.
"""
        elif extraction_type == "trends":
            base_prompt += """
Extract information about trends mentioned in the text. Include growth patterns, declines, or changes over time.

Format your answer as an array of trend objects, each containing:
[
  {
    "metric": "Monthly Active Users",   // The metric showing a trend
    "direction": "increasing",          // "increasing", "decreasing", or "stable"
    "magnitude": 12.5,                  // Percentage or absolute change value
    "unit": "percent",                  // Unit of the magnitude (percent, absolute, etc.)
    "period": "2022-2023",              // Time period over which the trend occurs
    "start_value": 10000,               // Starting value (if available)
    "end_value": 11250                  // Ending value (if available)
  },
  // Additional trends...
]

Be specific about the direction and magnitude. Provide start and end values when available.
"""
        elif extraction_type == "comparisons":
            base_prompt += """
Extract comparative data mentioned in the text. Look for comparisons between items, time periods, groups, etc.

Format your answer as an array of comparison objects, each containing:
[
  {
    "compared_items": ["Product A", "Product B"],  // Items being compared
    "metric": "Annual Sales",                     // What is being compared
    "values": [125000, 150000],                   // The corresponding values
    "unit": "units",                              // Unit of measurement
    "difference": 25000,                          // Absolute difference
    "percent_difference": 20                      // Percentage difference
  },
  // Additional comparisons...
]

For each comparison, include the compared items, the metric, and their values. Calculate the difference and percentage difference when possible.
"""
        else:
            base_prompt += """
Extract any structured data you can find in the text. Identify patterns, numbers, categories, or relationships.

Format your answer as an appropriate JSON structure (object or array) that best captures the data found in the text.
"""
        
        base_prompt += """
IMPORTANT: Return ONLY the JSON in the following format with no additional explanation:

```json
{
    "key": "value"
}
```

If you can't extract any relevant data, return an empty array.
"""
        
        return base_prompt