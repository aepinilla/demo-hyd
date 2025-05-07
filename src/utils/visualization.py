"""
Data visualization utilities for Streamlit and Seaborn integration.

This module provides functions for creating common data visualizations 
with Seaborn and displaying them in Streamlit.
"""

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Any

# Store the loaded dataset in session state
if "dataset" not in st.session_state:
    st.session_state.dataset = None

def load_dataset(file_path: str) -> str:
    """
    Load a dataset from a CSV file and store it in session state.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        str: Information about the loaded dataset
    """
    try:
        # Clean the file path
        if isinstance(file_path, str):
            file_path = clean_column_name(file_path)
        
        # Special case for example datasets
        if file_path.lower() == "example" or file_path.lower() == "sample":
            import os
            example_path = os.path.join("examples", "sample_data.csv")
            if os.path.exists(example_path):
                file_path = example_path
            else:
                return "Error: Example dataset not found. Please upload your own CSV file."
        
        if not file_path.endswith('.csv'):
            return "Error: Only CSV files are supported for visualization"
            
        # Load the dataset
        df = pd.read_csv(file_path)
        
        # Store in session state
        st.session_state.dataset = df
        
        # Return information about the dataset
        return f"""
## Dataset Loaded Successfully

**File:** {file_path}
**Shape:** {df.shape[0]} rows × {df.shape[1]} columns
**Columns:** {', '.join(df.columns.tolist())}

**Column Types:**
```
{df.dtypes.to_string()}
```

**Sample Data:**
```
{df.head(5).to_string()}
```
        """
    except Exception as e:
        return f"Error loading dataset: {str(e)}"

def plot_histogram(column: str, bins: int = 10, title: str = "Histogram") -> str:
    """
    Create a histogram using Seaborn and display it in Streamlit.
    
    Args:
        column (str): Column to plot
        bins (int): Number of bins
        title (str): Plot title
        
    Returns:
        str: Result message
    """
    if st.session_state.dataset is None:
        return "Error: No dataset loaded. Please load a dataset first using load_dataset."
    
    try:
        df = st.session_state.dataset
        
        # Check if column contains a JSON structure with parameters
        if isinstance(column, str) and '{' in column and '}' in column:
            params = parse_json_input(column)
            if 'column' in params:
                column = params['column']
            if 'bins' in params:
                try:
                    bins = int(params['bins'])
                except:
                    pass
            if 'title' in params:
                title = params['title']
        
        # Clean the column name
        column = clean_column_name(column)
        
        if column not in df.columns:
            return f"Error: Column '{column}' not found in dataset. Available columns: {', '.join(df.columns)}"
        
        # Create a new figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create the histogram
        sns.histplot(data=df, x=column, bins=bins, kde=True, ax=ax)
        
        # Set title and labels
        ax.set_title(title)
        ax.set_xlabel(column)
        ax.set_ylabel("Count")
        
        # Display the plot in Streamlit
        st.pyplot(fig)
        
        # Add some statistics
        stats = df[column].describe()
        
        return f"""
## Histogram for '{column}'

**Statistics:**
- **Mean:** {stats['mean']:.2f}
- **Median:** {stats['50%']:.2f}
- **Std Dev:** {stats['std']:.2f}
- **Min:** {stats['min']:.2f}
- **Max:** {stats['max']:.2f}
        """
    except Exception as e:
        return f"Error creating histogram: {str(e)}"

def parse_json_input(input_str):
    """Parse a JSON string input and return a dictionary of parameters"""
    if not isinstance(input_str, str):
        return {}
        
    # Remove quotes, newlines, and extra spaces
    clean_input = input_str.strip('"\'\'\n\r ')
    
    # Check if the input contains JSON-like formatting
    if '{' in clean_input and '}' in clean_input:
        import json
        try:
            # Try to parse as JSON
            json_data = json.loads(clean_input)
            if isinstance(json_data, dict):
                return json_data
        except:
            # If JSON parsing fails, try a simpler extraction
            import re
            # Extract all key-value pairs
            pairs = re.findall(r'"([^"]+)"\s*:\s*"([^"]+)"', clean_input)
            if pairs:
                return {key: value for key, value in pairs}
    
    return {}

def clean_column_name(column):
    """Helper function to clean column names from various input formats"""
    if not isinstance(column, str):
        return column
        
    # Remove quotes, newlines, and extra spaces
    clean_column = column.strip('"\'\'\n\r ')
    
    # Check if the column name contains JSON-like formatting
    if '{' in clean_column and '}' in clean_column:
        # Try to parse as JSON and extract the column name
        params = parse_json_input(clean_column)
        
        # Look for common column name keys
        for key in ['column', 'x_column', 'y_column']:
            if key in params:
                return params[key]
                
        # If no specific column key found, return the first string value
        for value in params.values():
            if isinstance(value, str):
                return value
    
    return clean_column

def plot_scatter(x_column: str, y_column: str = "", hue_column: Optional[str] = None, 
                 title: str = "Scatter Plot") -> str:
    """
    Create a scatter plot using Seaborn and display it in Streamlit.
    
    Args:
        x_column (str): Column for x-axis or JSON with parameters
        y_column (str, optional): Column for y-axis. If empty, will auto-select
        hue_column (str, optional): Column for color grouping
        title (str): Plot title
        
    Returns:
        str: Result message
    """
    if st.session_state.dataset is None:
        return "Error: No dataset loaded. Please load a dataset first using load_dataset."
    
    try:
        df = st.session_state.dataset
        
        # Check if x_column contains a JSON structure with both x and y columns
        if isinstance(x_column, str) and '{' in x_column and '}' in x_column:
            params = parse_json_input(x_column)
            if 'x_column' in params:
                x_column = params['x_column']
            if 'y_column' in params:
                y_column = params['y_column']
            if 'hue_column' in params:
                hue_column = params['hue_column']
            if 'title' in params:
                title = params['title']
        
        # Clean column names
        x_column = clean_column_name(x_column)
        y_column = clean_column_name(y_column) if y_column else ""
        if hue_column is not None:
            hue_column = clean_column_name(hue_column)
            
        # Auto-select the first two numeric columns if needed
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_cols or len(numeric_cols) < 2:
            return "Error: The dataset needs at least two numeric columns for a scatter plot."
            
        # If x_column is empty or not found, use the first numeric column
        if not x_column or x_column not in df.columns:
            x_column = numeric_cols[0]
            
        # If y_column is empty or not found, use the second numeric column
        # (or the first if x_column is already using the first)
        if not y_column or y_column not in df.columns:
            if x_column == numeric_cols[0] and len(numeric_cols) > 1:
                y_column = numeric_cols[1]
            else:
                y_column = numeric_cols[0]
        
        # Validate columns (should be valid now due to auto-selection above)
        if x_column not in df.columns:
            return f"Error: Column '{x_column}' not found in dataset. Available columns: {', '.join(df.columns)}"
            
        if y_column not in df.columns:
            return f"Error: Column '{y_column}' not found in dataset. Available columns: {', '.join(df.columns)}"
        
        if hue_column is not None and hue_column not in df.columns:
            return f"Error: Hue column '{hue_column}' not found in dataset."
        
        # Create a new figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create the scatter plot
        sns.scatterplot(data=df, x=x_column, y=y_column, hue=hue_column, ax=ax)
        
        # Set title and labels
        ax.set_title(title)
        
        # Display the plot in Streamlit
        st.pyplot(fig)
        
        # Calculate correlation
        correlation = df[[x_column, y_column]].corr().iloc[0, 1]
        
        return f"""
## Scatter Plot: {x_column} vs {y_column}

**Correlation:** {correlation:.4f}

This scatter plot shows the relationship between '{x_column}' and '{y_column}'.
{f"The data points are colored by '{hue_column}'." if hue_column else ""}

A correlation coefficient of {correlation:.4f} indicates a {"strong" if abs(correlation) > 0.7 else "moderate" if abs(correlation) > 0.3 else "weak"} 
{"positive" if correlation > 0 else "negative"} relationship.
        """
    except Exception as e:
        return f"Error creating scatter plot: {str(e)}"

def plot_line(x_column: str, y_column: str, title: str = "Line Plot") -> str:
    """
    Create a line plot using Seaborn and display it in Streamlit.
    
    Args:
        x_column (str): Column for x-axis
        y_column (str): Column for y-axis
        title (str): Plot title
        
    Returns:
        str: Result message
    """
    if st.session_state.dataset is None:
        return "Error: No dataset loaded. Please load a dataset first using load_dataset."
    
    try:
        df = st.session_state.dataset
        
        # Check if x_column contains a JSON structure with both x and y columns
        if isinstance(x_column, str) and '{' in x_column and '}' in x_column:
            params = parse_json_input(x_column)
            if 'x_column' in params:
                x_column = params['x_column']
            if 'y_column' in params:
                y_column = params['y_column']
            if 'title' in params:
                title = params['title']
        
        # Clean column names
        x_column = clean_column_name(x_column)
        y_column = clean_column_name(y_column)
        
        # Validate columns
        for col in [x_column, y_column]:
            if col not in df.columns:
                return f"Error: Column '{col}' not found in dataset. Available columns: {', '.join(df.columns)}"
        
        # Create a new figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort the dataframe by x_column to ensure proper line plot
        df_sorted = df.sort_values(by=x_column)
        
        # Create the line plot
        sns.lineplot(data=df_sorted, x=x_column, y=y_column, ax=ax)
        
        # Set title and labels
        ax.set_title(title)
        
        # Display the plot in Streamlit
        st.pyplot(fig)
        
        # Calculate trend (positive, negative, or flat)
        first_y = df_sorted.iloc[0][y_column]
        last_y = df_sorted.iloc[-1][y_column]
        trend = "increasing" if last_y > first_y else "decreasing" if last_y < first_y else "flat"
        
        return f"""
## Line Plot: {y_column} by {x_column}

This line plot shows how '{y_column}' changes with '{x_column}'.

The overall trend appears to be {trend} over the range of {x_column}.

**Range:**
- {x_column}: {df[x_column].min()} to {df[x_column].max()}
- {y_column}: {df[y_column].min()} to {df[y_column].max()}
        """
    except Exception as e:
        return f"Error creating line plot: {str(e)}"

def plot_heatmap(columns: List[str] | str, title: str = "Correlation Heatmap") -> str:
    """
    Create a correlation heatmap using Seaborn and display it in Streamlit.
    
    Args:
        columns (Union[List[str], str]): List of columns or comma-separated string of columns
        title (str): Plot title
        
    Returns:
        str: Result message
    """
    if st.session_state.dataset is None:
        return "Error: No dataset loaded. Please load a dataset first using load_dataset."
    
    try:
        df = st.session_state.dataset
        
        # Convert input to list of columns
        if isinstance(columns, str):
            # Remove any quotes that might be present
            columns = columns.strip('"\'')
            
            # Handle 'all' case
            if columns.lower() == "all":
                columns = df.select_dtypes(include=['number']).columns.tolist()
            else:
                # Split by comma and clean each column name
                columns = [col.strip() for col in columns.split(',')]
        
        # Clean column names
        columns = [col.strip('"\' ') for col in columns]
        
        # Validate columns
        for col in columns:
            if col not in df.columns:
                return f"Error: Column '{col}' not found in dataset. Available columns: {', '.join(df.columns)}"
        
        # Ensure we're only using numeric columns
        numeric_cols = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
        
        if not numeric_cols:
            return "Error: None of the specified columns are numeric. Correlation heatmap requires numeric data."
        
        if len(numeric_cols) < 2:
            return "Error: At least two numeric columns are required for a correlation heatmap."
        
        # Create correlation matrix
        corr_matrix = df[numeric_cols].corr()
        
        # Create a new figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create the heatmap
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        
        # Set title
        ax.set_title(title)
        
        # Display the plot in Streamlit
        st.pyplot(fig)
        
        # Find strongest correlations
        corr_pairs = [
            (numeric_cols[i], numeric_cols[j], corr_matrix.iloc[i, j])
            for i in range(len(numeric_cols))
            for j in range(i+1, len(numeric_cols))
        ]
        
        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        # Generate insights
        insights = "## Correlation Insights\n\n"
        
        if corr_pairs:
            insights += "**Strongest correlations:**\n"
            for col1, col2, corr in corr_pairs[:3]:
                strength = (
                    'strong positive' if corr > 0.7 else
                    'moderate positive' if corr > 0.3 else
                    'weak positive' if corr > 0 else
                    'strong negative' if corr < -0.7 else
                    'moderate negative' if corr < -0.3 else
                    'weak negative'
                )
                insights += f"- {col1} and {col2}: {corr:.4f} ({strength})\n"
        
        return insights
    except Exception as e:
        return f"Error creating heatmap: {str(e)}"
