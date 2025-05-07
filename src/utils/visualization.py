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

def plot_scatter(x_column: str, y_column: str, hue_column: Optional[str] = None, 
                 title: str = "Scatter Plot") -> str:
    """
    Create a scatter plot using Seaborn and display it in Streamlit.
    
    Args:
        x_column (str): Column for x-axis
        y_column (str): Column for y-axis
        hue_column (str, optional): Column for color grouping
        title (str): Plot title
        
    Returns:
        str: Result message
    """
    if st.session_state.dataset is None:
        return "Error: No dataset loaded. Please load a dataset first using load_dataset."
    
    try:
        df = st.session_state.dataset
        
        # Validate columns
        for col in [x_column, y_column]:
            if col not in df.columns:
                return f"Error: Column '{col}' not found in dataset. Available columns: {', '.join(df.columns)}"
        
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

def plot_heatmap(columns: List[str], title: str = "Correlation Heatmap") -> str:
    """
    Create a correlation heatmap using Seaborn and display it in Streamlit.
    
    Args:
        columns (List[str]): List of columns to include in the heatmap
        title (str): Plot title
        
    Returns:
        str: Result message
    """
    if st.session_state.dataset is None:
        return "Error: No dataset loaded. Please load a dataset first using load_dataset."
    
    try:
        df = st.session_state.dataset
        
        # Validate columns
        for col in columns:
            if col not in df.columns:
                return f"Error: Column '{col}' not found in dataset. Available columns: {', '.join(df.columns)}"
        
        # Ensure we're only using numeric columns
        numeric_cols = []
        for col in columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_cols.append(col)
        
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
        
        # Find strongest positive and negative correlations
        corr_pairs = []
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                corr_pairs.append((
                    numeric_cols[i], 
                    numeric_cols[j], 
                    corr_matrix.iloc[i, j]
                ))
        
        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        # Generate insights
        insights = "## Correlation Insights\n\n"
        
        if corr_pairs:
            insights += "**Strongest correlations:**\n"
            for col1, col2, corr in corr_pairs[:3]:
                insights += f"- {col1} and {col2}: {corr:.4f} ({('strong positive' if corr > 0.7 else 'moderate positive' if corr > 0.3 else 'weak positive') if corr > 0 else ('strong negative' if corr < -0.7 else 'moderate negative' if corr < -0.3 else 'weak negative')})\n"
        
        return insights
    except Exception as e:
        return f"Error creating heatmap: {str(e)}"
