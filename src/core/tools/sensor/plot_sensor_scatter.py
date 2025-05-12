import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from scipy.stats import pearsonr
from src.utils.visualization import prepare_sensor_data, create_sensor_pivot_table


def plot_sensor_scatter(x_variable: str = "P1", y_variable: str = "P2", title: str = "Relationship Between Sensor Variables") -> str:
    """
    Create a scatter plot comparing two sensor variables. This is specialized for visualizing
    relationships between particulate matter levels from sensors.
    
    Args:
        x_variable (str): Variable for x-axis (e.g., "P1")
        y_variable (str): Variable for y-axis (e.g., "P2")
        title (str): Plot title
        
    Returns:
        str: Result message with correlation statistics
    """
    
    # Handle JSON input from LangChain agent
    # Check if x_variable contains a JSON structure with all parameters
    if isinstance(x_variable, str) and '{' in x_variable and '}' in x_variable:
        try:
            # Try to parse as JSON
            params = json.loads(x_variable)
            
            # Extract parameters from the JSON object
            if 'x_variable' in params:
                x_variable = params['x_variable']
            if 'y_variable' in params:
                y_variable = params['y_variable']
            if 'title' in params:
                title = params['title']
                
            st.info(f"Parsed JSON input: x_variable={x_variable}, y_variable={y_variable}, title={title}")
        except json.JSONDecodeError:
            st.warning(f"Failed to parse JSON input: {x_variable}")
    
    # Force a refresh of the data to avoid caching issues
    df, _, error_msg = prepare_sensor_data("all", force_refresh=True)
    
    if error_msg:
        st.warning(error_msg)
        return error_msg
    
    # Get all available variables
    available_vars = df['value_type'].unique().tolist()
    
    # Try to find exact matches first
    x_var = x_variable if x_variable in available_vars else None
    y_var = y_variable if y_variable in available_vars else None
    
    # If not found, try case-insensitive partial matches
    if not x_var:
        for var in available_vars:
            if x_variable.lower() == var.lower():
                x_var = var
                break
    
    if not y_var:
        for var in available_vars:
            if y_variable.lower() == var.lower():
                y_var = var
                break
    
    # If still not found, provide a helpful error message
    if not x_var or not y_var:
        error = f"Error: Could not find requested variables '{x_variable}' and/or '{y_variable}' in the data. Available variables: {', '.join(available_vars)}"
        st.warning(error)
        return error
    
    # Create a pivot table to properly align measurements
    plot_df = create_sensor_pivot_table(df, [x_var, y_var])
    
    if len(plot_df) < 2:
        error = f"Error: Not enough data points for variables {x_var} and {y_var} to create a scatter plot."
        st.warning(error)
        return error
    
    # Create the scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot with regression line
    sns.regplot(x=x_var, y=y_var, data=plot_df, scatter_kws={'alpha':0.5}, line_kws={'color':'red'}, ax=ax)
    
    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel(f"{x_var} Value")
    ax.set_ylabel(f"{y_var} Value")
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Calculate correlation
    corr, p_value = pearsonr(plot_df[x_var], plot_df[y_var])
    
    # Add correlation annotation
    correlation_text = f"Correlation: {corr:.4f} (p-value: {p_value:.4f})"
    ax.annotate(correlation_text, xy=(0.05, 0.95), xycoords='axes fraction', 
                fontsize=12, ha='left', va='top',
                bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7))
    
    # Display the plot
    st.pyplot(fig)
    
    # Generate correlation interpretation with simplified logic
    corr_strength = "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.3 else "weak"
    corr_direction = "positive" if corr > 0 else "negative"
    significance = "statistically significant" if p_value < 0.05 else "not statistically significant"
    
    # Create a more concise interpretation
    relationship = f"As {x_var} increases, {y_var} tends to {('increase' if corr > 0 else 'decrease')}" if abs(corr) > 0.3 else "There is no strong linear relationship between these variables"
    
    return f"""## Scatter Plot Analysis: {x_var} vs {y_var}

### Correlation Statistics:
- **Pearson Correlation:** {corr:.4f} ({corr_strength} {corr_direction})
- **P-value:** {p_value:.4f} ({significance})
- **Sample Size:** {len(plot_df)} data points

### Interpretation:
{relationship}
"""