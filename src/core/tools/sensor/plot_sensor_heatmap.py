import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from src.utils.visualization import prepare_sensor_data, create_sensor_pivot_table, check_sensor_data_quality


def plot_sensor_heatmap(variables_to_plot: str = "all", title: str = "Correlation Heatmap of Sensor Variables") -> str:
    """
    Create a correlation heatmap for sensor variables. This is specialized for visualizing
    relationships between particulate matter levels and other sensor readings.
    
    Args:
        variables_to_plot (str): Comma-separated list of variables to include in the heatmap (e.g., "P1,P2,temperature")
        title (str): Plot title
        
    Returns:
        str: Result message with correlation insights
    """
    
    # Handle JSON input from LangChain agent
    # Check all parameters for JSON structure
    for param_name, param_value in {'variables_to_plot': variables_to_plot, 'title': title}.items():
        if isinstance(param_value, str) and '{' in param_value and '}' in param_value:
            # Clean up any markdown formatting or trailing characters
            cleaned_param = param_value
            
            # Remove trailing backticks and newlines if present
            if '\n' in cleaned_param or '```' in cleaned_param:
                # Remove any newline followed by backticks
                cleaned_param = cleaned_param.split('\n```')[0]
                # Remove any standalone backticks at the end
                if cleaned_param.endswith('```'):
                    cleaned_param = cleaned_param[:-3]
                # Trim whitespace
                cleaned_param = cleaned_param.strip()
            
            try:
                # Try to parse as JSON with cleaned parameter
                params = json.loads(cleaned_param)
                
                # Extract parameters from the JSON object
                if 'variables_to_plot' in params:
                    variables_to_plot = params['variables_to_plot']
                elif 'columns' in params:  # Support for standard tool parameter name
                    variables_to_plot = params['columns']
                if 'title' in params:
                    title = params['title']
                
                # Quietly log the parsed parameters
                print(f"Parsed JSON parameters: variables={variables_to_plot}, title={title}")
                break  # Only need to parse one JSON object
            except json.JSONDecodeError as e:
                # Log the error for debugging but continue with the original parameters
                print(f"JSON parse error: {e}")
                continue
    
    # Force a refresh of the data to avoid caching issues
    df, mapped_vars, error_msg = prepare_sensor_data(variables_to_plot, force_refresh=True)
    
    # If 'all' is specified, use all available variables
    if variables_to_plot.lower() == 'all':
        # Get all available variables
        available_vars = df['value_type'].unique().tolist()
        
        # Use all available variables
        variables_to_plot = ','.join(available_vars)
        print(f"Using all available variables: {variables_to_plot}")
        
        # Reload data with all the variables
        df, mapped_vars, error_msg = prepare_sensor_data(variables_to_plot, force_refresh=True)
    
    if error_msg:
        st.warning(error_msg)
        return error_msg
    
    # Ensure we have at least 2 variables
    if len(mapped_vars) < 2:
        available_vars = df['value_type'].unique().tolist()
        
        # If we have at least 2 available variables, use them instead
        if len(available_vars) >= 2:
            mapped_vars = available_vars[:5]  # Limit to 5 for readability
            st.info(f"Not enough variables specified. Using {', '.join(mapped_vars)} instead.")
        else:
            return "Error: At least 2 variables are required for a correlation heatmap."
    
    # Create a pivot table with properly aligned measurements
    pivot_df = create_sensor_pivot_table(df, mapped_vars)
    
    # Check if we have any data points
    if len(pivot_df) < 2:
        return "Error: Not enough data points for correlation analysis. Need at least 2 data points."
        
    # Display information about the data being used for correlation
    st.info(f"Analyzing correlations between {', '.join(mapped_vars)} using {len(pivot_df)} data points from {pivot_df['sensor_id'].nunique() if 'sensor_id' in pivot_df.columns else 'multiple'} sensors.")
    
    # Check data quality
    is_suspicious, warning_msg = check_sensor_data_quality(pivot_df, mapped_vars)
    if is_suspicious and warning_msg:
        st.warning(warning_msg)
    
    # Calculate the correlation matrix
    corr_matrix = pivot_df[mapped_vars].corr()
    
    # Create a new figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create the heatmap
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    
    # Set title
    ax.set_title(title)
    
    # Display the plot
    st.pyplot(fig)
    
    # Find strongest correlations
    corr_pairs = []
    for i in range(len(mapped_vars)):
        for j in range(i+1, len(mapped_vars)):
            corr_pairs.append((mapped_vars[i], mapped_vars[j], corr_matrix.iloc[i, j]))
    
    # Sort by absolute correlation value
    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    # Generate insights
    insights = "## Correlation Insights\n\n"
    
    if corr_pairs:
        insights += "**Strongest correlations:**\n"
        for var1, var2, corr in corr_pairs:
            corr_strength = "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.3 else "weak"
            corr_direction = "positive" if corr > 0 else "negative"
            insights += f"- {var1} and {var2}: {corr:.4f} ({corr_strength} {corr_direction})\n"
    
    return insights
