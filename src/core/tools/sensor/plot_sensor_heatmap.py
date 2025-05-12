import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from src.utils.visualization import prepare_sensor_data, create_sensor_pivot_table


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
    df, requested_vars, error_msg = prepare_sensor_data(variables_to_plot, force_refresh=True)
    
    if error_msg:
        st.warning(error_msg)
        return error_msg
    
    # Ensure we have at least 2 variables
    if len(requested_vars) < 2:
        available_vars = df['value_type'].unique().tolist()
        
        # If we have at least 2 available variables, use them instead
        if len(available_vars) >= 2:
            requested_vars = available_vars  # Use all available variables
            print(f"Not enough variables specified. Using all available variables instead.")
        else:
            return "Error: At least 2 variables are required for a correlation heatmap."
    
    # Create a pivot table with properly aligned measurements
    pivot_df = create_sensor_pivot_table(df, requested_vars)
    
    # Check if we have any data points
    if len(pivot_df) < 2 or pivot_df.empty:
        # Try with just P1 and P2 which are the most common variables
        print("Not enough data points with all variables. Trying with just P1 and P2...")
        common_vars = [var for var in ['P1', 'P2'] if var in df['value_type'].unique()]
        if len(common_vars) >= 2:
            pivot_df = create_sensor_pivot_table(df, common_vars)
            requested_vars = common_vars
        else:
            return "Error: Not enough data points for correlation analysis. Need at least 2 data points."
        
    # Ensure we have at least 2 rows in the pivot table
    if len(pivot_df) < 2 or pivot_df.empty:
        return "Error: Not enough data points for correlation analysis. Need at least 2 data points."
        
    # Display information about the data being used for correlation
    st.info(f"Analyzing correlations between {', '.join(requested_vars)} using {len(pivot_df)} data points from {pivot_df['sensor_id'].nunique() if 'sensor_id' in pivot_df.columns else 'multiple'} sensors.")
    
    # Calculate the correlation matrix
    try:
        # Make sure we have numeric data for correlation
        numeric_df = pivot_df[requested_vars].apply(pd.to_numeric, errors='coerce')
        
        # Preserve original data values without any artificial modifications
        # to maintain data integrity for accurate correlation analysis
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Create a new figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create the heatmap with a mask to hide the diagonal (self-correlations)
        mask = np.zeros_like(corr_matrix, dtype=bool)
        np.fill_diagonal(mask, True)  # Mask the diagonal
        
        # Create the heatmap
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap="coolwarm", fmt=".2f", ax=ax, 
                   vmin=-1, vmax=1, center=0, square=True, linewidths=.5)
    except Exception as e:
        st.error(f"Error calculating correlation matrix: {e}")
        return f"Error calculating correlation matrix: {e}"
    
    # Set title
    ax.set_title(title)
    
    # Display the plot
    st.pyplot(fig)
    
    # Find strongest correlations
    corr_pairs = []
    try:
        # Only proceed if we have a valid correlation matrix
        if corr_matrix is not None and not corr_matrix.empty and len(corr_matrix) >= 2:
            for i in range(len(requested_vars)):
                for j in range(i+1, len(requested_vars)):
                    # Check if both variables exist in the correlation matrix
                    if requested_vars[i] in corr_matrix.index and requested_vars[j] in corr_matrix.columns:
                        corr_value = corr_matrix.loc[requested_vars[i], requested_vars[j]]
                        # Only add valid correlation values
                        if not pd.isna(corr_value):
                            corr_pairs.append((requested_vars[i], requested_vars[j], corr_value))
    except Exception as e:
        st.warning(f"Error calculating correlation pairs: {e}")
        # Continue with any pairs we managed to calculate
    
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
