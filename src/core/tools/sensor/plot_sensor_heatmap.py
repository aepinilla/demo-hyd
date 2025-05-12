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
    
    # The sensor data is in 'long' format where each row has a value_type and a value
    # We need to reshape it to 'wide' format to properly correlate between different value types
    st.write("### Data Structure Before Reshaping:")
    st.write(df.head())
    
    # First, check what value_types are available in the data
    available_types = df['value_type'].unique().tolist()
    st.write(f"Available value types in data: {available_types}")
    
    # Filter to just the requested variables that are actually in the data
    if variables_to_plot.lower() == "all":
        valid_vars = available_types
    else:
        # Parse the variables_to_plot string into a list
        if isinstance(variables_to_plot, str):
            if ',' in variables_to_plot:
                requested_vars = [v.strip() for v in variables_to_plot.split(',')]
            else:
                requested_vars = [variables_to_plot.strip()]
        else:
            requested_vars = variables_to_plot
            
        valid_vars = [var for var in requested_vars if var in available_types]
    
    # If we don't have enough valid variables, use the most common ones
    if len(valid_vars) < 2:
        st.warning(f"Not enough valid variables found. Using P1 and P2 if available.")
        common_vars = [var for var in ['P1', 'P2'] if var in available_types]
        if len(common_vars) >= 2:
            valid_vars = common_vars
        else:
            # If P1 and P2 aren't available, use the first two available types
            if len(available_types) >= 2:
                valid_vars = available_types[:2]
            else:
                return "Error: Not enough data types available for correlation analysis."
    
    st.write(f"Using value types for correlation: {valid_vars}")
    
    # Filter to only include the valid value types
    filtered_df = df[df['value_type'].isin(valid_vars)].copy()
    
    # Create a wide format dataframe where each value_type becomes a column
    # We'll use the latest timestamp for each sensor to ensure we're comparing measurements from the same time
    # First, get the latest timestamp for each sensor
    latest_timestamps = filtered_df.groupby('sensor_id')['timestamp'].max().reset_index()
    latest_timestamps = latest_timestamps.rename(columns={'timestamp': 'latest_timestamp'})
    
    # Merge to get only the latest measurements for each sensor
    merged_df = pd.merge(filtered_df, latest_timestamps, on='sensor_id')
    latest_data = merged_df[merged_df['timestamp'] == merged_df['latest_timestamp']]
    
    # Now pivot to wide format
    wide_df = latest_data.pivot_table(
        index='sensor_id',
        columns='value_type',
        values='value',
        aggfunc='mean'  # Use mean if there are multiple values for the same sensor/value_type
    ).reset_index()
    
    # Add latitude and longitude if available
    if 'latitude' in df.columns and 'longitude' in df.columns:
        # Get the latitude and longitude for each sensor (using the first occurrence)
        geo_data = df.drop_duplicates('sensor_id')[['sensor_id', 'latitude', 'longitude']]
        wide_df = pd.merge(wide_df, geo_data, on='sensor_id', how='left')
    
    st.write("### Data After Reshaping to Wide Format:")
    st.write(wide_df.head())
    
    # Drop the sensor_id column for correlation
    corr_df = wide_df.drop(columns=['sensor_id'])
    
    # Check if we have enough data points
    if len(corr_df) < 2:
        st.warning("Not enough data points for correlation. Showing raw data instead.")
        st.write(filtered_df.head(10))
        return "Error: Not enough data points for correlation analysis. Need at least 2 data points."
        
    # Display information about the data being used for correlation
    st.info(f"Analyzing correlations between {len(corr_df.columns)} variables using {len(corr_df)} sensor data points.")
    
    # Calculate the correlation matrix
    try:
        # Make sure we have numeric data for correlation
        numeric_df = corr_df.apply(pd.to_numeric, errors='coerce')
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Debug: Print correlation matrix
        st.write("### Correlation Matrix:")
        st.write(corr_matrix)
        
        # Create a new figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create the heatmap without masking to show all correlations
        # Create the heatmap
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax, 
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
            # Get the column names from the correlation matrix
            corr_cols = corr_matrix.index.tolist()
            
            for i in range(len(corr_cols)):
                for j in range(i+1, len(corr_cols)):
                    # Get the correlation value
                    corr_value = corr_matrix.iloc[i, j]
                    # Only add valid correlation values
                    if not pd.isna(corr_value):
                        corr_pairs.append((corr_cols[i], corr_cols[j], corr_value))
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
