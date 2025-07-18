import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from src.utils.visualization import prepare_sensor_data


def plot_sensor_histogram(variables_to_plot: str = "P1,P2", bins: int = 30, title: str = "Distribution of Sensor Values") -> str:
    """
    Create a histogram of sensor data values. This is specialized for visualizing the distribution
    of particulate matter levels from sensors.
    
    Args:
        variables_to_plot (str): Comma-separated list of variables to plot (e.g., "P1,P2")
        bins (int): Number of bins for the histogram
        title (str): Plot title
        
    Returns:
        str: Result message with histogram statistics
    """
    
    # Handle JSON input from LangChain agent
    # Check all parameters for JSON structure
    for param_name, param_value in {'variables_to_plot': variables_to_plot, 'bins': bins, 'title': title}.items():
        if isinstance(param_value, str) and '{' in param_value and '}' in param_value:
            
            # Check for potential escape character issues
            if '\\"' in param_value or '\\\\' in param_value:
                st.warning(f"Found escape characters that might cause issues")
                # Try to clean up common escape character issues
                cleaned_value = param_value.replace('\\"', '"').replace('\\\\', '\\')
                st.write(f"Cleaned value: {repr(cleaned_value)}")
                try:
                    params = json.loads(cleaned_value)
                    st.success("Successfully parsed JSON after cleaning!")
                    # Extract parameters from the JSON object
                    if 'variables_to_plot' in params:
                        variables_to_plot = params['variables_to_plot']
                    elif 'column' in params:  # Support for standard tool parameter name
                        variables_to_plot = params['column']
                    if 'bins' in params:
                        try:
                            bins = int(params['bins'])
                        except (ValueError, TypeError):
                            pass
                    if 'title' in params:
                        title = params['title']
                        
                    st.info(f"Parsed JSON input: variables={variables_to_plot}, bins={bins}, title={title}")
                    break  # Only need to parse one JSON object
                except json.JSONDecodeError as e:
                    st.warning(f"Still failed to parse after cleaning: {e}")
            
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
                elif 'column' in params:  # Support for standard tool parameter name
                    variables_to_plot = params['column']
                if 'bins' in params:
                    try:
                        bins = int(params['bins'])
                    except (ValueError, TypeError):
                        pass
                if 'title' in params:
                    title = params['title']
                
                # Quietly break - no need to show success messages to the user
                print(f"Parsed JSON parameters: variables={variables_to_plot}, bins={bins}, title={title}")
                break  # Only need to parse one JSON object
            except json.JSONDecodeError as e:
                # Log the error for debugging but continue with the original parameters
                print(f"JSON parse error: {e}")
                continue
    
    # Special case: if 'value' is requested directly, use all available value types
    if variables_to_plot.lower() == 'value':
        # Get the dataset from session state
        if 'dataset' in st.session_state and st.session_state.dataset is not None:
            df = st.session_state.dataset
            # Use all available value types
            requested_vars = df['value_type'].unique().tolist()
            st.info(f"Using all available sensor value types: {', '.join(requested_vars)}")
            error_msg = None
        else:
            return "Error: No dataset available in session state. Please fetch sensor data first."
    else:
        # Force a refresh of the data to avoid caching issues
        df, requested_vars, error_msg = prepare_sensor_data(variables_to_plot, force_refresh=True)
        
        if error_msg:
            st.warning(error_msg)
            return error_msg
    
    # Filter the dataframe to only include the requested variables
    filtered_df = df[df['value_type'].isin(requested_vars)].copy()
    
    # Filter to use only the latest measurement of each unique sensor
    # First, get the latest timestamp for each sensor
    latest_timestamps = filtered_df.groupby('sensor_id')['timestamp'].max().reset_index()
    latest_timestamps = latest_timestamps.rename(columns={'timestamp': 'latest_timestamp'})
    
    # Merge to get only the latest measurements for each sensor
    merged_df = pd.merge(filtered_df, latest_timestamps, on='sensor_id')
    filtered_df = merged_df[merged_df['timestamp'] == merged_df['latest_timestamp']]
    
    if filtered_df.empty:
        return "Error: No data available for the requested variables after filtering."
    
    # Create a figure with subplots for each variable
    fig, axes = plt.subplots(len(requested_vars), 1, figsize=(10, 4 * len(requested_vars)), sharex=False)
    
    # If only one variable, axes is not an array
    if len(requested_vars) == 1:
        axes = [axes]
    
    # Plot histograms for each variable
    for i, var in enumerate(requested_vars):
        var_df = filtered_df[filtered_df['value_type'] == var]
        
        # Skip if no data for this variable
        if var_df.empty:
            continue
        
        # Plot histogram
        sns.histplot(var_df['value'], bins=bins, kde=True, ax=axes[i])
        
        # Set title and labels
        axes[i].set_title(f"Distribution of {var}")
        axes[i].set_xlabel(f"{var} Value")
        axes[i].set_ylabel("Frequency")
        
        # Add grid
        axes[i].grid(True, linestyle='--', alpha=0.7)
    
    # Set overall title
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
    
    # Display the plot
    st.pyplot(fig)
    
    # Generate statistics for each variable in a more concise way
    stats = []
    for var in requested_vars:
        var_df = filtered_df[filtered_df['value_type'] == var]
        if not var_df.empty:
            var_stats = var_df['value'].describe()
            stats_items = [
                f"### {var} Statistics",
                f"- **Count:** {var_stats['count']:.0f}",
                f"- **Mean:** {var_stats['mean']:.2f}",
                f"- **Std Dev:** {var_stats['std']:.2f}",
                f"- **Min/Max:** {var_stats['min']:.2f} / {var_stats['max']:.2f}",
                f"- **Quartiles:** {var_stats['25%']:.2f} / {var_stats['50%']:.2f} / {var_stats['75%']:.2f}\n"
            ]
            stats.extend(stats_items)
    
    return f"## Histogram Analysis of Sensor Data\n\n{''.join(stats)}"


