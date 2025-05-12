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
    if isinstance(variables_to_plot, str) and '{' in variables_to_plot and '}' in variables_to_plot:
        try:
            # Try to parse as JSON
            params = json.loads(variables_to_plot)
            
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
        except json.JSONDecodeError:
            st.warning(f"Failed to parse JSON input: {variables_to_plot}")
            # Continue with original parameters
    
    # Force a refresh of the data to avoid caching issues
    df, mapped_vars, error_msg = prepare_sensor_data(variables_to_plot, force_refresh=True)
    
    if error_msg:
        st.warning(error_msg)
        return error_msg
    
    # Filter the dataframe to only include the requested variables
    filtered_df = df[df['value_type'].isin(mapped_vars)]
    
    if filtered_df.empty:
        return "Error: No data available for the requested variables after filtering."
    
    # Create a figure with subplots for each variable
    fig, axes = plt.subplots(len(mapped_vars), 1, figsize=(10, 4 * len(mapped_vars)), sharex=False)
    
    # If only one variable, axes is not an array
    if len(mapped_vars) == 1:
        axes = [axes]
    
    # Plot histograms for each variable
    for i, var in enumerate(mapped_vars):
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
    for var in mapped_vars:
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


