from typing import List, Dict, Any, Optional, Union
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pydantic.v1 import BaseModel, Field
from langchain.tools import StructuredTool

# Import visualization utilities
from src.utils.visualization import (
    load_dataset,
    plot_histogram,
    plot_scatter,
    plot_heatmap,
    get_column_types
)

# Import sensor API utilities
from src.utils.sensor_api import (
    fetch_latest_data,
    get_sensor_types
)

def plot_sensor_data(x_column: str = "timestamp", y_column: str = "value", title: str = "Sensor Data Time Series", hue_column: str = "value_type", time_range: str = "recent", variables_to_plot: str = None) -> str:
    """
    Create a time series plot of sensor data. This is a specialized function for visualizing
    sensor data with sensible defaults for sensor data visualization.
    
    Args:
        x_column (str): Column for x-axis, defaults to 'timestamp'
        y_column (str): Column for y-axis, defaults to 'value'
        title (str): Plot title
        hue_column (str): Column to use for grouping/coloring the lines, defaults to 'value_type'
        time_range (str): Time range for data, options: 'recent', 'daily', 'weekly', 'monthly', 'yearly'
                          Note: The sensor.community API only provides recent data (last 5 minutes).
                          Historical data requires using the archive at https://archive.sensor.community/
        variables_to_plot (str): Comma-separated list of variables to plot (e.g., "PM10,PM2.5,temperature").
                                 Will be mapped to corresponding value_types in the dataset.
                                 If not provided, will ask user to specify variables (max 5).
        
    Returns:
        str: Result message with plot statistics
    """
    import streamlit as st
    import pandas as pd
    from src.utils.visualization import plot_line
    from datetime import datetime, timedelta
    
    # Check if we have sensor data in session state
    if 'latest_data' not in st.session_state or st.session_state.latest_data.empty:
        return "Error: No sensor data available. Please fetch sensor data first using fetch_latest_sensor_data."
    
    # Get the data and make a copy to avoid modifying the original
    df = st.session_state.latest_data.copy()
    
    # Verify that we have a timestamp column that can be used for time series visualization
    if 'timestamp' not in df.columns:
        return "Error: The sensor data does not contain a timestamp column required for time series visualization."
    
    # Check if the timestamp column is already a datetime type
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        st.warning("Converting timestamp column to datetime format...")
        try:
            # Try to convert the timestamp column to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            
            # Check if conversion was successful (no NaT values)
            if df['timestamp'].isna().all():
                return "Error: Could not convert timestamp column to datetime format. Please check the data."
            
            # Drop rows with NaT values if any
            if df['timestamp'].isna().any():
                original_len = len(df)
                df = df.dropna(subset=['timestamp'])
                st.warning(f"Dropped {original_len - len(df)} rows with invalid timestamps.")
                
        except Exception as e:
            return f"Error converting timestamp column: {str(e)}"
    
    # Check the time range requested and provide appropriate information
    if time_range != "recent":
        # Get the actual time range of the data
        min_date = df['timestamp'].min().strftime('%Y-%m-%d %H:%M')
        max_date = df['timestamp'].max().strftime('%Y-%m-%d %H:%M')
        time_diff = df['timestamp'].max() - df['timestamp'].min()
        
        # Inform the user about the data limitations
        st.warning(f"⚠️ **Data Limitation Notice**\n\n"
                  f"You requested '{time_range}' data, but the sensor.community API only provides data from the last 5 minutes.\n"
                  f"Current data range: {min_date} to {max_date} (approximately {time_diff.total_seconds()/60:.1f} minutes)\n\n"
                  f"For historical data, you would need to use the archive at https://archive.sensor.community/")
    
    # Store the processed data in session state
    st.session_state.dataset = df
    
    # Filter for SDS011 sensor data if requested in the title
    if "SDS011" in title and "sensor_type" in df.columns:
        df = df[df.sensor_type == "SDS011"]
        st.session_state.dataset = df
        st.info(f"Filtered data for SDS011 sensors: {len(df)} readings")
    
    # Get available value types in the dataset before filtering
    available_value_types = df['value_type'].unique().tolist()
    st.info(f"Available variables in the dataset: {', '.join(available_value_types)}")
    
    # Define common variable mappings
    variable_mappings = {
        "pm10": "P1",
        "pm2.5": "P2",
        "pm2_5": "P2",
        "temperature": "temperature",
        "humidity": "humidity",
        "pressure": "pressure",
        "p1": "P1",
        "p2": "P2"
    }
    
    # If variables_to_plot is not provided, either ask user to specify or limit to P1 and P2
    if variables_to_plot is None:
        # Always limit the number of variables when none are specified
        if len(available_value_types) > 5:
            st.warning("⚠️ Too many variables available. Please specify which ones you want to visualize or I'll show just PM10 (P1) and PM2.5 (P2).")
            
            # Default to just showing P1 and P2 if available
            default_vars = [var for var in available_value_types if var in ["P1", "P2"]]
            
            if default_vars:
                df = df[df['value_type'].isin(default_vars)]
                st.session_state.dataset = df
                st.info(f"Showing only the most common particulate matter measurements: {', '.join(default_vars)}")
            else:
                # If P1 and P2 aren't available, just take the first 2 value types
                limited_vars = available_value_types[:2]
                df = df[df['value_type'].isin(limited_vars)]
                st.session_state.dataset = df
                st.info(f"Limited to first two variables: {', '.join(limited_vars)}")
        else:
            # If there are 5 or fewer variables, we can show all of them
            st.info(f"Showing all available variables: {', '.join(available_value_types)}")
            
        # Update the session state with the filtered data
        st.session_state.dataset = df
    else:
        # Parse the variables to plot
        requested_vars = [v.strip().lower() for v in variables_to_plot.split(',')]
        
        # Map user-friendly names to actual value_types
        mapped_vars = []
        for var in requested_vars:
            if var in variable_mappings:
                mapped_vars.append(variable_mappings[var])
            else:
                # Try to find a close match
                for available_var in available_value_types:
                    if var in available_var.lower() or available_var.lower() in var:
                        mapped_vars.append(available_var)
                        break
        
        # Limit to 5 variables
        mapped_vars = mapped_vars[:5]
        
        if not mapped_vars:
            st.warning(f"None of the requested variables ({variables_to_plot}) could be mapped to available value types.")
            st.info(f"Available variables: {', '.join(available_value_types)}")
            return f"Error: Could not map requested variables to available value types. Available: {', '.join(available_value_types)}"
        
        # Filter the dataframe to only include the requested variables
        df = df[df['value_type'].isin(mapped_vars)]
        st.session_state.dataset = df
        st.info(f"Filtered data for variables: {', '.join(mapped_vars)} ({len(df)} readings)")
    
    # Limit the number of variables if there are too many
    if len(df['value_type'].unique()) > 5:
        top_value_types = df['value_type'].value_counts().head(5).index.tolist()
        df = df[df['value_type'].isin(top_value_types)]
        st.warning(f"Limiting visualization to the top 5 most frequent variables: {', '.join(top_value_types)}")
        st.session_state.dataset = df
    
    # Check if we have enough data points for a meaningful visualization
    if len(df) < 2:
        return "Error: Not enough data points for a time series visualization after filtering."
    
    # Update the title to reflect the actual time range and variables
    if "last year" in title.lower() or "yearly" in title.lower() or time_range in ["yearly", "monthly", "weekly", "daily"]:
        min_date = df['timestamp'].min().strftime('%Y-%m-%d %H:%M')
        max_date = df['timestamp'].max().strftime('%Y-%m-%d %H:%M')
        title = f"{title.split('(')[0].strip()} (Available data: {min_date} to {max_date})"
    
    # Update title to show which variables are being plotted
    plotted_vars = df['value_type'].unique().tolist()
    if len(plotted_vars) <= 5:
        title = f"{title} - Variables: {', '.join(plotted_vars)}"
    
    # Create a new figure for our plot to ensure we're only showing the filtered data
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.ticker import MaxNLocator
    
    # Create a new figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create the line plot with only our filtered data
    sns.lineplot(data=df, x=x_column, y=y_column, hue=hue_column, ax=ax)
    
    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    
    # Improve datetime axis formatting
    if pd.api.types.is_datetime64_any_dtype(df[x_column]):
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Determine appropriate date format based on the time range
        time_range = df[x_column].max() - df[x_column].min()
        if time_range.total_seconds() < 3600:  # Less than an hour
            date_format = '%H:%M:%S'
        elif time_range.total_seconds() < 86400:  # Less than a day
            date_format = '%H:%M'
        elif time_range.days < 7:  # Less than a week
            date_format = '%m-%d %H:%M'
        elif time_range.days < 365:  # Less than a year
            date_format = '%b %d'
        else:  # More than a year
            date_format = '%Y-%m-%d'
            
        # Format the date ticks nicely
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter(date_format))
        
        # Add grid lines for better readability
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set appropriate number of ticks based on data size
        if len(df) > 20:
            # For larger datasets, limit the number of ticks to avoid overcrowding
            ax.xaxis.set_major_locator(MaxNLocator(10))
        
        # Adjust layout to make room for rotated labels
        fig.tight_layout()
    
    # Display the plot in Streamlit
    st.pyplot(fig)
    
    # Add some statistics
    y_stats = df[y_column].describe()
    
    # For datetime columns, show date range
    if pd.api.types.is_datetime64_any_dtype(df[x_column]):
        date_range = f"From {df[x_column].min()} to {df[x_column].max()}"
        x_stats_text = f"**Date Range:** {date_range}"
    else:
        x_stats = df[x_column].describe()
        x_stats_text = f"**{x_column}:**\n- Mean: {x_stats['mean']:.2f}\n- Std Dev: {x_stats['std']:.2f}"
    
    # Return statistics about the plot
    return f"""
## Line Plot Statistics

{x_stats_text}

**{y_column}:**
- Mean: {y_stats['mean']:.2f}
- Std Dev: {y_stats['std']:.2f}

**Range:**
- {x_column}: {df[x_column].min()} to {df[x_column].max()}
- {y_column}: {df[y_column].min()} to {df[y_column].max()}
    """

def prepare_sensor_data_for_visualization() -> str:
    """
    Prepare the latest fetched sensor data for visualization with the plotting tools.
    This function makes the sensor data available to the visualization tools without
    requiring the user to save and load a CSV file.
    
    Returns:
        str: Information about the prepared dataset
    """
    import streamlit as st
    import pandas as pd
    
    # Check if we have sensor data in session state
    if 'latest_data' in st.session_state and not st.session_state.latest_data.empty:
        # Get the data and make a copy to avoid modifying the original
        df = st.session_state.latest_data.copy()
        
        # Verify that we have a timestamp column that can be used for time series visualization
        if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            st.warning("Converting timestamp column to datetime format...")
            try:
                # Try to convert the timestamp column to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                
                # Check if conversion was successful (no NaT values)
                if df['timestamp'].isna().all():
                    st.warning("Could not convert timestamp column to datetime format. Some visualizations may not work properly.")
                
                # Drop rows with NaT values if any
                if df['timestamp'].isna().any():
                    original_len = len(df)
                    df = df.dropna(subset=['timestamp'])
                    st.warning(f"Dropped {original_len - len(df)} rows with invalid timestamps.")
                    
            except Exception as e:
                st.warning(f"Error converting timestamp column: {str(e)}")
        
        # Copy the data to the dataset session state variable used by visualization tools
        st.session_state.dataset = df
        
        # Get information about the dataset
        return f"""
## Sensor Data Prepared for Visualization

**Source:** sensor.community API
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

You can now use visualization tools like plot_scatter, plot_histogram, etc. directly on this data.
        """
    else:
        return "Error: No sensor data available. Please fetch sensor data first using fetch_latest_sensor_data."


def _prepare_sensor_data(variables_to_plot: str = "all") -> tuple:
    """
    Helper function to prepare sensor data for visualization.
    
    Args:
        variables_to_plot (str): Comma-separated list of variables to plot or 'all'
        
    Returns:
        tuple: (DataFrame with sensor data, list of mapped variable names, error message or None)
    """
    import streamlit as st
    
    # Check if we have sensor data in session state and fetch it if not available
    if 'latest_data' not in st.session_state or st.session_state.latest_data.empty:
        st.info("Automatically fetching sensor data...")
        # Fetch the latest data
        fetch_latest_data()
        # Prepare the data for visualization
        prepare_sensor_data_for_visualization()
        
        # Check again if data is available
        if 'latest_data' not in st.session_state or st.session_state.latest_data.empty:
            return None, [], "Error: Unable to automatically fetch sensor data. Please try again."
    
    # Get the data and make a copy to avoid modifying the original
    df = st.session_state.latest_data.copy()
    
    # Define common variable mappings
    variable_mappings = {
        "pm10": "P1",
        "pm2.5": "P2",
        "pm2_5": "P2",
        "temperature": "temperature",
        "humidity": "humidity",
        "pressure": "pressure",
        "p1": "P1",
        "p2": "P2"
    }
    
    # Parse the variables to plot
    if variables_to_plot.lower() == "all":
        # Use all available variables
        mapped_vars = df["value_type"].unique().tolist()
    else:
        requested_vars = [v.strip().lower() for v in variables_to_plot.split(',')]
        
        # Map user-friendly names to actual value_types
        mapped_vars = []
        for var in requested_vars:
            if var in variable_mappings:
                mapped_vars.append(variable_mappings[var])
            else:
                # Try to find a close match
                for available_var in df['value_type'].unique():
                    if var in available_var.lower() or available_var.lower() in var:
                        mapped_vars.append(available_var)
                        break
    
    # Limit to 5 variables for readability
    mapped_vars = mapped_vars[:5]
    
    if not mapped_vars:
        available_vars = df['value_type'].unique().tolist()
        error_msg = f"Error: Could not map requested variables to available value types. Available: {', '.join(available_vars)}"
        return df, [], error_msg
    
    return df, mapped_vars, None

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
    import streamlit as st
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Use the helper function to prepare data
    df, mapped_vars, error_msg = _prepare_sensor_data(variables_to_plot)
    
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
    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import pearsonr
    
    # Use the helper function to prepare data
    df, _, error_msg = _prepare_sensor_data("all")
    
    if error_msg:
        st.warning(error_msg)
        return error_msg
    
    # Map variables using a simpler approach
    available_vars = df['value_type'].unique().tolist()
    x_var = None
    y_var = None
    
    # Try to find matches for x and y variables
    for var in available_vars:
        if x_variable.lower() in var.lower() or var.lower() in x_variable.lower():
            x_var = var
        if y_variable.lower() in var.lower() or var.lower() in y_variable.lower():
            y_var = var
    
    # Default to P1 and P2 if available
    if not x_var and "P1" in available_vars:
        x_var = "P1"
    if not y_var and "P2" in available_vars:
        y_var = "P2"
    
    if not x_var or not y_var:
        error = f"Error: Could not map requested variables to available value types. Available: {', '.join(available_vars)}"
        st.warning(error)
        return error
    
    # Filter data for each variable
    x_data = df[df['value_type'] == x_var]['value'].reset_index(drop=True)
    y_data = df[df['value_type'] == y_var]['value'].reset_index(drop=True)
    
    # If the datasets have different lengths, take the shorter one
    min_length = min(len(x_data), len(y_data))
    x_data = x_data[:min_length]
    y_data = y_data[:min_length]
    
    if min_length < 2:
        return f"Error: Not enough data points for variables {x_var} and {y_var} to create a scatter plot."
    
    # Create a dataframe for the scatter plot
    scatter_df = pd.DataFrame({
        x_var: x_data,
        y_var: y_data
    })
    
    # Create the scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot with regression line
    sns.regplot(x=x_var, y=y_var, data=scatter_df, scatter_kws={'alpha':0.5}, line_kws={'color':'red'}, ax=ax)
    
    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel(f"{x_var} Value")
    ax.set_ylabel(f"{y_var} Value")
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Calculate correlation
    corr, p_value = pearsonr(x_data, y_data)
    
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
- **Sample Size:** {min_length} data points

### Interpretation:
{relationship}
"""


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
    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Use the helper function to prepare data
    df, mapped_vars, error_msg = _prepare_sensor_data(variables_to_plot)
    
    if error_msg:
        st.warning(error_msg)
        return error_msg
    
    # Ensure we have at least 2 variables
    if len(mapped_vars) < 2:
        available_vars = df['value_type'].unique().tolist()
        
        # If we have at least 2 available variables, use them instead
        if len(available_vars) >= 2:
            mapped_vars = available_vars[:min(5, len(available_vars))]
            st.info(f"Need at least 2 variables for a correlation heatmap. Using these variables instead: {', '.join(mapped_vars)}")
        else:
            return "Error: Not enough variables available for a correlation heatmap."
    
    # Limit to 5 variables for readability
    mapped_vars = mapped_vars[:5]
    
    # Create a proper pivot table with one column per variable
    # First, filter to only include the mapped variables
    filtered_df = df[df['value_type'].isin(mapped_vars)].copy()
    
    # Create a pivot table based on available columns
    pivot_data = {}
    
    # Check if we have both sensor_id and timestamp columns for proper alignment
    if 'sensor_id' in filtered_df.columns and 'timestamp' in filtered_df.columns:
        # Create a proper pivot table using sensor_id and timestamp as index
        pivot_df = filtered_df.pivot_table(
            index=['sensor_id', 'timestamp'],
            columns='value_type',
            values='value',
            aggfunc='mean'  # In case there are duplicates
        ).reset_index()
        
        # Drop the index columns for correlation calculation
        pivot_data = pivot_df.drop(columns=['sensor_id', 'timestamp'])
    else:
        # Fallback to a simpler method if we don't have the required columns
        for var in mapped_vars:
            var_data = df[df['value_type'] == var]['value'].reset_index(drop=True)
            if not var_data.empty:
                pivot_data[var] = var_data
        pivot_data = pd.DataFrame(pivot_data)
    
    # Ensure we have data
    if pivot_data.empty:
        return "Error: No data available for the requested variables after filtering."
    
    # Drop rows with NaN values
    pivot_data = pivot_data.dropna()
    
    if len(pivot_data) < 2:
        return "Error: Not enough data points after filtering for NaN values."
    
    # Calculate correlation matrix
    corr_matrix = pivot_data.corr()
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr_matrix, annot=True, fmt=".4f", cmap=cmap, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
    
    # Set title
    ax.set_title(title, fontsize=16)
    
    # Display the plot
    st.pyplot(fig)
    
    # Find strongest correlations
    # Get the upper triangle of the correlation matrix (excluding diagonal)
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Convert to 1D series and sort by absolute value
    sorted_corrs = upper_tri.unstack().dropna().sort_values(key=abs, ascending=False)
    
    # Get top 3 correlations
    top_corrs = sorted_corrs[:3]
    
    # Generate insights
    insights = ["## Correlation Insights\n"]
    
    if not top_corrs.empty:
        insights.append("**Strongest correlations:**")
        for (var1, var2), corr in top_corrs.items():
            corr_strength = "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.3 else "weak"
            corr_direction = "positive" if corr > 0 else "negative"
            insights.append(f"- {var1} and {var2}: {corr:.4f} ({corr_strength} {corr_direction})")
    
    return "\n".join(insights)

def get_sensor_tools() -> List[StructuredTool]:
    """
    Get the list of sensor data tools focused on reliable SDS011 dust sensor data.
    
    Returns:
        List[StructuredTool]: A list of sensor data tool objects
    """
    return [
        StructuredTool.from_function(
            func=fetch_latest_data,
            name="fetch_latest_sensor_data",
            description="Fetch the latest data from sensor.community API, focusing on SDS011 dust sensors measuring PM10 (P1) and PM2.5 (P2) levels.",
            # Don't use args_schema to avoid parameter handling issues
            return_direct=False
        ),
        StructuredTool.from_function(
            func=prepare_sensor_data_for_visualization,
            name="prepare_sensor_data",
            description="Prepare the latest fetched sensor data for visualization with the plotting tools. Use this after fetch_latest_sensor_data and before using any visualization tools.",
            return_direct=False
        ),
        StructuredTool.from_function(
            func=plot_sensor_data,
            name="plot_sensor_data",
            description="Create a time series plot of sensor data with sensible defaults. This is a specialized function for visualizing sensor data that automatically handles datetime formatting and grouping by value_type. Specify variables_to_plot to limit which variables are shown (max 5). Note: The sensor.community API only provides recent data (last 5 minutes).",
            return_direct=False
        ),
        # Specialized visualization tools for sensor data
        StructuredTool.from_function(
            func=plot_sensor_histogram,
            name="plot_sensor_histogram",
            description="Create histograms for sensor data variables like PM10 (P1) and PM2.5 (P2) to visualize their distributions.",
            return_direct=False
        ),
        StructuredTool.from_function(
            func=plot_sensor_scatter,
            name="plot_sensor_scatter",
            description="Create a scatter plot comparing two sensor variables (e.g., PM10 vs PM2.5) to visualize their relationship.",
            return_direct=False
        ),
        StructuredTool.from_function(
            func=plot_sensor_heatmap,
            name="plot_sensor_heatmap",
            description="Create a correlation heatmap for sensor variables to visualize relationships between different measurements.",
            return_direct=False
        ),
        StructuredTool.from_function(
            func=get_sensor_types,
            name="get_available_sensor_types",
            description="Get a list of available sensor types in the sensor.community network, with SDS011 being the most reliable for dust measurements.",
            return_direct=False
        ),
    ]