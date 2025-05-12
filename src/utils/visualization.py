"""
Data visualization utilities for Streamlit and Seaborn integration.

This module provides functions for creating common data visualizations 
with Seaborn and displaying them in Streamlit.
"""

from typing import Any, Dict, List, Optional
from matplotlib.ticker import MaxNLocator

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

# Store the loaded dataset in session state
if "dataset" not in st.session_state:
    st.session_state.dataset = None
    
# Store the last time sensor data was fetched
if "last_sensor_data_fetch" not in st.session_state:
    st.session_state.last_sensor_data_fetch = None

def load_dataset(file_path: str) -> str:
    """
    Load a dataset from a CSV file or use sensor data from session state.
    
    Args:
        file_path (str): Path to the CSV file or special keyword for session data
        
    Returns:
        str: Information about the loaded dataset
    """
    try:
        # Clean the file path
        if isinstance(file_path, str):
            file_path = clean_column_name(file_path)
        
        # Special case for sensor data
        if file_path.lower() in ['sensor_data', 'sensor', 'latest_sensor_data', 'latest_sds011_sensor_data.csv', 'sds011']:
            # Check if we have sensor data in session state
            if 'latest_data' in st.session_state and not st.session_state.latest_data.empty:
                df = st.session_state.latest_data
                # Store in dataset session state for visualization tools
                st.session_state.dataset = df
                source = "sensor.community API"
            else:
                return "Error: No sensor data available. Please fetch sensor data first using fetch_latest_sensor_data."
        elif not file_path.endswith('.csv'):
            return "Error: Only CSV files are supported for file-based visualization. For sensor data, use 'sensor_data' as the file path."
        else:
            # Load the dataset from CSV file
            df = pd.read_csv(file_path)
            # Store in session state
            st.session_state.dataset = df
            source = file_path
        
        # Return information about the dataset
        return f"""
## Dataset Loaded Successfully

**Source:** {source}
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

def plot_histogram(column: Optional[str] = None, bins: int = 10, title: str = "Histogram") -> str:
    """
    Create a histogram using Seaborn and display it in Streamlit.
    
    Args:
        column (Optional[str]): Column to plot. If None, will auto-select first numeric column.
        bins (int): Number of bins
        title (str): Plot title
        
    Returns:
        str: Result message
    """
    if st.session_state.dataset is None:
        return "Error: No dataset loaded. Please load a dataset first using load_dataset."
    
    try:
        df = st.session_state.dataset
        
        # Get all numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_cols:
            return "Error: The dataset has no numeric columns to plot a histogram with."
        
        # Display info to the user about available columns
        st.info(f"Available numeric columns for histogram: {', '.join(numeric_cols)}")
        
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
        
        # If no column specified, use the first numeric column
        if column is None:
            column = numeric_cols[0]
            st.info(f"Auto-selected numeric column: {column}")
        else:
            # Clean the column name
            column = clean_column_name(column)
        
        # Check if column exists
        if column not in df.columns:
            st.warning(f"Column '{column}' not found in dataset. Auto-selecting first numeric column.")
            column = numeric_cols[0]
        
        # Check if column is numeric
        if column not in numeric_cols:
            st.warning(f"Column '{column}' is not numeric. Auto-selecting first numeric column for histogram.")
            column = numeric_cols[0]
        
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
        import traceback
        st.error(f"Error creating histogram: {str(e)}")
        st.error(traceback.format_exc())
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

def plot_scatter(x_column: Optional[str] = None, y_column: Optional[str] = None, hue_column: Optional[str] = None, 
                 title: str = "Scatter Plot") -> str:
    """
    Create a scatter plot using Seaborn and display it in Streamlit.
    
    Args:
        x_column (str, optional): Column for x-axis or JSON with parameters. If None, will auto-select.
        y_column (str, optional): Column for y-axis. If None, will auto-select.
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
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_cols or len(numeric_cols) < 1:
            return "Error: The dataset needs at least one numeric column for a scatter plot."
            
        # Display info to the user about available columns
        st.info(f"Available numeric columns: {', '.join(numeric_cols)}")
        
        # Clean column names if provided
        if x_column:
            x_column = clean_column_name(x_column)
        if y_column:
            y_column = clean_column_name(y_column)
        if hue_column is not None:
            hue_column = clean_column_name(hue_column)
            
        # Auto-select numeric columns if needed
        if x_column is None or x_column == "":
            # Auto-select first numeric column
            x_column = numeric_cols[0]
            st.info(f"Auto-selected '{x_column}' for x-axis")
        elif x_column not in df.columns:
            st.warning(f"Column '{x_column}' not found. Auto-selecting first numeric column.")
            x_column = numeric_cols[0]
        elif x_column not in numeric_cols:
            st.warning(f"Column '{x_column}' is not numeric. Auto-selecting first numeric column.")
            x_column = numeric_cols[0]
            
        if y_column is None or y_column == "":
            # Auto-select second numeric column (or first if only one is available)
            if len(numeric_cols) > 1:
                # Try to select a different column than x_column
                if x_column == numeric_cols[0]:
                    y_column = numeric_cols[1]
                else:
                    y_column = numeric_cols[0]
            else:
                # If only one numeric column, use it for both axes
                y_column = numeric_cols[0]
            st.info(f"Auto-selected '{y_column}' for y-axis")
        elif y_column not in df.columns:
            st.warning(f"Column '{y_column}' not found. Auto-selecting an appropriate numeric column.")
            if len(numeric_cols) > 1 and numeric_cols[0] == x_column:
                y_column = numeric_cols[1]
            else:
                y_column = numeric_cols[0] 
        elif y_column not in numeric_cols:
            st.warning(f"Column '{y_column}' is not numeric. Auto-selecting an appropriate numeric column.")
            if len(numeric_cols) > 1 and numeric_cols[0] == x_column:
                y_column = numeric_cols[1]
            else:
                y_column = numeric_cols[0]
        
        # Validate hue column if provided
        if hue_column is not None:
            if hue_column not in df.columns:
                st.warning(f"Hue column '{hue_column}' not found. Not using color grouping.")
                hue_column = None
        
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
        import traceback
        st.error(f"Error creating scatter plot: {str(e)}")
        st.error(traceback.format_exc())
        return f"Error creating scatter plot: {str(e)}"

def get_column_types(columns: str | None = None) -> str:
    """
    Get the data types of columns in the dataset.
    
    Args:
        columns (str, optional): Comma-separated list of columns to check.
                                If None, checks all columns.
    
    Returns:
        str: Information about column data types and basic statistics
    """
    if st.session_state.dataset is None:
        return "Error: No dataset loaded. Please load a dataset first using load_dataset."
    
    try:
        df = st.session_state.dataset
        
        # Determine columns to check
        if columns is not None:
            # Handle special cases for malformed inputs
            try:
                # Remove extra quotes if present
                columns = columns.strip('"\'')
                
                # If there are trailing quotes, strip them too
                if columns.endswith('"') or columns.endswith("'"):
                    columns = columns[:-1]
                
                # Clean and split columns
                col_list = [clean_column_name(col) for col in columns.split(',')]
                
                # Remove any empty strings
                col_list = [col for col in col_list if col]
                
                # If we end up with no columns, use all
                if not col_list:
                    col_list = df.columns.tolist()
            except Exception as e:
                # If any error occurs in parsing, default to all columns
                st.warning(f"Error parsing column list: {str(e)}. Using all columns instead.")
                col_list = df.columns.tolist()
        else:
            col_list = df.columns.tolist()
        
        # Validate columns - ensure they exist in dataset
        valid_cols = [col for col in col_list if col in df.columns]
        
        # If no valid columns, show an error
        if not valid_cols:
            return f"Error: None of the specified columns were found in the dataset. Available columns: {', '.join(df.columns)}"
        
        # If some columns were invalid, show a warning
        invalid_cols = [col for col in col_list if col not in df.columns]
        if invalid_cols:
            st.warning(f"Some columns were not found: {', '.join(invalid_cols)}")
        
        # Build information about each column
        info = ["## Column Information\n"]
        
        for col in valid_cols:
            dtype = df[col].dtype
            is_numeric = pd.api.types.is_numeric_dtype(dtype)
            n_unique = df[col].nunique()
            n_missing = df[col].isna().sum()
            
            info.append(f"### {col}")
            info.append(f"- **Type:** {dtype}")
            info.append(f"- **Unique Values:** {n_unique}")
            info.append(f"- **Missing Values:** {n_missing}")
            
            # Add sample values with error handling
            try:
                sample_values = df[col].dropna().head(3).tolist()
                # Convert all sample values to strings to ensure they can be displayed
                sample_values = [str(val)[:50] for val in sample_values]
                info.append(f"- Sample Values: {sample_values}\n")
            except Exception as e:
                info.append(f"- Sample Values: Error getting samples - {str(e)}\n")
            
            # Add statistics for numeric columns with error handling
            if is_numeric:
                try:
                    stats = df[col].describe()
                    info.append("**Statistics:**")
                    info.append(f"- Mean: {stats.get('mean', 'N/A')}")
                    info.append(f"- Std Dev: {stats.get('std', 'N/A')}")
                    info.append(f"- Min: {stats.get('min', 'N/A')}")
                    info.append(f"- Max: {stats.get('max', 'N/A')}\n")
                except Exception as e:
                    info.append(f"**Statistics:** Error calculating statistics - {str(e)}\n")
        
        return "\n".join(info)
    
    except Exception as e:
        return f"Error getting column types: {str(e)}"
        
def match_variables_with_llm(variables_to_plot: str, df: pd.DataFrame) -> list:
    """
    Use an LLM to understand which variables the user is asking for and match them with
    available variables in the dataset.
    
    Args:
        variables_to_plot (str): User's request for variables to plot
        df (pd.DataFrame): The sensor data DataFrame
        
    Returns:
        list: Matched variable names from the dataset
    """
    try:
        # Get available variables from the dataset
        available_vars = df['value_type'].unique().tolist()
        
        # Create a prompt for the LLM
        prompt = f"""
        I have sensor data with the following variable types: {', '.join(available_vars)}.
        
        The user has requested to plot the following variables: "{variables_to_plot}".
        
        Based on the user's request, which of the available variables should I use?
        Return only the exact variable names from the available list, separated by commas.
        """
        
        # For now, we'll use a simple rule-based approach as a placeholder for the LLM
        # In a real implementation, this would call an LLM API
        matched_vars = []
        
        # Common variable mappings
        common_mappings = {
            'pm10': 'P1',
            'pm2.5': 'P2',
            'pm2_5': 'P2',
            'pm 10': 'P1',
            'pm 2.5': 'P2',
            'particulate matter': ['P1', 'P2'],
            'dust': ['P1', 'P2'],
            'temperature': 'temperature',
            'humidity': 'humidity',
            'pressure': 'pressure',
            'air quality': ['P1', 'P2']
        }
        
        # Process user request
        user_request = variables_to_plot.lower()
        
        # Check for common terms in the user request
        for term, mapping in common_mappings.items():
            if term in user_request:
                if isinstance(mapping, list):
                    for var in mapping:
                        if var in available_vars and var not in matched_vars:
                            matched_vars.append(var)
                else:
                    if mapping in available_vars and mapping not in matched_vars:
                        matched_vars.append(mapping)
        
        # If no matches found, return empty list to fall back to basic matching
        return matched_vars
    except Exception as e:
        print(f"Error in LLM variable matching: {e}")
        return []


def prepare_sensor_data(variables_to_plot: str = "all", force_refresh: bool = False) -> tuple:
    """
    Prepare sensor data for visualization. This function handles the specific requirements
    of sensor data, including fetching from APIs, data transformation, and caching.
    
    Args:
        variables_to_plot (str): Comma-separated list of variables to plot (e.g., "P1,P2")
                               or "all" to include all available variables
        force_refresh (bool): Whether to force a refresh of the data from the API
                            even if it was recently fetched
        
    Returns:
        tuple: (DataFrame, list of mapped variables, error message or None)
    """
    import streamlit as st
    import pandas as pd
    import datetime
    
    # Check if we need to fetch new data
    current_time = datetime.datetime.now()
    data_is_stale = True
    
    if not force_refresh and 'latest_data' in st.session_state and not st.session_state.latest_data.empty:
        if st.session_state.last_sensor_data_fetch:
            # Data is stale if it's more than 5 minutes old
            time_diff = current_time - st.session_state.last_sensor_data_fetch
            if time_diff.total_seconds() < 300:  # 5 minutes
                data_is_stale = False
    
    # Fetch new data if needed
    if data_is_stale or force_refresh or 'latest_data' not in st.session_state or st.session_state.latest_data.empty:
        try:
            # This would normally call the API, but for now we'll use a placeholder
            st.info("Fetching fresh sensor data...")
            
            # In a real implementation, this would be a call to the sensor API
            # For now, we'll just check if there's already data in the session state
            if 'latest_data' not in st.session_state or st.session_state.latest_data.empty:
                return pd.DataFrame(), [], "Error: No sensor data available. Please fetch sensor data first."
            
            # Update the timestamp
            st.session_state.last_sensor_data_fetch = current_time
        except Exception as e:
            return pd.DataFrame(), [], f"Error fetching sensor data: {str(e)}"
    
    # Get the data from session state
    df = st.session_state.latest_data
    
    # Handle 'all' case
    if variables_to_plot.lower() == 'all':
        mapped_vars = df['value_type'].unique().tolist()
    else:
        # Use LLM-based variable matching
        mapped_vars = match_variables_with_llm(variables_to_plot, df)
        
        # If LLM matching fails, fall back to basic matching
        if not mapped_vars:
            # Split by comma and clean each variable name
            requested_vars = [v.strip() for v in variables_to_plot.split(',')]
            
            # Use the exact variable names from the data
            available_vars = df['value_type'].unique().tolist()
            mapped_vars = []
            
            for var in requested_vars:
                if var in available_vars:
                    # Exact match
                    mapped_vars.append(var)
                else:
                    # Case-insensitive match
                    for available_var in available_vars:
                        if var.lower() == available_var.lower():
                            mapped_vars.append(available_var)
                            break
    
    if not mapped_vars:
        available_vars = df['value_type'].unique().tolist()
        error_msg = f"Error: Could not map requested variables to available value types. Available: {', '.join(available_vars)}"
        return df, [], error_msg
    
    return df, mapped_vars, None


def create_sensor_pivot_table(df: pd.DataFrame, variables: list) -> pd.DataFrame:
    """
    Create a simple pivot table from sensor data for correlation analysis.
    
    Args:
        df (DataFrame): The sensor data DataFrame
        variables (list): List of variables to include in the pivot table
        
    Returns:
        DataFrame: Pivot table with sensor measurements
    """
    import streamlit as st
    
    # Filter to only include the requested variables
    filtered_df = df[df['value_type'].isin(variables)].copy()
    
    # Check how many unique sensors we have
    unique_sensors = filtered_df['sensor_id'].nunique()
    if unique_sensors < 2:
        print(f"Limited data diversity: Only {unique_sensors} unique sensors found. Correlations may not be representative.")
    
    # Create a simple pivot table with sensor_id as index
    pivot_df = filtered_df.pivot_table(
        index='sensor_id',
        columns='value_type',
        values='value',
        aggfunc='mean'  # Use mean for multiple readings from the same sensor
    ).reset_index()
    
    # Drop rows with NaN values in any of the requested variables
    pivot_df = pivot_df.dropna(subset=variables)
    
    # # Ensure we have enough variation in the data
    # for var in variables:
    #     if var in pivot_df.columns and pivot_df[var].nunique() < 3:
    #         st.warning(f"Limited variation in {var} values. Correlation results may not be meaningful.")
    
    return pivot_df


def plot_heatmap(columns: List[str] | str = "all", title: str = "Correlation Heatmap") -> str:
    """
    Create a correlation heatmap using Seaborn and display it in Streamlit.
    
    Args:
        columns (Union[List[str], str], optional): List of columns, comma-separated string of columns, or "all"
            If "all" (default), automatically uses all numeric columns.
        title (str): Plot title
        
    Returns:
        str: Result message
    """
    if st.session_state.dataset is None:
        return "Error: No dataset loaded. Please load a dataset first using load_dataset."
    
    try:
        df = st.session_state.dataset
        
        # Get all numeric columns in the dataset
        all_numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if len(all_numeric_cols) < 2:
            return "Error: The dataset needs at least two numeric columns for a correlation heatmap."
        
        # Display info to the user about available columns
        st.info(f"Available numeric columns: {', '.join(all_numeric_cols)}")
        
        # Convert input to list of columns
        if isinstance(columns, str):
            # Remove any quotes that might be present
            columns = columns.strip('"\'')
            
            # Handle 'all' case - use all numeric columns
            if columns.lower() == "all":
                columns = all_numeric_cols
                st.info("Using all numeric columns for the heatmap.")
            else:
                # Split by comma and clean each column name
                columns = [col.strip() for col in columns.split(',')]
        
        # Clean column names
        columns = [clean_column_name(col) for col in columns]
        
        # Check which columns exist and are numeric
        valid_columns = []
        invalid_columns = []
        non_numeric_columns = []
        
        for col in columns:
            if col not in df.columns:
                invalid_columns.append(col)
            elif col not in all_numeric_cols:
                non_numeric_columns.append(col)
            else:
                valid_columns.append(col)
        
        # Provide feedback about invalid or non-numeric columns
        if invalid_columns:
            st.warning(f"Columns not found in dataset: {', '.join(invalid_columns)}")
        
        if non_numeric_columns:
            st.warning(f"Non-numeric columns (excluded from heatmap): {', '.join(non_numeric_columns)}")
        
        # If no valid columns, use all numeric columns
        if not valid_columns:
            st.warning("No valid numeric columns specified. Using all numeric columns.")
            valid_columns = all_numeric_cols
        
        # Ensure we have at least 2 columns
        if len(valid_columns) < 2:
            st.warning("Not enough valid numeric columns. Adding more columns from the dataset.")
            # Add more numeric columns that weren't specified
            additional_cols = [col for col in all_numeric_cols if col not in valid_columns]
            valid_columns.extend(additional_cols[:2-len(valid_columns)])
        
        # Create correlation matrix
        corr_matrix = df[valid_columns].corr()
        
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
            (valid_columns[i], valid_columns[j], corr_matrix.iloc[i, j])
            for i in range(len(valid_columns))
            for j in range(i+1, len(valid_columns))
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
        import traceback
        st.error(f"Error creating heatmap: {str(e)}")
        st.error(traceback.format_exc())
        return f"Error creating heatmap: {str(e)}"
