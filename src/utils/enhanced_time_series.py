"""
Enhanced time series visualization utilities for sensor.community data.

This module provides advanced time series visualization capabilities that combine
the strengths of Streamlit for interactivity, Matplotlib for customization,
and Seaborn for statistical insight.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from typing import Dict, Any, Optional, Union, List, Tuple
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.axes import Axes

# Set the default style for all visualizations
sns.set_theme(style="whitegrid")

def prepare_time_series_data(
    df: pd.DataFrame,
    value_column: str,
    time_column: str = 'timestamp',
    group_by: Optional[str] = None,
    resample_freq: Optional[str] = None,
    rolling_window: int = 1,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Prepare time series data for visualization by applying filters,
    resampling, and rolling averages.
    
    Args:
        df: DataFrame containing the data
        value_column: Column containing the values to plot
        time_column: Column containing the timestamps (default: 'timestamp')
        group_by: Optional column to group by
        resample_freq: Optional frequency for resampling
        rolling_window: Window size for rolling average calculation
        start_time: Optional start time for filtering
        end_time: Optional end time for filtering
        
    Returns:
        Processed DataFrame ready for visualization
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Convert object columns to strings to avoid PyArrow serialization issues
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str)
    
    # Ensure the time column is datetime
    df[time_column] = pd.to_datetime(df[time_column], errors='coerce')
    
    # Filter by time range if provided
    if start_time:
        df = df[df[time_column] >= start_time]
    if end_time:
        df = df[df[time_column] <= end_time]
    
    # Ensure value column is numeric
    df[value_column] = pd.to_numeric(df[value_column], errors='coerce')
    
    # Group and process data
    if group_by and group_by in df.columns:
        # Create a list to store processed group dataframes
        processed_groups = []
        
        # Get unique groups
        groups = df[group_by].unique()
        
        for group in groups:
            group_data = df[df[group_by] == group].copy()
            
            # Sort by timestamp
            group_data = group_data.sort_values(by=time_column)
            
            # Resample if specified
            if resample_freq:
                group_data = group_data.set_index(time_column)
                group_data = group_data.resample(resample_freq)[value_column].mean().reset_index()
            
            # Apply rolling average if window > 1
            if rolling_window > 1 and len(group_data) > rolling_window:
                group_data['rolling_avg'] = group_data[value_column].rolling(window=rolling_window, center=True).mean()
                # For the first and last (window//2) points, use the original values
                window_half = rolling_window // 2
                group_data['rolling_avg'].iloc[:window_half] = group_data[value_column].iloc[:window_half]
                group_data['rolling_avg'].iloc[-window_half:] = group_data[value_column].iloc[-window_half:]
                group_data[value_column] = group_data['rolling_avg']
                group_data = group_data.drop('rolling_avg', axis=1)
            
            # Add group identifier
            group_data[group_by] = group
            
            processed_groups.append(group_data)
        
        # Combine all processed groups
        if processed_groups:
            df = pd.concat(processed_groups, ignore_index=True)
        else:
            return pd.DataFrame()  # Return empty DataFrame if no groups were processed
    else:
        # Sort by timestamp
        df = df.sort_values(by=time_column)
        
        # Resample if specified
        if resample_freq:
            df = df.set_index(time_column)
            df = df.resample(resample_freq)[value_column].mean().reset_index()
        
        # Apply rolling average if window > 1
        if rolling_window > 1 and len(df) > rolling_window:
            df['rolling_avg'] = df[value_column].rolling(window=rolling_window, center=True).mean()
            # For the first and last (window//2) points, use the original values
            window_half = rolling_window // 2
            df['rolling_avg'].iloc[:window_half] = df[value_column].iloc[:window_half]
            df['rolling_avg'].iloc[-window_half:] = df[value_column].iloc[-window_half:]
            df[value_column] = df['rolling_avg']
            df = df.drop('rolling_avg', axis=1)
    
    return df

def create_enhanced_time_series_plot(
    df: pd.DataFrame,
    value_column: str,
    time_column: str = 'timestamp',
    group_by: Optional[str] = None,
    title: str = 'Time Series Plot',
    plot_type: str = 'Line Plot',
    color_palette: str = 'viridis',
    plot_height: int = 400,
    show_grid: bool = True,
    show_confidence: bool = False,
    threshold_values: Optional[Dict[str, float]] = None
) -> Figure:
    """
    Create an enhanced time series plot with advanced formatting and styling.
    
    Args:
        df: DataFrame containing the data
        value_column: Column containing the values to plot
        time_column: Column containing the timestamps (default: 'timestamp')
        group_by: Optional column to group by
        title: Plot title
        plot_type: Type of plot ('Line Plot', 'Area Chart', 'Bar Chart', 'Scatter Plot')
        color_palette: Seaborn color palette
        plot_height: Height of the plot in pixels
        show_grid: Whether to show grid lines
        show_confidence: Whether to show confidence intervals
        threshold_values: Optional dictionary of threshold values to display as horizontal lines
        
    Returns:
        Matplotlib Figure object
    """
    # Set up the figure with an appropriate aspect ratio
    fig_width = 10
    fig_height = plot_height / 100
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Set the color palette
    sns.set_palette(color_palette)
    
    # Configure the grid
    ax.grid(show_grid, alpha=0.3)
    
    # If grouping is specified
    if group_by and group_by in df.columns:
        # Get unique groups
        groups = df[group_by].unique()
        
        for i, group in enumerate(groups):
            group_data = df[df[group_by] == group].copy()
            
            # Sort by timestamp
            group_data = group_data.sort_values(by=time_column)
            
            color = sns.color_palette(color_palette, n_colors=len(groups))[i]
            
            # Create the specified plot type
            if plot_type == 'Line Plot':
                if show_confidence:
                    # Calculate confidence interval (using standard error)
                    std_err = group_data[value_column].std() / np.sqrt(len(group_data))
                    ci_low = group_data[value_column] - 1.96 * std_err
                    ci_high = group_data[value_column] + 1.96 * std_err
                    
                    # Plot confidence interval
                    ax.fill_between(
                        group_data[time_column], 
                        ci_low, 
                        ci_high, 
                        alpha=0.2, 
                        color=color
                    )
                
                # Plot the line
                ax.plot(
                    group_data[time_column], 
                    group_data[value_column], 
                    marker='o', 
                    linestyle='-', 
                    label=str(group),
                    alpha=0.8,
                    markersize=5,
                    color=color
                )
            
            elif plot_type == 'Area Chart':
                ax.fill_between(
                    group_data[time_column], 
                    0, 
                    group_data[value_column], 
                    alpha=0.6, 
                    label=str(group),
                    color=color
                )
                
            elif plot_type == 'Bar Chart':
                # For bar charts with grouped data, we need to handle differently
                bar_width = 0.8 / len(groups)
                offset = (i - len(groups)/2 + 0.5) * bar_width
                
                ax.bar(
                    [pd.Timestamp(x).timestamp() + offset for x in group_data[time_column]], 
                    group_data[value_column], 
                    width=bar_width,
                    label=str(group),
                    alpha=0.8,
                    color=color
                )
                
                # Custom x-axis formatting for bar charts
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
                
            elif plot_type == 'Scatter Plot':
                ax.scatter(
                    group_data[time_column], 
                    group_data[value_column], 
                    label=str(group),
                    alpha=0.7,
                    s=50,
                    color=color
                )
        
        plt.legend(title=group_by)
    else:
        # Sort by timestamp
        plot_df = df.sort_values(by=time_column).copy()
        
        color = sns.color_palette(color_palette)[0]
        
        # Create the specified plot type
        if plot_type == 'Line Plot':
            if show_confidence:
                # Calculate confidence interval (using standard error)
                std_err = plot_df[value_column].std() / np.sqrt(len(plot_df))
                ci_low = plot_df[value_column] - 1.96 * std_err
                ci_high = plot_df[value_column] + 1.96 * std_err
                
                # Plot confidence interval
                ax.fill_between(
                    plot_df[time_column], 
                    ci_low, 
                    ci_high, 
                    alpha=0.2, 
                    color=color
                )
            
            # Plot the line
            ax.plot(
                plot_df[time_column], 
                plot_df[value_column], 
                marker='o', 
                linestyle='-',
                alpha=0.8,
                markersize=5,
                color=color
            )
            
        elif plot_type == 'Area Chart':
            ax.fill_between(
                plot_df[time_column], 
                0, 
                plot_df[value_column], 
                alpha=0.6,
                color=color
            )
            
        elif plot_type == 'Bar Chart':
            ax.bar(
                plot_df[time_column], 
                plot_df[value_column], 
                alpha=0.8,
                color=color
            )
            
        elif plot_type == 'Scatter Plot':
            ax.scatter(
                plot_df[time_column], 
                plot_df[value_column], 
                alpha=0.7,
                s=50,
                color=color
            )
    
    # Add threshold lines if provided
    if threshold_values:
        for label, value in threshold_values.items():
            ax.axhline(
                y=value, 
                color='red', 
                linestyle='--', 
                alpha=0.7, 
                label=label
            )
            # Add text label for the threshold
            ax.text(
                df[time_column].min() + (df[time_column].max() - df[time_column].min()) * 0.02, 
                value * 1.05, 
                label, 
                color='red',
                fontsize=9,
                alpha=0.8
            )
    
    # Set labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel(value_column)
    ax.set_title(title)
    
    # Format x-axis
    plt.xticks(rotation=45)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    
    # Format y-axis to avoid scientific notation
    ax.ticklabel_format(useOffset=False, style='plain', axis='y')
    
    # Add annotations for min and max values
    if not df.empty:
        # Find min and max points
        idx_max = df[value_column].idxmax()
        idx_min = df[value_column].idxmin()
        
        # Get the timestamps and values
        if idx_max is not None and idx_min is not None:
            max_point = (df.loc[idx_max, time_column], df.loc[idx_max, value_column])
            min_point = (df.loc[idx_min, time_column], df.loc[idx_min, value_column])
            
            # Annotate max point
            ax.annotate(
                f'Max: {max_point[1]:.2f}', 
                xy=max_point,
                xytext=(10, 10),
                textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.5),
                fontsize=9,
                color='darkred'
            )
            
            # Annotate min point
            ax.annotate(
                f'Min: {min_point[1]:.2f}', 
                xy=min_point,
                xytext=(10, -15),
                textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.5),
                fontsize=9,
                color='darkblue'
            )
    
    plt.tight_layout()
    
    return fig

def display_enhanced_time_series(
    df: pd.DataFrame,
    value_column: str,
    time_column: str = 'timestamp',
    group_by: Optional[str] = None,
    title: str = 'Time Series Plot',
    plot_type: str = 'Line Plot',
    color_palette: str = 'viridis',
    plot_height: int = 400,
    show_grid: bool = True,
    show_confidence: bool = False,
    threshold_values: Optional[Dict[str, float]] = None,
    show_statistics: bool = True
) -> None:
    """
    Display an enhanced time series visualization in Streamlit with statistics.
    
    Args:
        df: DataFrame containing the data
        value_column: Column containing the values to plot
        time_column: Column containing the timestamps (default: 'timestamp')
        group_by: Optional column to group by
        title: Plot title
        plot_type: Type of plot ('Line Plot', 'Area Chart', 'Bar Chart', 'Scatter Plot')
        color_palette: Seaborn color palette
        plot_height: Height of the plot in pixels
        show_grid: Whether to show grid lines
        show_confidence: Whether to show confidence intervals
        threshold_values: Optional dictionary of threshold values to display as horizontal lines
        show_statistics: Whether to show statistical summary
    """
    # Check if DataFrame is empty
    if df.empty:
        st.error("No data available for time series visualization.")
        return
    
    # Ensure time column is present
    if time_column not in df.columns:
        st.error(f"Time column '{time_column}' not found in the data.")
        return
    
    # Ensure value column is present
    if value_column not in df.columns:
        st.error(f"Value column '{value_column}' not found in the data.")
        return
    
    # Create and display the figure
    try:
        fig = create_enhanced_time_series_plot(
            df,
            value_column,
            time_column,
            group_by,
            title,
            plot_type,
            color_palette,
            plot_height,
            show_grid,
            show_confidence,
            threshold_values
        )
        
        # Display the plot
        st.pyplot(fig)
        
        # Close the figure to free memory
        plt.close(fig)
        
        # Show statistics if requested
        if show_statistics:
            st.subheader("Summary Statistics")
            
            if group_by and group_by in df.columns:
                # Create a tab for each group
                groups = df[group_by].unique()
                tabs = st.tabs([f"Group: {group}" for group in groups])
                
                for i, group in enumerate(groups):
                    group_data = df[df[group_by] == group][value_column]
                    
                    with tabs[i]:
                        display_statistics(group_data)
            else:
                display_statistics(df[value_column])
        
    except Exception as e:
        st.error(f"Error displaying enhanced time series plot: {str(e)}")
        import traceback
        print(f"Error in display_enhanced_time_series: {str(e)}")
        print(traceback.format_exc())

def display_statistics(series: pd.Series) -> None:
    """
    Display detailed statistics for a pandas Series.
    
    Args:
        series: The pandas Series containing the data to analyze
    """
    # Convert series to numeric
    series = pd.to_numeric(series, errors='coerce')
    
    # Create basic statistics
    stats = pd.DataFrame({
        'Statistic': ['Count', 'Mean', 'Median', 'Min', 'Max', 'Std Dev', 'Skewness', 'Kurtosis'],
        'Value': [
            series.count(),
            f"{series.mean():.2f}",
            f"{series.median():.2f}",
            f"{series.min():.2f}",
            f"{series.max():.2f}",
            f"{series.std():.2f}",
            f"{series.skew():.2f}",
            f"{series.kurtosis():.2f}"
        ]
    })
    
    # Display the statistics table
    st.dataframe(stats)
    
    # Create a row with two columns for visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Create distribution plot
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(series.dropna(), kde=True, ax=ax)
        ax.set_title('Distribution')
        st.pyplot(fig)
        plt.close(fig)
    
    with col2:
        # Create box plot
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x=series.dropna(), ax=ax)
        ax.set_title('Box Plot')
        st.pyplot(fig)
        plt.close(fig)
    
    # Calculate and display percentiles
    percentiles = [0, 25, 50, 75, 90, 95, 99, 100]
    percentile_values = [np.percentile(series.dropna(), p) for p in percentiles]
    
    percentile_df = pd.DataFrame({
        'Percentile': [f"{p}%" for p in percentiles],
        'Value': [f"{v:.2f}" for v in percentile_values]
    })
    
    st.subheader("Percentiles")
    st.dataframe(percentile_df)

def create_pm_comparison_plot(
    df: pd.DataFrame,
    time_period: str = '24h',
    resample_freq: str = '1H'
) -> None:
    """
    Create a specialized visualization comparing PM10 and PM2.5 levels over time.
    This function uses the enhanced plotting capabilities for better visualization.
    
    Args:
        df: DataFrame containing pollution data
        time_period: Time period to analyze ('24h', '7d', '30d')
        resample_freq: Frequency for resampling time series data
    """
    if df.empty:
        st.error("No data available for comparison.")
        return
    
    # Ensure timestamp column is datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
    else:
        st.error("Timestamp column not found in data.")
        return
    
    # Filter for recent data based on time_period
    now = datetime.now()
    if time_period == '24h':
        start_time = now - timedelta(hours=24)
        period_name = "Last 24 Hours"
    elif time_period == '7d':
        start_time = now - timedelta(days=7)
        period_name = "Last 7 Days"
    elif time_period == '30d':
        start_time = now - timedelta(days=30)
        period_name = "Last 30 Days"
    else:
        start_time = now - timedelta(hours=24)  # Default to 24 hours
        period_name = "Last 24 Hours"
    
    df = df[df['timestamp'] >= start_time]
    
    if df.empty:
        st.error(f"No data available for the selected time period ({period_name}).")
        return
    
    # Filter for PM10 (P1) and PM2.5 (P2) data
    pm10_data = df[df['value_type'] == 'P1'].copy()
    pm25_data = df[df['value_type'] == 'P2'].copy()
    
    if pm10_data.empty or pm25_data.empty:
        st.error("Missing either PM10 or PM2.5 data for comparison.")
        return
    
    # Prepare both datasets using the enhanced data preparation function
    pm10_data = prepare_time_series_data(
        pm10_data,
        value_column='value',
        time_column='timestamp',
        resample_freq=resample_freq
    )
    
    pm25_data = prepare_time_series_data(
        pm25_data,
        value_column='value',
        time_column='timestamp',
        resample_freq=resample_freq
    )
    
    # Create the enhanced visualizations
    st.subheader(f"PM10 Levels - {period_name}")
    
    # Define PM10 thresholds for the plot
    pm10_thresholds = {
        "EU Daily Limit": 50,
        "WHO Guideline": 20
    }
    
    # Display PM10 plot with thresholds
    display_enhanced_time_series(
        pm10_data,
        value_column='value',
        time_column='timestamp',
        title=f'PM10 Levels - {period_name}',
        plot_type='Line Plot',
        color_palette='Reds',
        threshold_values=pm10_thresholds,
        show_confidence=True
    )
    
    st.subheader(f"PM2.5 Levels - {period_name}")
    
    # Define PM2.5 thresholds for the plot
    pm25_thresholds = {
        "EU Daily Limit": 25,
        "WHO Guideline": 10
    }
    
    # Display PM2.5 plot with thresholds
    display_enhanced_time_series(
        pm25_data,
        value_column='value',
        time_column='timestamp',
        title=f'PM2.5 Levels - {period_name}',
        plot_type='Line Plot',
        color_palette='Purples',
        threshold_values=pm25_thresholds,
        show_confidence=True
    )
    
    # Calculate correlation between PM10 and PM2.5
    # Merge the resampled data on timestamp
    merged_df = pd.merge(
        pm10_data, 
        pm25_data, 
        on='timestamp', 
        suffixes=('_pm10', '_pm25')
    )
    
    if merged_df.empty:
        st.warning("Unable to calculate correlation: no matching timestamps between PM10 and PM2.5 data.")
    else:
        correlation = merged_df['value_pm10'].corr(merged_df['value_pm25'])
        
        # Create a scatter plot with regression line to show the relationship
        st.subheader("PM2.5 vs PM10 Correlation")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.regplot(
            x='value_pm10', 
            y='value_pm25', 
            data=merged_df, 
            ax=ax, 
            scatter_kws={'alpha': 0.5}
        )
        ax.set_xlabel('PM10 (µg/m³)')
        ax.set_ylabel('PM2.5 (µg/m³)')
        ax.set_title(f'PM2.5 vs PM10 Correlation (r = {correlation:.2f})')
        ax.grid(True, alpha=0.3)
        
        # Display the correlation plot
        st.pyplot(fig)
        plt.close(fig)
        
        # Calculate PM2.5/PM10 ratio
        ratio = pm25_data['value'].mean() / pm10_data['value'].mean()
        
        # Create an info box with the ratio and explanation
        st.info(f"""
        **Average PM2.5/PM10 Ratio: {ratio:.2f}**
        
        - Typical urban ratios range from 0.5 to 0.8
        - Higher ratios (>0.6) often indicate more combustion-related pollution sources
        - Lower ratios (<0.5) may suggest more dust and mechanical sources of particles
        """)
        
        # Display comprehensive statistics comparing both pollutants
        st.subheader("Comparison Statistics")
        
        stats_df = pd.DataFrame({
            'Statistic': ['Mean', 'Median', 'Max', 'Min', 'Std Dev', 'Correlation'],
            'PM10 (µg/m³)': [
                f"{pm10_data['value'].mean():.2f}",
                f"{pm10_data['value'].median():.2f}",
                f"{pm10_data['value'].max():.2f}",
                f"{pm10_data['value'].min():.2f}",
                f"{pm10_data['value'].std():.2f}",
                f"{correlation:.2f}"
            ],
            'PM2.5 (µg/m³)': [
                f"{pm25_data['value'].mean():.2f}",
                f"{pm25_data['value'].median():.2f}",
                f"{pm25_data['value'].max():.2f}",
                f"{pm25_data['value'].min():.2f}",
                f"{pm25_data['value'].std():.2f}",
                ""
            ]
        })
        
        st.table(stats_df)
