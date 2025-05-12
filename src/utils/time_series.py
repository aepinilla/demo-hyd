"""
Time series visualization utilities for sensor data.

This module provides specialized functions for visualizing time-based sensor data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Union

def plot_time_series(df: pd.DataFrame, 
                    value_column: str,
                    time_column: str = 'timestamp',
                    group_by: Optional[str] = None,
                    title: str = 'Time Series Plot',
                    resample_freq: Optional[str] = None) -> None:
    """
    Create a time series line plot for sensor data.
    
    Args:
        df: DataFrame containing the sensor data
        value_column: Column containing the values to plot
        time_column: Column containing the timestamps (default: 'timestamp')
        group_by: Optional column to group by (e.g., 'sensor_id' or 'sensor_type')
        title: Plot title
        resample_freq: Optional frequency to resample data (e.g., '1H' for hourly)
    """
    # Check if DataFrame is empty
    if df.empty:
        st.error("No data available for time series visualization.")
        return
    
    # Ensure timestamp column is datetime
    if time_column in df.columns:
        df[time_column] = pd.to_datetime(df[time_column], errors='coerce')
        
        # Drop rows with invalid timestamps
        df = df.dropna(subset=[time_column])
        
        if df.empty:
            st.error("No valid timestamp data available for time series visualization.")
            return
    else:
        st.error(f"Time column '{time_column}' not found in the data.")
        return
    
    # Check if value column exists
    if value_column not in df.columns:
        st.error(f"Value column '{value_column}' not found in the data.")
        return
    
    # Convert value column to numeric
    df[value_column] = pd.to_numeric(df[value_column], errors='coerce')
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # If grouping is specified
    if group_by and group_by in df.columns:
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
            
            # Plot the line
            ax.plot(group_data[time_column], group_data[value_column], marker='o', linestyle='-', label=str(group))
        
        plt.legend(title=group_by)
    else:
        # Sort by timestamp
        plot_df = df.sort_values(by=time_column).copy()
        
        # Resample if specified
        if resample_freq:
            plot_df = plot_df.set_index(time_column)
            plot_df = plot_df.resample(resample_freq)[value_column].mean().reset_index()
        
        # Plot the line
        ax.plot(plot_df[time_column], plot_df[value_column], marker='o', linestyle='-')
    
    # Set labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel(value_column)
    ax.set_title(title)
    
    # Format x-axis
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Display the plot
    st.pyplot(fig)
    
    # Show summary statistics
    st.write("### Summary Statistics")
    stats = df[value_column].describe()
    st.write(stats)
    
    return

def compare_time_periods(df: pd.DataFrame,
                        value_column: str,
                        time_column: str = 'timestamp',
                        period1: str = 'morning',
                        period2: str = 'evening',
                        title: str = 'Time Period Comparison') -> None:
    """
    Compare sensor readings between two time periods (e.g., morning vs evening).
    
    Args:
        df: DataFrame containing the sensor data
        value_column: Column containing the values to compare
        time_column: Column containing the timestamps (default: 'timestamp')
        period1: Name of the first period (default: 'morning')
        period2: Name of the second period (default: 'evening')
        title: Plot title
    """
    # Check if DataFrame is empty
    if df.empty:
        st.error("No data available for time period comparison.")
        return
    
    # Ensure timestamp column is datetime
    if time_column in df.columns:
        df[time_column] = pd.to_datetime(df[time_column], errors='coerce')
        
        # Drop rows with invalid timestamps
        df = df.dropna(subset=[time_column])
        
        if df.empty:
            st.error("No valid timestamp data available for comparison.")
            return
    else:
        st.error(f"Time column '{time_column}' not found in the data.")
        return
    
    # Check if value column exists
    if value_column not in df.columns:
        st.error(f"Value column '{value_column}' not found in the data.")
        return
    
    # Convert value column to numeric
    df[value_column] = pd.to_numeric(df[value_column], errors='coerce')
    
    # Extract hour from timestamp
    df['hour'] = df[time_column].dt.hour
    
    # Define morning and evening periods
    morning_mask = (df['hour'] >= 6) & (df['hour'] < 12)
    evening_mask = (df['hour'] >= 18) & (df['hour'] < 24)
    
    morning_data = df[morning_mask][value_column]
    evening_data = df[evening_mask][value_column]
    
    if morning_data.empty or evening_data.empty:
        st.error("Not enough data for both time periods.")
        return
    
    # Create comparison visualizations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Box plot
    comparison_df = pd.DataFrame({
        period1: morning_data,
        period2: evening_data
    })
    
    sns.boxplot(data=comparison_df, ax=ax1)
    ax1.set_title(f"{value_column} by Time Period")
    ax1.set_ylabel(value_column)
    
    # Bar plot with means and error bars
    means = [morning_data.mean(), evening_data.mean()]
    stds = [morning_data.std(), evening_data.std()]
    
    ax2.bar([period1, period2], means, yerr=stds, capsize=10)
    ax2.set_title(f"Mean {value_column} by Time Period")
    ax2.set_ylabel(f"Mean {value_column}")
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Show summary statistics
    st.write("### Summary Statistics")
    stats = pd.DataFrame({
        'Statistic': ['Count', 'Mean', 'Std Dev', 'Min', '25%', 'Median', '75%', 'Max'],
        period1: [
            morning_data.count(),
            morning_data.mean(),
            morning_data.std(),
            morning_data.min(),
            morning_data.quantile(0.25),
            morning_data.median(),
            morning_data.quantile(0.75),
            morning_data.max()
        ],
        period2: [
            evening_data.count(),
            evening_data.mean(),
            evening_data.std(),
            evening_data.min(),
            evening_data.quantile(0.25),
            evening_data.median(),
            evening_data.quantile(0.75),
            evening_data.max()
        ]
    })
    
    st.write(stats)
    
    # Run t-test to check if difference is significant
    from scipy import stats as scipy_stats
    t_stat, p_value = scipy_stats.ttest_ind(morning_data, evening_data, equal_var=False)
    
    st.write("### Statistical Significance")
    st.write(f"t-statistic: {t_stat:.4f}")
    st.write(f"p-value: {p_value:.4f}")
    
    if p_value < 0.05:
        st.write(f"The difference between {period1} and {period2} is statistically significant (p < 0.05).")
    else:
        st.write(f"The difference between {period1} and {period2} is not statistically significant (p ≥ 0.05).")
    
    return
