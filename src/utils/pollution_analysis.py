"""
Pollution data analysis utilities.

This module provides specialized functions for analyzing air quality data from sensor.community.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from typing import Dict, Any, Optional, Union, List, Tuple
from datetime import datetime, timedelta

# Air Quality Index (AQI) thresholds based on European standards
# Values in μg/m³
PM10_THRESHOLDS = [0, 20, 40, 50, 100, 150, 1200]  # PM10 thresholds
PM25_THRESHOLDS = [0, 10, 20, 25, 50, 75, 800]     # PM2.5 thresholds
AQI_LABELS = ["Very Good", "Good", "Moderate", "Poor", "Very Poor", "Extremely Poor"]
AQI_COLORS = ["#50F0E6", "#50CCAA", "#F0E641", "#FF5050", "#960032", "#7D2181"]

def analyze_pollution_data(
    df_or_data_type: Union[pd.DataFrame, str, Dict[str, Any]],
    pollutant_type: str = "P1",
    calculate_aqi: bool = True,
    show_statistics: bool = True,
    show_visualization: bool = True
) -> pd.DataFrame:
    """
    Analyze pollution data and provide insights on air quality.
    
    Args:
        df_or_data_type: Either a DataFrame, a string with a data type, or a dict with parameters
        pollutant_type: Type of pollutant to analyze ('P1' for PM10 or 'P2' for PM2.5)
        calculate_aqi: Whether to calculate Air Quality Index
        show_statistics: Whether to show descriptive statistics
        show_visualization: Whether to create visualizations
        
    Returns:
        DataFrame with analysis results
    """
    # Handle the case where df is a dictionary
    if isinstance(df_or_data_type, dict):
        # Extract parameters from the dictionary
        params = df_or_data_type
        
        # If 'df' is in the dictionary, use it as the DataFrame
        if 'df' in params:
            df_param = params.get('df')
            
            # Check if df_param is a string (e.g., "latest_sensor_data")
            if isinstance(df_param, str) and 'latest_sensor_data' in df_param:
                # Use the most recently fetched data
                if 'latest_data' in st.session_state:
                    df = st.session_state.latest_data
                else:
                    st.error("No sensor data has been fetched yet. Please fetch data first.")
                    return pd.DataFrame()
            else:
                # Assume df_param is the actual DataFrame
                df = df_param
        else:
            # No DataFrame provided, check if we have latest data in session state
            if 'latest_data' in st.session_state:
                df = st.session_state.latest_data
            else:
                st.error("No sensor data has been fetched yet. Please fetch data first.")
                return pd.DataFrame()
        
        # Extract other parameters if provided
        pollutant_type = params.get('pollutant_type', pollutant_type)
        calculate_aqi = params.get('calculate_aqi', calculate_aqi)
        show_statistics = params.get('show_statistics', show_statistics)
        show_visualization = params.get('show_visualization', show_visualization)
    
    # Handle the case where df is a string (e.g., a sensor type)
    elif isinstance(df_or_data_type, str):
        # Check if we have latest data in session state
        if 'latest_data' in st.session_state:
            df = st.session_state.latest_data
            
            # If the string is a sensor type, filter the data
            if df_or_data_type in ['SDS011', 'SPS30', 'PMS5003', 'PMS7003']:
                df = df[df['sensor_type'] == df_or_data_type]
        else:
            st.error("No sensor data has been fetched yet. Please fetch data first.")
            return pd.DataFrame()
    
    # Handle the case where df is already a DataFrame
    else:
        df = df_or_data_type
    
    # Filter for the specified pollutant type
    if pollutant_type not in ['P1', 'P2']:
        st.warning(f"Invalid pollutant type: {pollutant_type}. Using P1 (PM10) as default.")
        pollutant_type = 'P1'
    
    # Filter the data for the specified pollutant
    filtered_df = df[df['value_type'] == pollutant_type].copy()
    
    if filtered_df.empty:
        st.error(f"No data found for pollutant type {pollutant_type}.")
        return pd.DataFrame()
    
    # Ensure value column is numeric
    filtered_df['value'] = pd.to_numeric(filtered_df['value'], errors='coerce')
    
    # Convert object columns to strings to avoid PyArrow serialization issues
    for col in filtered_df.columns:
        if filtered_df[col].dtype == 'object':
            filtered_df[col] = filtered_df[col].astype(str)
    
    # Calculate Air Quality Index if requested
    if calculate_aqi:
        if pollutant_type == 'P1':  # PM10
            filtered_df['aqi_category'] = pd.cut(
                filtered_df['value'], 
                bins=PM10_THRESHOLDS, 
                labels=AQI_LABELS, 
                include_lowest=True
            )
        else:  # PM2.5 (P2)
            filtered_df['aqi_category'] = pd.cut(
                filtered_df['value'], 
                bins=PM25_THRESHOLDS, 
                labels=AQI_LABELS, 
                include_lowest=True
            )
        
        # Count occurrences of each AQI category
        aqi_counts = filtered_df['aqi_category'].value_counts().sort_index()
        
        # Display AQI distribution
        st.subheader(f"Air Quality Index Distribution for {get_pollutant_name(pollutant_type)}")
        
        # Create a DataFrame for the AQI distribution
        aqi_df = pd.DataFrame({
            'AQI Category': aqi_counts.index,
            'Count': aqi_counts.values,
            'Percentage': (aqi_counts.values / len(filtered_df) * 100).round(1)
        })
        
        # Display the AQI distribution table
        st.dataframe(aqi_df)
        
        # Create a pie chart of AQI distribution
        if show_visualization:
            fig, ax = plt.subplots(figsize=(10, 6))
            wedges, texts, autotexts = ax.pie(
                aqi_counts.values,
                labels=aqi_counts.index,
                autopct='%1.1f%%',
                colors=[AQI_COLORS[list(AQI_LABELS).index(cat)] for cat in aqi_counts.index],
                startangle=90
            )
            ax.set_title(f'Air Quality Distribution - {get_pollutant_name(pollutant_type)}')
            plt.setp(autotexts, size=10, weight='bold')
            st.pyplot(fig)
    
    # Show descriptive statistics if requested
    if show_statistics:
        st.subheader(f"Statistical Summary for {get_pollutant_name(pollutant_type)}")
        
        # Calculate basic statistics
        stats = {
            'Mean': filtered_df['value'].mean(),
            'Median': filtered_df['value'].median(),
            'Min': filtered_df['value'].min(),
            'Max': filtered_df['value'].max(),
            'Standard Deviation': filtered_df['value'].std(),
            'Number of Sensors': filtered_df['sensor_id'].nunique(),
            'Number of Readings': len(filtered_df)
        }
        
        # Create a DataFrame for the statistics
        stats_df = pd.DataFrame({
            'Statistic': list(stats.keys()),
            'Value': list(stats.values())
        })
        
        # Display the statistics table
        st.dataframe(stats_df)
    
    # Create additional visualizations if requested
    if show_visualization:
        st.subheader(f"Visualizations for {get_pollutant_name(pollutant_type)}")
        
        # Create a histogram of pollution values
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(filtered_df['value'], bins=30, kde=True, ax=ax)
        ax.set_title(f'Distribution of {get_pollutant_name(pollutant_type)} Values')
        ax.set_xlabel(f'{get_pollutant_name(pollutant_type)} (μg/m³)')
        ax.set_ylabel('Frequency')
        
        # Add vertical lines for AQI thresholds
        if pollutant_type == 'P1':
            thresholds = PM10_THRESHOLDS[1:-1]  # Skip the first and last threshold
        else:
            thresholds = PM25_THRESHOLDS[1:-1]  # Skip the first and last threshold
            
        for i, threshold in enumerate(thresholds):
            ax.axvline(x=threshold, color=AQI_COLORS[i], linestyle='--', alpha=0.7)
            ax.text(threshold + 1, ax.get_ylim()[1] * 0.9, AQI_LABELS[i], 
                   rotation=90, verticalalignment='top', color=AQI_COLORS[i])
        
        st.pyplot(fig)
        
        # If we have location data, create a map
        if 'latitude' in filtered_df.columns and 'longitude' in filtered_df.columns:
            try:
                # First, ensure we have numeric latitude and longitude
                filtered_df['latitude'] = pd.to_numeric(filtered_df['latitude'], errors='coerce')
                filtered_df['longitude'] = pd.to_numeric(filtered_df['longitude'], errors='coerce')
                
                # Filter out rows with invalid coordinates
                valid_coords_df = filtered_df.dropna(subset=['latitude', 'longitude'])
                
                # Group by sensor location and calculate average - avoid MultiIndex issues
                map_data = valid_coords_df.groupby('sensor_id').agg({
                    'latitude': 'first',  # Take the first latitude for each sensor
                    'longitude': 'first',  # Take the first longitude for each sensor
                    'value': 'mean'       # Calculate the mean value for each sensor
                }).reset_index()
                
                # Remove rows with missing coordinates
                map_data = map_data.dropna(subset=['latitude', 'longitude'])
            except Exception as e:
                st.warning(f"Could not create geographical distribution map: {str(e)}")
                map_data = pd.DataFrame()  # Empty DataFrame if there's an error
            
            if not map_data.empty:
                st.subheader(f"Geographical Distribution of {get_pollutant_name(pollutant_type)}")
                
                # Determine color based on AQI category
                if pollutant_type == 'P1':  # PM10
                    map_data['aqi_category'] = pd.cut(
                        map_data['value'], 
                        bins=PM10_THRESHOLDS, 
                        labels=AQI_LABELS, 
                        include_lowest=True
                    )
                else:  # PM2.5 (P2)
                    map_data['aqi_category'] = pd.cut(
                        map_data['value'], 
                        bins=PM25_THRESHOLDS, 
                        labels=AQI_LABELS, 
                        include_lowest=True
                    )
                
                # Create a color mapping
                def get_color(aqi_category):
                    if pd.isna(aqi_category):
                        return "#CCCCCC"  # Default gray color for NaN values
                    try:
                        return AQI_COLORS[list(AQI_LABELS).index(aqi_category)]
                    except (ValueError, IndexError):
                        return "#CCCCCC"  # Default color if category not found
                
                map_data['color'] = map_data['aqi_category'].apply(get_color)
                
                # Ensure all values in the color column are valid strings
                map_data = map_data.dropna(subset=['color'])
                
                # Display the map
                st.map(map_data, latitude='latitude', longitude='longitude', color='color')
    
    return filtered_df

def get_pollutant_name(pollutant_type: str) -> str:
    """
    Get the full name of a pollutant based on its code.
    
    Args:
        pollutant_type: Pollutant code (e.g., 'P1', 'P2')
        
    Returns:
        Full name of the pollutant
    """
    if pollutant_type == 'P1':
        return 'PM10 (Particulate Matter ≤ 10µm)'
    elif pollutant_type == 'P2':
        return 'PM2.5 (Fine Particulate Matter ≤ 2.5µm)'
    else:
        return pollutant_type


def compare_pm10_pm25(df: pd.DataFrame, time_period: str = '24h', resample_freq: str = '1H') -> None:
    """
    Create a specialized visualization comparing PM10 and PM2.5 levels over time.
    
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
    
    # Ensure value columns are numeric
    pm10_data['value'] = pd.to_numeric(pm10_data['value'], errors='coerce')
    pm25_data['value'] = pd.to_numeric(pm25_data['value'], errors='coerce')
    
    # Convert object columns to strings to avoid PyArrow serialization issues
    for col in pm10_data.columns:
        if pm10_data[col].dtype == 'object':
            pm10_data[col] = pm10_data[col].astype(str)
            
    for col in pm25_data.columns:
        if pm25_data[col].dtype == 'object':
            pm25_data[col] = pm25_data[col].astype(str)
    
    # Extra check for PyArrow serialization compatibility
    try:
        # Try converting to pyarrow table to check for errors
        import pyarrow as pa
        try:
            pa.Table.from_pandas(pm10_data)
            pa.Table.from_pandas(pm25_data)
        except Exception as e:
            st.warning(f"Fixing PyArrow serialization issue: {str(e)}")
            # Additional conversion for problematic columns
            for col in pm10_data.columns:
                try:
                    if pm10_data[col].dtype.name == 'object':
                        pm10_data[col] = pm10_data[col].astype(str)
                except:
                    # Last resort - convert the column to string
                    pm10_data[col] = pm10_data[col].astype(str)
                    
            for col in pm25_data.columns:
                try:
                    if pm25_data[col].dtype.name == 'object':
                        pm25_data[col] = pm25_data[col].astype(str)
                except:
                    # Last resort - convert the column to string
                    pm25_data[col] = pm25_data[col].astype(str)
    except ImportError:
        # PyArrow not available, no need to check
        pass
    
    # Resample data to get regular time intervals
    pm10_resampled = pm10_data.set_index('timestamp')['value'].resample(resample_freq).mean().reset_index()
    pm25_resampled = pm25_data.set_index('timestamp')['value'].resample(resample_freq).mean().reset_index()
    
    # Create figure with two subplots
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot PM10 data
        ax1.plot(pm10_resampled['timestamp'], pm10_resampled['value'], 'o-', color='#FF5050', label='PM10')
        ax1.set_ylabel('PM10 (µg/m³)')
        ax1.set_title(f'PM10 Levels - {period_name}')
        ax1.grid(True, alpha=0.3)
        
        # Add PM10 threshold lines
        ax1.axhline(y=50, color='#FFA500', linestyle='--', alpha=0.7, label='EU Daily Limit (50 µg/m³)')
        ax1.axhline(y=20, color='#50CCAA', linestyle='--', alpha=0.7, label='WHO Guideline (20 µg/m³)')
        ax1.legend()
        
        # Plot PM2.5 data
        ax2.plot(pm25_resampled['timestamp'], pm25_resampled['value'], 'o-', color='#960032', label='PM2.5')
        ax2.set_ylabel('PM2.5 (µg/m³)')
        ax2.set_xlabel('Time')
        ax2.set_title(f'PM2.5 Levels - {period_name}')
        ax2.grid(True, alpha=0.3)
        
        # Add PM2.5 threshold lines
        ax2.axhline(y=25, color='#FFA500', linestyle='--', alpha=0.7, label='EU Daily Limit (25 µg/m³)')
        ax2.axhline(y=10, color='#50CCAA', linestyle='--', alpha=0.7, label='WHO Guideline (10 µg/m³)')
        ax2.legend()
        
        # Format x-axis
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Display the plot
        st.pyplot(fig)
        plt.close(fig)
        
        # Calculate correlation between PM10 and PM2.5
        # Merge the resampled data on timestamp
        merged_df = pd.merge(pm10_resampled, pm25_resampled, on='timestamp', suffixes=('_pm10', '_pm25'))
        correlation = merged_df['value_pm10'].corr(merged_df['value_pm25'])
        
        # Create a scatter plot to show the relationship
        fig2, ax = plt.subplots(figsize=(8, 6))
        sns.regplot(x='value_pm10', y='value_pm25', data=merged_df, ax=ax, scatter_kws={'alpha':0.5})
        ax.set_xlabel('PM10 (µg/m³)')
        ax.set_ylabel('PM2.5 (µg/m³)')
        ax.set_title(f'PM2.5 vs PM10 Correlation (r = {correlation:.2f})')
        ax.grid(True, alpha=0.3)
        
        # Display the correlation plot
        st.pyplot(fig2)
        plt.close(fig2)
        
        # Display statistics
        st.subheader("Comparison Statistics")
        stats_df = pd.DataFrame({
            'Statistic': ['Mean', 'Median', 'Max', 'Min', 'Std Dev', 'Correlation'],
            'PM10 (µg/m³)': [
                f"{pm10_resampled['value'].mean():.2f}",
                f"{pm10_resampled['value'].median():.2f}",
                f"{pm10_resampled['value'].max():.2f}",
                f"{pm10_resampled['value'].min():.2f}",
                f"{pm10_resampled['value'].std():.2f}",
                f"{correlation:.2f}"
            ],
            'PM2.5 (µg/m³)': [
                f"{pm25_resampled['value'].mean():.2f}",
                f"{pm25_resampled['value'].median():.2f}",
                f"{pm25_resampled['value'].max():.2f}",
                f"{pm25_resampled['value'].min():.2f}",
                f"{pm25_resampled['value'].std():.2f}",
                ""
            ]
        })
        st.table(stats_df)
        
        # Calculate PM2.5/PM10 ratio
        ratio = pm25_resampled['value'].mean() / pm10_resampled['value'].mean()
        st.info(f"Average PM2.5/PM10 Ratio: {ratio:.2f} (typical urban ratios range from 0.5 to 0.8)")
        
    except Exception as e:
        st.error(f"Error creating comparison visualization: {str(e)}")
        print(f"Error in compare_pm10_pm25: {str(e)}")
        import traceback
        print(traceback.format_exc())
    return pollutant_type
