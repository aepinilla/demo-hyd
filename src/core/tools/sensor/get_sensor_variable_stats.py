import streamlit as st
import pandas as pd
import numpy as np
import datetime
import requests


def get_sensor_variable_stats() -> str:
    """
    Get descriptive statistics about the available variables in the sensor API.
    This tool provides information about what variables are available and their basic statistics.
    
    Returns:
        str: Information about available sensor variables and their statistics
    """
    
    # Check if we already have data in the session state
    if 'latest_data' not in st.session_state or st.session_state.latest_data.empty:
        # No data available, so fetch it automatically
        st.info("No sensor data found. Automatically fetching the latest data from sensor.community API...")
        
        try:
            # Use the default endpoint that has the most data
            endpoint = "https://data.sensor.community/static/v1/data.json"
            
            # Add proper headers to avoid API rejection
            headers = {
                'User-Agent': 'HackYourDistrict2025/1.0 (data-visualization-assistant; contact@example.com)'
            }
            
            # Make the API request
            st.info(f"Making API request to: {endpoint}")
            response = requests.get(endpoint, headers=headers)
            response.raise_for_status()  # Raise an exception for HTTP errors
            
            # Parse the JSON response
            data = response.json()
            st.info(f"Successfully retrieved {len(data)} records from the API")
            
            # Process the data into a DataFrame
            rows = []
            for item in data:
                sensor_id = item.get('sensor', {}).get('id')
                if not sensor_id:
                    continue
                    
                location_id = item.get('sensor', {}).get('sensor_type', {}).get('id')
                sensor_type = item.get('sensor', {}).get('sensor_type', {}).get('name')
                lat = item.get('location', {}).get('latitude')
                lon = item.get('location', {}).get('longitude')
                timestamp = item.get('timestamp')
                
                # Process all sensordatavalues
                for value_item in item.get('sensordatavalues', []):
                    value_type = value_item.get('value_type')
                    value = value_item.get('value')
                    
                    try:
                        value = float(value)
                    except (ValueError, TypeError):
                        continue
                        
                    rows.append({
                        'sensor_id': sensor_id,
                        'location_id': location_id,
                        'sensor_type': sensor_type,
                        'latitude': lat,
                        'longitude': lon,
                        'timestamp': timestamp,
                        'value_type': value_type,
                        'value': value
                    })
            
            # Create DataFrame
            df = pd.DataFrame(rows)
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Store in session state
            st.session_state.latest_data = df
            st.session_state.last_sensor_data_fetch = datetime.datetime.now()
            
            st.info(f"Processed {len(df)} data points")
            
        except Exception as e:
            return f"Error fetching sensor data: {str(e)}"
    
    # Get the data from session state
    df = st.session_state.latest_data
    
    if df.empty:
        return "Error: No sensor data available. The API returned empty results."
    
    # Filter to use only the latest measurement of each unique sensor
    # First, get the latest timestamp for each sensor
    latest_timestamps = df.groupby('sensor_id')['timestamp'].max().reset_index()
    latest_timestamps = latest_timestamps.rename(columns={'timestamp': 'latest_timestamp'})
    
    # Merge to get only the latest measurements for each sensor
    merged_df = pd.merge(df, latest_timestamps, on='sensor_id')
    df = merged_df[merged_df['timestamp'] == merged_df['latest_timestamp']]
    
    # Get unique sensor types
    sensor_types = df['sensor_type'].unique().tolist()
    
    # Get unique value types
    value_types = df['value_type'].unique().tolist()
    
    # Create a summary of available data
    summary = []
    summary.append(f"## Sensor Data Variables Summary\n")
    summary.append(f"**Total Records:** {len(df)}\n")
    summary.append(f"**Available Sensor Types:** {', '.join(sensor_types)}\n")
    summary.append(f"**Available Value Types:** {', '.join(value_types)}\n")
    
    # Create a pivot table with statistics for each value type
    stats_df = pd.DataFrame()
    for val_type in value_types:
        val_data = df[df['value_type'] == val_type]['value']
        if not val_data.empty:
            stats = val_data.describe()
            stats_df[val_type] = stats
    
    # Format the statistics table
    if not stats_df.empty:
        summary.append("## Descriptive Statistics by Variable\n")
        summary.append("```")
        summary.append(stats_df.to_string())
        summary.append("```\n")
    
    # Add information about sensor distribution
    sensor_counts = df['sensor_type'].value_counts()
    summary.append("## Sensor Type Distribution\n")
    for sensor_type, count in sensor_counts.items():
        summary.append(f"- **{sensor_type}:** {count} measurements\n")
    
    # Add information about geographical distribution if available
    if 'latitude' in df.columns and 'longitude' in df.columns:
        unique_locations = df.groupby(['latitude', 'longitude']).size().reset_index().rename(columns={0: 'count'})
        summary.append(f"\n## Geographical Distribution\n")
        summary.append(f"**Unique Locations:** {len(unique_locations)}\n")
        
        # Show a few sample locations
        if len(unique_locations) > 0:
            sample_size = min(5, len(unique_locations))
            summary.append(f"**Sample Locations (showing {sample_size} of {len(unique_locations)}):**\n")
            for _, row in unique_locations.head(sample_size).iterrows():
                # Convert latitude and longitude to float before formatting
                try:
                    lat = float(row['latitude'])
                    lon = float(row['longitude'])
                    summary.append(f"- Lat: {lat:.4f}, Lon: {lon:.4f} ({row['count']} measurements)\n")
                except (ValueError, TypeError):
                    # Fallback if conversion fails
                    summary.append(f"- Lat: {row['latitude']}, Lon: {row['longitude']} ({row['count']} measurements)\n")
    
    return "\n".join(summary)