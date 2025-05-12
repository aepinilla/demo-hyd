"""
Process sensor data from the sensor.community API.

This module provides functions for processing the raw data from the sensor.community API.
"""

import pandas as pd
import streamlit as st
from typing import Dict, Any, List, Optional

def process_sensor_data(response_data):
    """
    Process the sensor data from the API response.
    
    Args:
        response_data: JSON response from the API
        
    Returns:
        DataFrame containing the processed sensor data
    """
    # Process the response
    try:
        # Convert to DataFrame
        if isinstance(response_data, list):
            # Process the response
            processed_data = []
            
            for item in response_data:
                # Extract sensor information
                sensor_info = item.get('sensor', {})
                sensor_id = item.get('id')
                timestamp = item.get('timestamp')
                
                # Extract location information
                location = item.get('location', {})
                location_id = location.get('id')
                latitude = location.get('latitude')
                longitude = location.get('longitude')
                
                # Process sensor data values
                for value_item in item.get('sensordatavalues', []):
                    value_type = value_item.get('value_type')
                    value = value_item.get('value')
                    
                    # Skip entries with missing essential data
                    if not value or not value_type:
                        continue
                        
                    processed_data.append({
                        'sensor_id': sensor_id,
                        'sensor_type': sensor_info.get('sensor_type', {}).get('name'),
                        'location_id': location_id,
                        'latitude': latitude,
                        'longitude': longitude,
                        'timestamp': timestamp,
                        'value_type': value_type,
                        'value': value
                    })
            
            # If no data was processed, return an empty DataFrame
            if not processed_data:
                print("No valid data entries found in API response")
                return pd.DataFrame()
                
            df = pd.DataFrame(processed_data)
            
            # Store the data in session state for visualization tools to use
            import streamlit as st
            st.session_state.latest_data = df
            
            # Convert numeric values
            numeric_columns = ['value', 'latitude', 'longitude']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert timestamp to datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                
            # Log some information about the processed data
            print(f"Processed {len(df)} data points")
            if 'sensor_type' in df.columns:
                sensor_types = df['sensor_type'].unique()
                print(f"Found sensor types: {sensor_types}")
            if 'value_type' in df.columns:
                value_types = df['value_type'].unique()
                print(f"Found value types: {value_types}")
            
            return df
        else:
            # Handle case when response is not a list
            print("API response is not a list, attempting to process as a single record")
            return pd.DataFrame([response_data])
            
    except Exception as e:
        print(f"Error processing sensor data: {str(e)}")
        return pd.DataFrame()
