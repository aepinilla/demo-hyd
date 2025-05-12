"""
Sensor.community API integration.

This module provides functions for fetching and processing data from sensor.community APIs.
"""

import requests
import pandas as pd
from typing import Dict, Any, List, Optional
import time
from datetime import datetime, timedelta

# API Base URLs
DATA_BASE_URL = "https://data.sensor.community"
API_BASE_URL = "https://api.sensor.community"

def fetch_latest_data(data_type: Optional[str] = None, country: Optional[str] = None, 
                     area: Optional[str] = None, box: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch the latest data from sensor.community API.
    
    Args:
        data_type: Optional comma-separated list of sensor types (e.g., 'SDS011,BME280')
        country: Optional comma-separated list of country codes (e.g., 'DE,BE')
        area: Optional area filter in format 'lat,lon,distance' (e.g., '52.5200,13.4050,10')
        box: Optional box filter in format 'lat1,lon1,lat2,lon2' (e.g., '52.1,13.0,53.5,13.5')
    
    Returns:
        DataFrame containing the sensor data
    """
    # Debug information
    print(f"Fetching sensor data with parameters: data_type={data_type}, country={country}, area={area}, box={box}")
    
    # Handle common user input issues
    if country and len(country) > 2 and ',' not in country:
        # User likely entered a full country name instead of a code
        country_map = {
            'germany': 'DE',
            'france': 'FR',
            'italy': 'IT',
            'spain': 'ES',
            'united kingdom': 'GB',
            'uk': 'GB',
            'netherlands': 'NL',
            'belgium': 'BE',
            'austria': 'AT',
            'switzerland': 'CH',
            'poland': 'PL'
        }
        country_lower = country.lower()
        if country_lower in country_map:
            print(f"Converting country name '{country}' to country code '{country_map[country_lower]}'")
            country = country_map[country_lower]
    # Construct the query parameters
    params = {}
    if data_type:
        params['type'] = data_type
    if country:
        params['country'] = country
    if area:
        params['area'] = area
    if box:
        params['box'] = box
    
    # Set User-Agent header as required by the API
    headers = {'User-Agent': 'HackYourDistrict2025/1.0 (data-visualization-assistant)'}
    
    # Determine which endpoint to use
    if any([data_type, country, area, box]):
        url = f"{DATA_BASE_URL}/airrohr/v1/filter/"
    else:
        url = f"{DATA_BASE_URL}/static/v1/data.json"
    
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        # Convert to DataFrame
        if isinstance(data, list):
            # Flatten the nested structure for easier analysis
            flat_data = []
            for item in data:
                base_info = {
                    'sensor_id': item.get('id'),
                    'sensor_type': item.get('sensor', {}).get('sensor_type', {}).get('name'),
                    'location_id': item.get('location', {}).get('id'),
                    'latitude': item.get('location', {}).get('latitude'),
                    'longitude': item.get('location', {}).get('longitude'),
                    'timestamp': item.get('timestamp')
                }
                
                # Extract sensordatavalues
                for value in item.get('sensordatavalues', []):
                    entry = base_info.copy()
                    entry['value_type'] = value.get('value_type')
                    entry['value'] = value.get('value')
                    flat_data.append(entry)
            
            df = pd.DataFrame(flat_data)
            
            # Convert numeric values
            numeric_columns = ['value', 'latitude', 'longitude']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert timestamp to datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            
            return df
        else:
            # Handle case when response is not a list
            return pd.DataFrame([data])
    
    except Exception as e:
        print(f"Error fetching sensor data: {str(e)}")
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=['sensor_id', 'sensor_type', 'location_id', 
                                    'latitude', 'longitude', 'timestamp', 
                                    'value_type', 'value'])

def fetch_sensor_data(sensor_id: str) -> pd.DataFrame:
    """
    Fetch data for a specific sensor by its API ID.
    
    Args:
        sensor_id: The API ID of the sensor
    
    Returns:
        DataFrame containing the sensor's data
    """
    url = f"{DATA_BASE_URL}/airrohr/v1/sensor/{sensor_id}/"
    headers = {'User-Agent': 'HackYourDistrict2025/1.0 (data-visualization-assistant)'}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        # Convert to DataFrame (similar structure to above)
        flat_data = []
        for item in data:
            base_info = {
                'sensor_id': item.get('id'),
                'sensor_type': item.get('sensor', {}).get('sensor_type', {}).get('name'),
                'location_id': item.get('location', {}).get('id'),
                'latitude': item.get('location', {}).get('latitude'),
                'longitude': item.get('location', {}).get('longitude'),
                'timestamp': item.get('timestamp')
            }
            
            # Extract sensordatavalues
            for value in item.get('sensordatavalues', []):
                entry = base_info.copy()
                entry['value_type'] = value.get('value_type')
                entry['value'] = value.get('value')
                flat_data.append(entry)
        
        df = pd.DataFrame(flat_data)
        
        # Convert numeric values
        numeric_columns = ['value', 'latitude', 'longitude']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        return df
    
    except Exception as e:
        print(f"Error fetching sensor data: {str(e)}")
        return pd.DataFrame(columns=['sensor_id', 'sensor_type', 'location_id', 
                                    'latitude', 'longitude', 'timestamp', 
                                    'value_type', 'value'])

def fetch_average_data(timespan: str = '5m', data_type: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch average data for all sensors within a specific timespan.
    
    Args:
        timespan: Time window for which to get data ('5m', '1h', or '24h')
        data_type: Optional filter for dust or temperature sensors ('dust' or 'temp')
    
    Returns:
        DataFrame containing the average sensor data
    """
    # Map timespan to endpoint
    timespan_map = {
        '5m': 'data.json',
        '1h': 'data.1h.json',
        '24h': 'data.24h.json'
    }
    
    # Map data_type to endpoint suffix
    data_type_map = {
        'dust': 'data.dust.min.json',
        'temp': 'data.temp.min.json'
    }
    
    # Determine the endpoint
    if data_type and data_type in data_type_map:
        endpoint = data_type_map[data_type]
    elif timespan in timespan_map:
        endpoint = timespan_map[timespan]
    else:
        endpoint = 'data.json'  # Default to 5 min data
    
    url = f"{DATA_BASE_URL}/static/v2/{endpoint}"
    headers = {'User-Agent': 'HackYourDistrict2025/1.0 (data-visualization-assistant)'}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        flat_data = []
        for item in data:
            record = {
                'sensor_id': item.get('id'),
                'latitude': item.get('latitude'),
                'longitude': item.get('longitude'),
                'timestamp': item.get('timestamp')
            }
            
            # Extract sensordatavalues
            for key, value in item.get('sensordatavalues', {}).items():
                record[key] = value
                
            flat_data.append(record)
        
        df = pd.DataFrame(flat_data)
        
        # Convert timestamp to datetime if it exists
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            
        return df
    
    except Exception as e:
        print(f"Error fetching average sensor data: {str(e)}")
        return pd.DataFrame()

def get_sensor_types() -> List[str]:
    """
    Get a list of available sensor types from the API.
    
    Returns:
        List of sensor type names
    """
    # This is based on common sensor types in the API documentation
    return [
        'SDS011', 'BME280', 'DHT22', 'BMP180', 
        'PMS1003', 'PMS3003', 'PMS5003', 'PMS7003'
    ]

def get_value_types() -> Dict[str, str]:
    """
    Get a mapping of value types to their descriptions.
    
    Returns:
        Dictionary mapping value types to descriptions
    """
    return {
        'P1': 'PM10 - particulate matter 10µm or less (µg/m³)',
        'P2': 'PM2.5 - particulate matter 2.5µm or less (µg/m³)',
        'temperature': 'Temperature (°C)',
        'humidity': 'Relative humidity (%)',
        'pressure': 'Atmospheric pressure (Pa)'
    }
