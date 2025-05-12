"""
Sensor.community API integration.

This module provides functions for fetching and processing data from sensor.community APIs.
"""

import requests
import pandas as pd
from typing import Dict, Any, List, Optional
import time
from datetime import datetime, timedelta

# Import the process_sensor_data function
from src.utils.process_sensor_data import process_sensor_data

# API Base URLs
DATA_BASE_URL = "https://data.sensor.community"
API_BASE_URL = "https://api.sensor.community"

def fetch_latest_data(data_type: Optional[str] = None, country: Optional[str] = None, 
                     area: Optional[str] = None, box: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch the latest data from sensor.community API.
    
    Args:
        data_type: Optional sensor type to filter by (e.g., 'SDS011' or 'DHT22')
        country: Optional comma-separated list of country codes (e.g., 'DE,BE')
        area: Optional area filter in format 'lat,lon,distance' (e.g., '52.5200,13.4050,10')
        box: Optional box filter in format 'lat1,lon1,lat2,lon2' (e.g., '52.1,13.0,53.5,13.5')
    
    Returns:
        DataFrame containing the sensor data
    """
    # Note: Based on API testing, the available sensor types are 'DHT22' and 'SDS011'
    # Also, the country filter may not work as expected (returns empty results)
    
    # Import the sample data module
    from src.utils.sample_data import get_sample_data
    
    # Flag to determine if we should use sample data
    # Set to False to try the real API first, falling back to sample data if needed
    use_sample_data = False
    # Print raw inputs for debugging
    print(f"Raw inputs - data_type: {type(data_type)}, {data_type}")
    print(f"Raw inputs - country: {type(country)}, {country}")
    
    # Handle JSON input if it was passed as a string or dict
    if isinstance(data_type, dict):
        # Extract parameters from the dict
        params_dict = data_type
        print(f"Found dict input: {params_dict}")
        
        # Extract parameters
        if 'data_type' in params_dict:
            data_type = params_dict.get('data_type')
        if 'country' in params_dict:
            country = params_dict.get('country')
        if 'area' in params_dict:
            area = params_dict.get('area')
        if 'box' in params_dict:
            box = params_dict.get('box')
    elif isinstance(data_type, str) and '{' in data_type:
        # It might be a JSON string, try to parse it
        try:
            import json
            # Clean up the string by removing newlines and extra quotes
            clean_json = data_type.replace('\n', '').replace('```', '').strip()
            print(f"Attempting to parse JSON string: {clean_json}")
            
            # Try to extract just the JSON part if there's other text
            if '{' in clean_json and '}' in clean_json:
                start = clean_json.find('{')
                end = clean_json.rfind('}') + 1
                json_part = clean_json[start:end]
                print(f"Extracted JSON part: {json_part}")
                params_dict = json.loads(json_part)
            else:
                params_dict = json.loads(clean_json)
            
            # Extract parameters
            if 'data_type' in params_dict:
                data_type = params_dict.get('data_type')
                print(f"Extracted data_type: {data_type}")
            if 'country' in params_dict:
                country = params_dict.get('country')
                print(f"Extracted country: {country}")
            if 'area' in params_dict:
                area = params_dict.get('area')
            if 'box' in params_dict:
                box = params_dict.get('box')
        except Exception as e:
            # If parsing fails, keep the original value
            print(f"JSON parsing failed: {e}")
            
            # Try a more aggressive approach to extract parameters
            try:
                # Try to extract data_type and country using regex
                import re
                data_type_match = re.search(r'"data_type"\s*:\s*"([^"]+)"', data_type)
                country_match = re.search(r'"country"\s*:\s*"([^"]+)"', data_type)
                
                if data_type_match:
                    data_type = data_type_match.group(1)
                    print(f"Regex extracted data_type: {data_type}")
                
                if country_match:
                    country = country_match.group(1)
                    print(f"Regex extracted country: {country}")
            except Exception as regex_error:
                print(f"Regex extraction failed: {regex_error}")
                pass
    
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
    
    # Based on API testing, we should prioritize the static endpoints since they have more data
    # The filter endpoint with country=DE returns empty results
    
    # Choose the appropriate endpoint based on the parameters
    if data_type and data_type.lower() in ['dht22', 'temperature', 'humidity']:
        # For temperature/humidity sensors
        url = f"{DATA_BASE_URL}/static/v2/data.temp.min.json"
        print(f"Using temperature/humidity endpoint: {url}")
    elif data_type and data_type.lower() in ['sds011', 'pm2.5', 'pm10', 'p1', 'p2', 'dust']:
        # For dust/particulate matter sensors
        url = f"{DATA_BASE_URL}/static/v2/data.dust.min.json"
        print(f"Using dust/particulate matter endpoint: {url}")
    elif any([country, area, box]):
        # Try the filter endpoint, but it may return empty results
        url = f"{DATA_BASE_URL}/airrohr/v1/filter/"
        print(f"Using filter endpoint: {url} (note: may return empty results)")
    else:
        # Default to all data
        url = f"{DATA_BASE_URL}/static/v1/data.json"
        print(f"Using default endpoint: {url}")
    
    # Only use sample data if explicitly requested
    if use_sample_data:
        print("Using sample data instead of live API")
        sample_df = get_sample_data(data_type)
        
        # Store the data in session state for visualization tools to use
        import streamlit as st
        st.session_state.latest_data = sample_df
        
        return sample_df
    
    # Make the API request if not using sample data
    try:
        # Use a more informative User-Agent as required by the API
        headers = {
            'User-Agent': 'HackYourDistrict2025/1.0 (data-visualization-assistant; contact@example.com)'
        }
        
        print(f"Making API request to: {url}")
        print(f"With parameters: {params}")
        print(f"Using headers: {headers}")
        
        # Increase timeout to handle potential slow responses
        response = requests.get(url, params=params, headers=headers, timeout=10)
        
        # Check if the request was successful
        response.raise_for_status()
        
        # Parse the JSON response
        data = response.json()
        
        # Check if we got data
        if not data:
            print("No data returned from API, falling back to sample data")
            sample_df = get_sample_data(data_type)
            
            # Store the data in session state for visualization tools to use
            import streamlit as st
            st.session_state.latest_data = sample_df
            
            return sample_df
            
        print(f"Successfully retrieved {len(data)} records from the API")
        
        # Process the sensor data
        df = process_sensor_data(data)
        
        # Convert object columns to strings to avoid PyArrow serialization issues
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str)
        
        return df
    
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
    headers = {'User-Agent': 'HackYourDistrict2025/1.0 (data-visualization-assistant; contact@example.com)'}
    
    try:
        print(f"Making API request to: {url}")
        print(f"Using headers: {headers}")
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        print(f"Successfully retrieved data from {url}")
        
        # Check if data is a list (which is the expected format)
        if not isinstance(data, list):
            print(f"Unexpected data format: {type(data)}")
            return pd.DataFrame()
            
        print(f"Processing {len(data)} records")
        
        flat_data = []
        for item in data:
            # Basic sensor info
            record = {
                'sensor_id': item.get('id'),
                'sensor_type': None,  # Will be populated from sensordatavalues if available
                'latitude': item.get('latitude'),
                'longitude': item.get('longitude'),
                'timestamp': item.get('timestamp')
            }
            
            # Extract sensordatavalues - this is a list of dicts in v2 API
            if 'sensordatavalues' in item:
                sensor_data = item.get('sensordatavalues')
                
                # Handle different formats of sensordatavalues
                if isinstance(sensor_data, list):
                    # Format: list of dicts with value_type and value
                    for value_item in sensor_data:
                        value_type = value_item.get('value_type')
                        value = value_item.get('value')
                        
                        if value_type and value:
                            new_record = record.copy()
                            new_record['value_type'] = value_type
                            new_record['value'] = value
                            flat_data.append(new_record)
                            
                            # Try to determine sensor type from value_type
                            if value_type in ['P1', 'P2'] and not new_record['sensor_type']:
                                new_record['sensor_type'] = 'SDS011'
                            elif value_type in ['temperature', 'humidity'] and not new_record['sensor_type']:
                                new_record['sensor_type'] = 'DHT22'
                elif isinstance(sensor_data, dict):
                    # Format: dict with value_type as keys
                    base_record = record.copy()
                    for value_type, value in sensor_data.items():
                        new_record = base_record.copy()
                        new_record['value_type'] = value_type
                        new_record['value'] = value
                        flat_data.append(new_record)
                        
                        # Try to determine sensor type from value_type
                        if value_type in ['P1', 'P2'] and not new_record['sensor_type']:
                            new_record['sensor_type'] = 'SDS011'
                        elif value_type in ['temperature', 'humidity'] and not new_record['sensor_type']:
                            new_record['sensor_type'] = 'DHT22'
        
        # If no data was processed, return empty DataFrame
        if not flat_data:
            print("No data could be extracted from the API response")
            return pd.DataFrame()
            
        df = pd.DataFrame(flat_data)
        
        # Convert numeric values
        numeric_columns = ['value', 'latitude', 'longitude']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert timestamp to datetime if it exists
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        print(f"Processed {len(df)} data points")
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
    # Based on API testing, these are the sensor types actually available in the data
    return [
        'SDS011', 'DHT22'
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
