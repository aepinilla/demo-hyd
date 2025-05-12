"""
Sample sensor data for demonstration purposes.

This module provides sample sensor data that can be used when the live API
is not returning results.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_sample_sds011_data(num_sensors=5, num_readings=100):
    """
    Generate sample SDS011 dust sensor data.
    
    Args:
        num_sensors: Number of unique sensors to simulate
        num_readings: Number of readings per sensor
        
    Returns:
        DataFrame with sample sensor data
    """
    data = []
    
    # Create sensor IDs
    sensor_ids = [f"sample_sensor_{i}" for i in range(1, num_sensors + 1)]
    
    # Create timestamps (last 24 hours, every 15 minutes)
    end_time = datetime.now()
    timestamps = [end_time - timedelta(minutes=15 * i) for i in range(num_readings)]
    
    # Generate random locations in Europe
    locations = [
        {"id": 1, "lat": 52.52, "lon": 13.41, "name": "Berlin"},  # Berlin
        {"id": 2, "lat": 48.85, "lon": 2.35, "name": "Paris"},    # Paris
        {"id": 3, "lat": 51.51, "lon": -0.13, "name": "London"},  # London
        {"id": 4, "lat": 41.90, "lon": 12.50, "name": "Rome"},    # Rome
        {"id": 5, "lat": 40.42, "lon": -3.70, "name": "Madrid"}   # Madrid
    ]
    
    # For each sensor
    for i, sensor_id in enumerate(sensor_ids):
        location = locations[i]
        
        # For each timestamp
        for ts in timestamps:
            # Generate realistic P1 (PM10) values (0-100 μg/m³)
            # Higher during morning and evening rush hours
            hour = ts.hour
            if 7 <= hour <= 9 or 16 <= hour <= 19:
                p1_base = random.uniform(15, 45)  # Rush hour - higher values
            else:
                p1_base = random.uniform(5, 25)   # Normal hours
                
            # Add some random variation
            p1_value = p1_base + random.uniform(-5, 5)
            p1_value = max(0, p1_value)  # Ensure non-negative
            
            # P2 (PM2.5) is typically lower than P1
            p2_value = p1_value * random.uniform(0.3, 0.7)
            
            # Add P1 reading
            data.append({
                'sensor_id': sensor_id,
                'sensor_type': 'SDS011',
                'location_id': location["id"],
                'latitude': location["lat"],
                'longitude': location["lon"],
                'timestamp': ts,
                'value_type': 'P1',
                'value': f"{p1_value:.2f}"
            })
            
            # Add P2 reading
            data.append({
                'sensor_id': sensor_id,
                'sensor_type': 'SDS011',
                'location_id': location["id"],
                'latitude': location["lat"],
                'longitude': location["lon"],
                'timestamp': ts,
                'value_type': 'P2',
                'value': f"{p2_value:.2f}"
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Convert value to numeric
    df['value'] = pd.to_numeric(df['value'])
    
    return df

def generate_sample_dht22_data(num_sensors=5, num_readings=100):
    """
    Generate sample DHT22 temperature and humidity sensor data.
    
    Args:
        num_sensors: Number of unique sensors to simulate
        num_readings: Number of readings per sensor
        
    Returns:
        DataFrame with sample sensor data
    """
    data = []
    
    # Create sensor IDs
    sensor_ids = [f"sample_temp_sensor_{i}" for i in range(1, num_sensors + 1)]
    
    # Create timestamps (last 24 hours, every 15 minutes)
    end_time = datetime.now()
    timestamps = [end_time - timedelta(minutes=15 * i) for i in range(num_readings)]
    
    # Generate random locations in Europe
    locations = [
        {"id": 6, "lat": 52.52, "lon": 13.41, "name": "Berlin"},  # Berlin
        {"id": 7, "lat": 48.85, "lon": 2.35, "name": "Paris"},    # Paris
        {"id": 8, "lat": 51.51, "lon": -0.13, "name": "London"},  # London
        {"id": 9, "lat": 41.90, "lon": 12.50, "name": "Rome"},    # Rome
        {"id": 10, "lat": 40.42, "lon": -3.70, "name": "Madrid"}  # Madrid
    ]
    
    # For each sensor
    for i, sensor_id in enumerate(sensor_ids):
        location = locations[i]
        
        # Base temperature for the location (different for each city)
        base_temp = 20 + random.uniform(-5, 5)
        
        # For each timestamp
        for ts in timestamps:
            # Temperature varies by time of day
            hour = ts.hour
            if 0 <= hour < 6:  # Night
                temp_offset = -3
            elif 6 <= hour < 12:  # Morning
                temp_offset = hour - 6  # Gradually warming up
            elif 12 <= hour < 18:  # Afternoon
                temp_offset = 6 - (hour - 12) * 0.5  # Gradually cooling down
            else:  # Evening
                temp_offset = 3 - (hour - 18) * 0.5  # Cooling down more
            
            # Add random variation
            temp_value = base_temp + temp_offset + random.uniform(-1, 1)
            
            # Humidity is inversely related to temperature
            humidity_base = 70 - (temp_value - 15)
            humidity_value = humidity_base + random.uniform(-10, 10)
            humidity_value = max(10, min(humidity_value, 95))  # Keep between 10-95%
            
            # Add temperature reading
            data.append({
                'sensor_id': sensor_id,
                'sensor_type': 'DHT22',
                'location_id': location["id"],
                'latitude': location["lat"],
                'longitude': location["lon"],
                'timestamp': ts,
                'value_type': 'temperature',
                'value': f"{temp_value:.2f}"
            })
            
            # Add humidity reading
            data.append({
                'sensor_id': sensor_id,
                'sensor_type': 'DHT22',
                'location_id': location["id"],
                'latitude': location["lat"],
                'longitude': location["lon"],
                'timestamp': ts,
                'value_type': 'humidity',
                'value': f"{humidity_value:.2f}"
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Convert value to numeric
    df['value'] = pd.to_numeric(df['value'])
    
    return df

def get_sample_data(data_type=None):
    """
    Get sample sensor data based on the requested data type.
    
    Args:
        data_type: Type of sensor data to get (e.g., 'SDS011', 'DHT22', 'P1', 'P2')
        
    Returns:
        DataFrame with sample sensor data
    """
    # Generate both types of data
    dust_data = generate_sample_sds011_data()
    temp_data = generate_sample_dht22_data()
    
    # Combine the data
    all_data = pd.concat([dust_data, temp_data], ignore_index=True)
    
    # Filter based on data_type if specified
    if data_type:
        data_type = data_type.upper() if data_type.upper() in ['P1', 'P2'] else data_type
        
        if data_type == 'SDS011':
            return dust_data
        elif data_type == 'DHT22':
            return temp_data
        elif data_type in ['P1', 'P2']:
            return dust_data[dust_data['value_type'] == data_type]
        elif data_type == 'TEMPERATURE':
            return temp_data[temp_data['value_type'] == 'temperature']
        elif data_type == 'HUMIDITY':
            return temp_data[temp_data['value_type'] == 'humidity']
    
    # Return all data if no filter or filter doesn't match
    return all_data
