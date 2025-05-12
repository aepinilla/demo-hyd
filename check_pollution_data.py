"""
Script to check for pollution data in the sensor.community API.
"""

from src.utils.api_tester import test_api_endpoint
import json

def check_pollution_data():
    print("Checking for pollution data in dust sensors endpoint...")
    result = test_api_endpoint('static/v2/data.dust.min.json')
    
    if not result.get('success'):
        print("Failed to fetch dust sensor data")
        return
    
    sample = result.get('sample', [])
    if not sample:
        print("No sample data available")
        return
    
    # Check the first sample for sensordatavalues
    print("\nSample sensor data values:")
    values = sample[0].get('sensordatavalues', [])
    print(json.dumps(values, indent=2))
    
    # Check for pollution-related value types
    pollution_types = set()
    for item in sample:
        for value in item.get('sensordatavalues', []):
            value_type = value.get('value_type')
            if value_type in ['P1', 'P2', 'P0', 'PM10', 'PM2.5']:
                pollution_types.add(value_type)
    
    print(f"\nFound pollution-related value types: {pollution_types}")
    
    # Check sensor types
    sensor_types = set()
    for item in sample:
        sensor_info = item.get('sensor', {}).get('sensor_type', {})
        sensor_type = sensor_info.get('name')
        if sensor_type:
            sensor_types.add(sensor_type)
    
    print(f"\nFound sensor types: {sensor_types}")

if __name__ == "__main__":
    check_pollution_data()
