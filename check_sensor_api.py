"""
Sensor.community API Checker

This script tests the sensor.community API endpoints and helps debug data retrieval issues.
It runs the API tests and displays the results in a readable format.
"""

import pandas as pd
import json
from src.utils.api_tester import test_all_endpoints, test_sensor_types
from src.utils.sensor_api import fetch_latest_data, fetch_average_data
import time

def print_separator():
    print("\n" + "="*80 + "\n")

def main():
    print("SENSOR.COMMUNITY API CHECKER")
    print_separator()
    
    # Test all endpoints
    print("1. TESTING ALL MAIN API ENDPOINTS")
    results = test_all_endpoints()
    
    # Check which endpoints are working
    working_endpoints = [endpoint for endpoint, result in results.items() if result.get("success")]
    print(f"\nWorking endpoints: {len(working_endpoints)}/{len(results)}")
    for endpoint in working_endpoints:
        print(f"  ✓ {endpoint}")
    
    failed_endpoints = [endpoint for endpoint, result in results.items() if not result.get("success")]
    if failed_endpoints:
        print(f"\nFailed endpoints: {len(failed_endpoints)}/{len(results)}")
        for endpoint in failed_endpoints:
            print(f"  ✗ {endpoint} - Error: {results[endpoint].get('error')}")
    
    print_separator()
    
    # Detect sensor types
    print("2. DETECTING AVAILABLE SENSOR TYPES")
    sensor_types = test_sensor_types()
    if sensor_types:
        print(f"Found {len(sensor_types)} sensor types:")
        for sensor_type in sensor_types:
            print(f"  • {sensor_type}")
    else:
        print("No sensor types detected. API might be returning empty data.")
    
    print_separator()
    
    # Test fetch_latest_data function
    print("3. TESTING fetch_latest_data FUNCTION")
    
    # Try with no parameters
    print("\nFetching all latest data (no filters)...")
    start_time = time.time()
    all_data = fetch_latest_data()
    elapsed = time.time() - start_time
    
    if not all_data.empty:
        print(f"Success! Retrieved {len(all_data)} records in {elapsed:.2f} seconds")
        print(f"Data shape: {all_data.shape}")
        print("\nSample data:")
        print(all_data.head(3))
        
        # Check value types
        if 'value_type' in all_data.columns:
            value_types = all_data['value_type'].unique()
            print(f"\nValue types found: {value_types}")
        
        # Check sensor types
        if 'sensor_type' in all_data.columns:
            sensor_types = all_data['sensor_type'].unique()
            print(f"Sensor types found: {sensor_types}")
    else:
        print("No data returned from fetch_latest_data")
    
    # Try with SDS011 filter
    print("\nFetching SDS011 (dust) sensor data...")
    sds011_data = fetch_latest_data(data_type="SDS011")
    
    if not sds011_data.empty:
        print(f"Success! Retrieved {len(sds011_data)} SDS011 records")
        print("\nSample data:")
        print(sds011_data.head(3))
    else:
        print("No SDS011 data returned")
    
    # Try with DHT22 filter
    print("\nFetching DHT22 (temperature/humidity) sensor data...")
    dht22_data = fetch_latest_data(data_type="DHT22")
    
    if not dht22_data.empty:
        print(f"Success! Retrieved {len(dht22_data)} DHT22 records")
        print("\nSample data:")
        print(dht22_data.head(3))
    else:
        print("No DHT22 data returned")
    
    print_separator()
    
    # Test fetch_average_data function
    print("4. TESTING fetch_average_data FUNCTION")
    
    # Try with different timespans
    for timespan in ['5m', '1h', '24h']:
        print(f"\nFetching {timespan} average data...")
        avg_data = fetch_average_data(timespan=timespan)
        
        if not avg_data.empty:
            print(f"Success! Retrieved {len(avg_data)} records")
            print("\nSample data:")
            print(avg_data.head(3))
        else:
            print(f"No {timespan} average data returned")
    
    print_separator()
    print("API TESTING COMPLETE")

if __name__ == "__main__":
    main()
