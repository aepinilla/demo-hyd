"""
API Tester for sensor.community.

This module provides functions to directly test the sensor.community API endpoints.
"""

import requests
import json
import pandas as pd
from typing import Dict, Any, Optional, List

def test_api_endpoint(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Test a sensor.community API endpoint directly.
    
    Args:
        endpoint: The API endpoint to test (e.g., 'static/v1/data.json')
        params: Optional query parameters
        
    Returns:
        Dictionary with test results
    """
    base_url = "https://data.sensor.community"
    headers = {'User-Agent': 'HackYourDistrict2025/1.0 (data-visualization-assistant)'}
    
    url = f"{base_url}/{endpoint}"
    
    try:
        print(f"Testing API endpoint: {url}")
        if params:
            print(f"With parameters: {params}")
            
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        
        # Try to parse as JSON
        data = response.json()
        
        # Check if it's a list or dict
        if isinstance(data, list):
            print(f"Success! Received a list with {len(data)} items.")
            
            # Return sample of data
            sample_size = min(3, len(data))
            sample = data[:sample_size]
            
            # Convert to DataFrame if it's a list of records
            if sample_size > 0 and isinstance(sample[0], dict):
                df = pd.DataFrame(sample)
                print(f"Sample data as DataFrame:\n{df.head()}")
            
            return {
                "success": True,
                "count": len(data),
                "sample": sample
            }
        else:
            print(f"Success! Received a dictionary with {len(data.keys())} keys.")
            return {
                "success": True,
                "data": data
            }
            
    except requests.exceptions.RequestException as e:
        print(f"Error accessing API: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }
    except json.JSONDecodeError:
        print("Received non-JSON response")
        return {
            "success": False,
            "error": "Non-JSON response",
            "content": response.text[:200] + "..." if len(response.text) > 200 else response.text
        }
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

def test_all_endpoints() -> Dict[str, Any]:
    """
    Test all main sensor.community API endpoints.
    
    Returns:
        Dictionary with test results for each endpoint
    """
    endpoints = [
        "static/v1/data.json",
        "static/v2/data.json",
        "static/v2/data.1h.json",
        "static/v2/data.24h.json",
        "static/v2/data.dust.min.json",
        "static/v2/data.temp.min.json"
    ]
    
    results = {}
    
    for endpoint in endpoints:
        print(f"\nTesting endpoint: {endpoint}")
        results[endpoint] = test_api_endpoint(endpoint)
    
    return results

def test_country_filter(country_code: str) -> Dict[str, Any]:
    """
    Test the country filter endpoint.
    
    Args:
        country_code: Country code to filter by (e.g., 'DE')
        
    Returns:
        Dictionary with test results
    """
    endpoint = "airrohr/v1/filter/"
    params = {"country": country_code}
    
    return test_api_endpoint(endpoint, params)

def test_sensor_types() -> List[str]:
    """
    Get a list of available sensor types from the API data.
    
    Returns:
        List of sensor type names found in the data
    """
    result = test_api_endpoint("static/v1/data.json")
    
    if not result["success"] or "sample" not in result:
        return []
    
    sensor_types = set()
    
    # Extract sensor types from the sample data
    for item in result["sample"]:
        if "sensor" in item and "sensor_type" in item["sensor"]:
            sensor_type = item["sensor"]["sensor_type"].get("name")
            if sensor_type:
                sensor_types.add(sensor_type)
    
    return list(sensor_types)

if __name__ == "__main__":
    # Run tests when script is executed directly
    print("Testing sensor.community API...")
    results = test_all_endpoints()
    
    print("\nTesting country filter for Germany (DE)...")
    de_results = test_country_filter("DE")
    
    print("\nDetecting available sensor types...")
    sensor_types = test_sensor_types()
    print(f"Found sensor types: {sensor_types}")
