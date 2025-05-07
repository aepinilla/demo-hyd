"""
Utility functions for the chatbot application.

This module contains helper functions used throughout the application.
"""

import os
import pandas as pd
from typing import Any, Dict, List, Union

def read_file(file_path: str) -> str:
    """
    Read a file and return its contents.
    
    Args:
        file_path (str): Path to the file to read
        
    Returns:
        str: The contents of the file
    """
    try:
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path).to_string()
        elif file_path.endswith('.txt'):
            with open(file_path, 'r') as file:
                return file.read()
        else:
            return "Unsupported file format"
    except Exception as e:
        return f"Error reading file: {str(e)}"

def ensure_directory_exists(directory_path: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory_path (str): Path to the directory
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def format_response(response: str) -> str:
    """
    Format a response for display.
    
    Args:
        response (str): Raw response text
        
    Returns:
        str: Formatted response text
    """
    # Simple formatting for now, can be expanded later
    return response.strip()
