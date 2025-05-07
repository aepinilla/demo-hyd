"""
Utility functions for the chatbot application.

This module contains helper functions that serve as tools for the LangChain agent.
"""

import os
import pandas as pd
import json
from typing import Any, Dict, List, Union, Optional

def read_file(file_path: str) -> str:
    """
    Read a file and return its contents.
    
    Args:
        file_path (str): Path to the file to read
        
    Returns:
        str: The contents of the file or an error message
    """
    try:
        if not os.path.exists(file_path):
            return f"Error: File not found at path {file_path}"
            
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            # Convert to Markdown table for better display in Streamlit
            return f"CSV file contents:\n\n```\n{df.to_string()}\n```\n\nFile has {len(df)} rows and {len(df.columns)} columns."
        elif file_path.endswith('.json'):
            with open(file_path, 'r') as file:
                data = json.load(file)
            return f"JSON file contents:\n\n```json\n{json.dumps(data, indent=2)}\n```"
        elif file_path.endswith('.txt'):
            with open(file_path, 'r') as file:
                content = file.read()
            return f"Text file contents:\n\n```\n{content}\n```"
        else:
            with open(file_path, 'r') as file:
                content = file.read()
            return f"File contents ({os.path.basename(file_path)}):\n\n```\n{content}\n```"
    except Exception as e:
        return f"Error reading file: {str(e)}"

def ensure_directory_exists(directory_path: str) -> str:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory_path (str): Path to the directory
        
    Returns:
        str: Success message or error message
    """
    try:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            return f"Successfully created directory: {directory_path}"
        else:
            return f"Directory already exists: {directory_path}"
    except Exception as e:
        return f"Error creating directory: {str(e)}"

def format_response(response: str, format_type: str = "markdown") -> str:
    """
    Format a response for better readability.
    
    Args:
        response (str): Raw response text
        format_type (str): The type of formatting to apply (markdown, bullet, numbered)
        
    Returns:
        str: Formatted response text
    """
    if format_type == "markdown":
        # Just return the response as is, Streamlit supports markdown
        return response.strip()
    elif format_type == "bullet":
        # Convert text to bullet points
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        return '\n\n'.join([f"• {line}" for line in lines])
    elif format_type == "numbered":
        # Convert text to numbered list
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        return '\n\n'.join([f"{i+1}. {line}" for i, line in enumerate(lines)])
    else:
        return response.strip()

def list_files(directory_path: str = ".", pattern: str = "*") -> str:
    """
    List files in a directory matching a pattern.
    
    Args:
        directory_path (str): Path to the directory to list files from
        pattern (str): Pattern to match files against (e.g., "*.csv")
        
    Returns:
        str: List of files matching the pattern
    """
    try:
        import glob
        files = glob.glob(os.path.join(directory_path, pattern))
        if not files:
            return f"No files found matching pattern '{pattern}' in {directory_path}"
        file_list = "\n".join([f"- {os.path.basename(f)}" for f in files])
        return f"Files matching pattern '{pattern}' in {directory_path}:\n\n{file_list}"
    except Exception as e:
        return f"Error listing files: {str(e)}"

def analyze_data(file_path: str, analysis_type: str = "summary") -> str:
    """
    Perform basic data analysis on a CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        analysis_type (str): Type of analysis to perform (summary, describe, columns)
        
    Returns:
        str: Analysis results
    """
    try:
        if not file_path.endswith('.csv'):
            return "Error: This function only works with CSV files"
            
        if not os.path.exists(file_path):
            return f"Error: File not found at path {file_path}"
            
        df = pd.read_csv(file_path)
        
        if analysis_type == "summary":
            result = f"Data Summary:\n\n"
            result += f"- Rows: {len(df)}\n"
            result += f"- Columns: {len(df.columns)}\n"
            result += f"- Column names: {', '.join(df.columns)}\n"
            result += f"- Data types:\n"
            for col, dtype in df.dtypes.items():
                result += f"  - {col}: {dtype}\n"
            return result
            
        elif analysis_type == "describe":
            description = df.describe()
            return f"Statistical Description:\n\n```\n{description.to_string()}\n```"
            
        elif analysis_type == "columns":
            column_info = []
            for col in df.columns:
                n_unique = df[col].nunique()
                n_missing = df[col].isna().sum()
                column_info.append(f"- {col}: {n_unique} unique values, {n_missing} missing values")
            return "Column Information:\n\n" + "\n".join(column_info)
            
        else:
            return f"Unknown analysis type: {analysis_type}"
    except Exception as e:
        return f"Error analyzing data: {str(e)}"

