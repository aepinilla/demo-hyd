"""
Simplified callable functions for the basic chatbot application.
"""

import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# Load OpenAI API Token From the .env File
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def read_file(file_path: str):
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

def process_message(user_message: str):
    """
    Process a user message and return a response using OpenAI.
    
    Args:
        user_message (str): The message from the user
        
    Returns:
        str: The response to the user's message
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_message}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error processing message: {str(e)}"
