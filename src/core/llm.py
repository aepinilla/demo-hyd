"""
Language model integration for the chatbot application.

This module handles the integration with OpenAI's language models.
"""

import os
from typing import List, Dict, Any, Optional, Generator
from openai import OpenAI
from dotenv import load_dotenv

from src.config.settings import GPT_MODEL

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_openai_response(messages: List[Dict[str, str]]) -> str:
    """
    Get a response from the OpenAI API.
    
    Args:
        messages (List[Dict[str, str]]): List of message dictionaries
        
    Returns:
        str: Response from the OpenAI API
    """
    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def get_openai_streaming_response(messages: List[Dict[str, str]]) -> Generator[str, None, None]:
    """
    Get a streaming response from the OpenAI API.
    
    Args:
        messages (List[Dict[str, str]]): List of message dictionaries
        
    Yields:
        str: Chunks of the response from the OpenAI API
    """
    try:
        stream = client.chat.completions.create(
            model=GPT_MODEL,
            messages=messages,
            stream=True
        )
        
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content
    except Exception as e:
        yield f"Error: {str(e)}"
