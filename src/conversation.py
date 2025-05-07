"""
Simple conversation handler for the chatbot application.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# Load API Key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def process_message(user_message):
    """
    Process a user message and return a response.
    
    Args:
        user_message (str): The message from the user
        
    Returns:
        str: The response to the user's message
    """
    # Initialize the OpenAI client
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Create a simple message list
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_message}
    ]
    
    # Get response from OpenAI
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    
    return response.choices[0].message.content
