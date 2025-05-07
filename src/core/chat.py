"""
Core chat functionality for the chatbot application.

This module handles the chat logic, including message processing and history management.
"""

import os
from typing import List, Dict, Any

from src.config.settings import SYSTEM_PROMPT
from src.core.llm import get_openai_response

def initialize_chat_history() -> List[Dict[str, str]]:
    """
    Initialize a new chat history with the system prompt.
    
    Returns:
        List[Dict[str, str]]: Initial chat history with system message
    """
    return [{"role": "system", "content": SYSTEM_PROMPT}]

def add_message_to_history(history: List[Dict[str, str]], role: str, content: str) -> List[Dict[str, str]]:
    """
    Add a new message to the chat history.
    
    Args:
        history (List[Dict[str, str]]): Current chat history
        role (str): Role of the message sender (user or assistant)
        content (str): Content of the message
        
    Returns:
        List[Dict[str, str]]: Updated chat history
    """
    history.append({"role": role, "content": content})
    return history

def process_user_message(history: List[Dict[str, str]], user_message: str) -> str:
    """
    Process a user message and generate a response.
    
    Args:
        history (List[Dict[str, str]]): Current chat history
        user_message (str): Message from the user
        
    Returns:
        str: Response from the assistant
    """
    # Add user message to history
    updated_history = add_message_to_history(history, "user", user_message)
    
    # Get response from LLM
    response = get_openai_response(updated_history)
    
    return response
