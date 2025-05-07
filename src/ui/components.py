"""
UI components for the Streamlit chatbot application.

This module contains reusable UI components for the Streamlit interface.
"""

import streamlit as st
from typing import List, Dict, Any, Callable

from src.config.settings import USER_AVATAR, BOT_AVATAR

def display_chat_history(messages: List[Dict[str, str]]) -> None:
    """
    Display the chat history in the Streamlit interface.
    
    Args:
        messages (List[Dict[str, str]]): Chat history to display
    """
    # Filter out system messages and display only user/assistant messages
    for message in messages:
        if message["role"] in ["user", "assistant"]:
            avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

def create_chat_input() -> str:
    """
    Create a chat input field in the Streamlit interface.
    
    Returns:
        str: User input or None if no input
    """
    return st.chat_input("How can I help?")

def display_streaming_response(response_generator: Callable) -> str:
    """
    Display a streaming response in the Streamlit interface.
    
    Args:
        response_generator: Generator function that yields response chunks
        
    Returns:
        str: Full response text
    """
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            for chunk in response_generator:
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            return full_response
            
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            message_placeholder.markdown(error_message)
            return error_message
