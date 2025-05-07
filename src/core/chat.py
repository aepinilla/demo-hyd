"""
Core chat functionality for the chatbot application.

This module handles the chat logic, including message processing and history management
using LangChain's StreamlitChatMessageHistory for persistence.
"""

import os
from typing import List, Dict, Any, Optional
import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from src.config.settings import SYSTEM_PROMPT
from src.core.llm import get_openai_response

def get_chat_history(key: str = "langchain_messages") -> BaseChatMessageHistory:
    """
    Get a StreamlitChatMessageHistory object for storing chat history.
    
    Args:
        key (str): Session state key for the chat history
        
    Returns:
        BaseChatMessageHistory: A chat history object
    """
    return StreamlitChatMessageHistory(key=key)

def initialize_chat_history() -> List[Dict[str, str]]:
    """
    Initialize a new chat history with the system prompt.
    
    Returns:
        List[Dict[str, str]]: Initial chat history with system message
    """
    # Initialize both traditional dict format and LangChain format
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = get_chat_history()
        
    if len(st.session_state.chat_history.messages) == 0:
        # Add system message to LangChain history
        st.session_state.chat_history.add_message(
            SystemMessage(content=SYSTEM_PROMPT)
        )
    
    # Return as dict format for compatibility
    return [{"role": "system", "content": SYSTEM_PROMPT}]

def add_message_to_history(history: List[Dict[str, str]], role: str, content: str) -> List[Dict[str, str]]:
    """
    Add a new message to both the traditional history and LangChain history.
    
    Args:
        history (List[Dict[str, str]]): Current chat history in dict format
        role (str): Role of the message sender (user or assistant)
        content (str): Content of the message
        
    Returns:
        List[Dict[str, str]]: Updated chat history in dict format
    """
    # Add to traditional dict format
    history.append({"role": role, "content": content})
    
    # Add to LangChain history if it exists
    if "chat_history" in st.session_state:
        if role == "user":
            st.session_state.chat_history.add_message(
                HumanMessage(content=content)
            )
        elif role == "assistant":
            st.session_state.chat_history.add_message(
                AIMessage(content=content)
            )
    
    return history

def get_messages_for_llm() -> List[Dict[str, str]]:
    """
    Get messages in the format expected by the LLM.
    
    Returns:
        List[Dict[str, str]]: Messages formatted for the LLM
    """
    # Start with system message
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Add messages from LangChain history
    if "chat_history" in st.session_state:
        for msg in st.session_state.chat_history.messages:
            if isinstance(msg, SystemMessage):
                continue  # Skip system message as we already added it
            elif isinstance(msg, HumanMessage):
                messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                messages.append({"role": "assistant", "content": msg.content})
    
    return messages

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
    response = get_openai_response(get_messages_for_llm())
    
    return response
