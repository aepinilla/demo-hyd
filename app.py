"""
Main application entry point for the Streamlit chatbot.

This script integrates all components of the chatbot application
and provides the main Streamlit interface.
"""

import streamlit as st
from typing import List, Dict

from src.config.settings import APP_TITLE, USER_AVATAR, BOT_AVATAR
from src.core.chat import initialize_chat_history, add_message_to_history
from src.core.llm import get_openai_streaming_response
from src.ui.components import display_chat_history, create_chat_input, display_streaming_response

# Set up Streamlit page
st.title(APP_TITLE)

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = initialize_chat_history()

# Display chat history
display_chat_history(st.session_state.messages)

# Chat input
if user_message := create_chat_input():
    # Add user message to chat history
    st.session_state.messages = add_message_to_history(
        st.session_state.messages, "user", user_message
    )
    
    # Display user message
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(user_message)
    
    # Function to generate streaming response
    def response_generator():
        return get_openai_streaming_response(st.session_state.messages)
    
    # Display streaming response
    full_response = display_streaming_response(response_generator())
    
    # Add assistant response to chat history
    st.session_state.messages = add_message_to_history(
        st.session_state.messages, "assistant", full_response
    )
