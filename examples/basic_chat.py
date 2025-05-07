"""
Basic chat example using the chatbot application.

This script demonstrates the basic functionality of the chatbot.
"""

import streamlit as st
import sys
import os

# Add the parent directory to the path so we can import the src package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.settings import APP_TITLE
from src.core.chat import initialize_chat_history, add_message_to_history, process_user_message
from src.ui.components import display_chat_history, create_chat_input

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
    
    # Display user message (will be shown in the next rerun)
    st.rerun()
    
    # Process user message and get response
    response = process_user_message(st.session_state.messages, user_message)
    
    # Add assistant response to chat history
    st.session_state.messages = add_message_to_history(
        st.session_state.messages, "assistant", response
    )
    
    # Rerun to display the assistant message
    st.rerun()
