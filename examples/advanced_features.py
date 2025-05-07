"""
Advanced features example using the chatbot application.

This script demonstrates more advanced functionality of the chatbot,
including streaming responses and file uploading.
"""

import streamlit as st
import sys
import os

# Add the parent directory to the path so we can import the src package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.settings import APP_TITLE, USER_AVATAR, BOT_AVATAR
from src.core.chat import initialize_chat_history, add_message_to_history
from src.core.llm import get_openai_streaming_response
from src.ui.components import display_chat_history
from src.utils.helpers import read_file, ensure_directory_exists

# Set up Streamlit page
st.title(f"{APP_TITLE} - Advanced Features")

# Create temp directory if it doesn't exist
ensure_directory_exists("temp")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = initialize_chat_history()

# Sidebar with additional features
with st.sidebar:
    st.header("Advanced Features")
    
    # File upload feature
    uploaded_file = st.file_uploader("Upload a file for context", type=["txt", "csv"])
    if uploaded_file is not None:
        # Save the uploaded file
        file_path = os.path.join("temp", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File uploaded: {uploaded_file.name}")
        
        # Read the file contents
        file_contents = read_file(file_path)
        
        # Add file context to session state
        st.session_state.file_context = f"Content from file {uploaded_file.name}: {file_contents}"
    
    # Clear chat history button
    if st.button("Clear Chat History"):
        st.session_state.messages = initialize_chat_history()
        if "file_context" in st.session_state:
            del st.session_state.file_context
        st.rerun()

# Display chat history
display_chat_history(st.session_state.messages)

# Chat input
if user_message := st.chat_input("How can I help?"):
    # If we have file context, append it to the user message
    if "file_context" in st.session_state:
        system_message = st.session_state.messages[0]["content"]
        st.session_state.messages[0]["content"] = f"{system_message}\n\nAdditional context: {st.session_state.file_context}"
    
    # Add user message to chat history
    st.session_state.messages = add_message_to_history(
        st.session_state.messages, "user", user_message
    )
    
    # Display user message
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(user_message)
    
    # Generate and display assistant response with streaming
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Get streaming response
            for chunk in get_openai_streaming_response(st.session_state.messages):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            
            # Add assistant response to chat history
            st.session_state.messages = add_message_to_history(
                st.session_state.messages, "assistant", full_response
            )
            
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            message_placeholder.markdown(error_message)
            st.session_state.messages = add_message_to_history(
                st.session_state.messages, "assistant", error_message
            )
