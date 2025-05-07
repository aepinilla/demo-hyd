"""
Data Visualization Assistant with Streamlit and Seaborn.

This script provides an interactive interface for data visualization
using Streamlit and Seaborn through a LangChain agent.
"""

import streamlit as st
from typing import List, Dict
from langchain.callbacks import StreamlitCallbackHandler
import pandas as pd

from src.config.settings import USER_AVATAR, BOT_AVATAR
from src.core.chat import initialize_chat_history, add_message_to_history, get_messages_for_llm
from src.core.agents import create_agent_executor
from src.core.tools import get_tools
from src.ui.components import display_chat_history, create_chat_input

# Set up Streamlit page config
st.set_page_config(
    page_title="Data Visualization Assistant",
    page_icon="📊",
    layout="wide"
)

# Set up Streamlit page
st.title("Data Visualization Assistant")
st.subheader("Powered by Streamlit, Seaborn, and LangChain")

# Initialize tools and agent in session state if not already present
if "agent_executor" not in st.session_state:
    tools = get_tools()
    st.session_state.agent_executor = create_agent_executor(tools)

# Initialize session state for messages and dataset
if "messages" not in st.session_state:
    st.session_state.messages = initialize_chat_history()

if "dataset" not in st.session_state:
    st.session_state.dataset = None

# Display two-column layout
col1, col2 = st.columns([3, 2])

with col2:
    st.header("Dataset Upload")
    uploaded_file = st.file_uploader("Upload a CSV file for visualization", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Load the dataset
            df = pd.read_csv(uploaded_file)
            
            # Store in session state
            st.session_state.dataset = df
            
            # Show dataset info
            st.success(f"Dataset loaded: {uploaded_file.name}")
            st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            
            # Show sample of the data
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Show column info
            st.subheader("Column Information")
            col_info = pd.DataFrame({
                'Type': df.dtypes,
                'Non-Null Count': df.count(),
                'Unique Values': [df[col].nunique() for col in df.columns]
            })
            st.dataframe(col_info)
            
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
    
    # Example suggestions
    st.header("Example Queries")
    st.markdown("""
    Try asking the assistant:
    - Load the dataset
    - Show a histogram of [column]
    - Create a scatter plot of [column] vs [column]
    - Plot a correlation heatmap for all numeric columns
    - Generate a line plot showing trends in [column]
    """)

with col1:
    st.header("Chat with the Assistant")
    
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
        
        # Display agent response with callback handler for showing steps
        with st.chat_message("assistant", avatar=BOT_AVATAR):
            st_callback = StreamlitCallbackHandler(st.container())
            
            # Create placeholder for final response
            response_placeholder = st.empty()
            
            try:
                # Add dataset information as context if available
                context = ""
                if st.session_state.dataset is not None:
                    df = st.session_state.dataset
                    context = f"The user has loaded a dataset with {df.shape[0]} rows and {df.shape[1]} columns. "
                    context += f"The columns are: {', '.join(df.columns.tolist())}. "
                
                # Run the agent with the callback handler
                response = st.session_state.agent_executor.invoke(
                    {
                        "input": context + user_message,
                        "chat_history": get_messages_for_llm()
                    },
                    {"callbacks": [st_callback]}
                )
                
                # Display the final output
                full_response = response["output"]
                response_placeholder.markdown(full_response)
                
                # Add assistant response to chat history
                st.session_state.messages = add_message_to_history(
                    st.session_state.messages, "assistant", full_response
                )
                
            except Exception as e:
                error_message = f"An error occurred: {str(e)}"
                response_placeholder.markdown(error_message)
                st.session_state.messages = add_message_to_history(
                    st.session_state.messages, "assistant", error_message
                )
