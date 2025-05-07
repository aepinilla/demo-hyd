
"""
Data Visualization Assistant with Streamlit and Seaborn.

This script provides an interactive interface for data visualization
using Streamlit and Seaborn through a LangChain agent with a modern UI.
"""

import streamlit as st
from typing import List, Dict
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
import pandas as pd
from datetime import datetime

from src.config.settings import USER_AVATAR, BOT_AVATAR
from src.core.chat import initialize_chat_history, add_message_to_history, get_messages_for_llm
from src.core.agents import create_agent_executor
from src.core.tools import get_tools
from src.ui.components import display_chat_history, create_chat_input

# Set up Streamlit page config
st.set_page_config(
    page_title="Data Visualization Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Set up Streamlit page header
st.title("Data Visualization Assistant")
st.caption("Powered by Streamlit, Seaborn, and LangChain")
st.markdown("<hr>", unsafe_allow_html=True)

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
    st.markdown("<div class='data-container'>", unsafe_allow_html=True)
    
    # Dataset upload section
    st.subheader("Dataset Upload")
    st.markdown("<hr>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload a CSV file for visualization", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Load the dataset
            df = pd.read_csv(uploaded_file)
            
            # Store in session state
            st.session_state.dataset = df
            
            # Show dataset info with standard Streamlit components instead of shadcn
            st.success(f"Dataset loaded: {uploaded_file.name}")
            st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            
            # Show sample of the data
            st.write("### Data Preview")
            with st.container():
                st.dataframe(
                    df.head(),
                    use_container_width=True,
                    hide_index=False,
                )
            
            # Show column info
            st.write("### Column Information")
            col_info = pd.DataFrame({
                'Type': df.dtypes,
                'Non-Null Count': df.count(),
                'Unique Values': [df[col].nunique() for col in df.columns]
            })
            with st.container():
                st.dataframe(
                    col_info,
                    use_container_width=True,
                    hide_index=False,
                )
            
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Example suggestions
    st.markdown("<div class='data-container'>", unsafe_allow_html=True)
    st.subheader("Example Queries")
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Convert example queries to interactive buttons
    examples = [
        "Show a histogram of the first numeric column",
        "Create a scatter plot comparing the first two numeric columns",
        "Plot a correlation heatmap for all numeric columns",
        "Generate a line plot showing trends over time"
    ]
    
    for i, example in enumerate(examples):
        if st.button(example, key=f"example_{i}"):
            # Check if dataset is loaded before processing visualization queries
            if st.session_state.dataset is None:
                st.error("Please upload a dataset first before using visualization examples.")
            else:
                # Set the selected example as the user input
                st.session_state.user_input = example
                st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)

with col1:
    # Create a container with a clean background for the chat
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    
    st.subheader("Chat with the Assistant")
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Chat history display
    if st.session_state.messages:
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                if message["role"] == "user":
                    with st.chat_message("user", avatar=USER_AVATAR):
                        st.markdown(message["content"])
                else:
                    with st.chat_message("assistant", avatar=BOT_AVATAR):
                        st.markdown(message["content"])
    else:
        # Display welcome message with standard Streamlit
        st.info("Welcome! Upload a dataset and start asking questions about your data. I'll help you create beautiful visualizations and extract insights from your data.")
    
    # Process pre-selected example if exists in session state
    user_message = ""
    if "user_input" in st.session_state and st.session_state.user_input:
        user_message = st.session_state.user_input
        # Clear it to avoid repeated submission
        st.session_state.user_input = ""
    else:
        # Chat input field
        user_message = st.chat_input(
            "Ask a question about your data...", 
            key="chat_input",
        )
    
    if user_message:
        # Add user message to chat history
        st.session_state.messages = add_message_to_history(
            st.session_state.messages, "user", user_message
        )
        
        # Display user message
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(user_message)
        
        # Display agent thinking and response
        with st.chat_message("assistant", avatar=BOT_AVATAR):
            # Show typing indicator
            typing_container = st.empty()
            typing_container.markdown("```\nThinking...\n```")
            
            # Create callback handler and response placeholder
            st_callback = StreamlitCallbackHandler(st.container())
            response_container = st.empty()
            
            try:
                # Add dataset information as context if available
                context = ""
                if st.session_state.dataset is not None:
                    df = st.session_state.dataset
                    context = f"The user has loaded a dataset with {df.shape[0]} rows and {df.shape[1]} columns. "
                    context += f"The columns are: {', '.join(df.columns.tolist())}. "
                
                # Check if we're trying to visualize without a dataset
                if ("histogram" in user_message.lower() or 
                    "scatter" in user_message.lower() or 
                    "plot" in user_message.lower() or 
                    "heatmap" in user_message.lower() or
                    "correlation" in user_message.lower()) and st.session_state.dataset is None:
                    
                    # Handle the case where visualization is requested without a dataset
                    full_response = "Please upload a dataset first before creating visualizations."
                    typing_container.empty()
                    st.error(full_response)
                    
                else:
                    # Run the agent with the callback handler
                    response = st.session_state.agent_executor.invoke(
                        {
                            "input": context + user_message,
                            "chat_history": get_messages_for_llm()
                        },
                        {"callbacks": [st_callback]}
                    )
                    
                    # Clear typing indicator
                    typing_container.empty()
                    
                    # Display the final output
                    full_response = response["output"]
                    st.markdown(full_response)
                
                # Note: Output handling moved to the conditional block above
                
                # Add feedback buttons with standard Streamlit
                col1, col2, col3 = st.columns([1, 1, 10])
                with col1:
                    st.button("👍", key=f"like_{len(st.session_state.messages)}")
                with col2:
                    st.button("👎", key=f"dislike_{len(st.session_state.messages)}")
                
                # Show timestamp
                st.caption(f"Response generated at {datetime.now().strftime('%H:%M:%S')}")
                
                # Add assistant response to chat history
                st.session_state.messages = add_message_to_history(
                    st.session_state.messages, "assistant", full_response
                )
                
            except Exception as e:
                # Clear typing indicator
                typing_container.empty()
                
                # Show error message
                error_message = f"An error occurred: {str(e)}"
                st.error(error_message)
                
                # Add to chat history
                st.session_state.messages = add_message_to_history(
                    st.session_state.messages, "assistant", error_message
                )
    
    st.markdown("</div>", unsafe_allow_html=True)
