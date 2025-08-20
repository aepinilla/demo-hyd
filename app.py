"""
Data Visualization Assistant with Streamlit and Seaborn.

This script provides an interactive interface for data visualization
using Streamlit and Seaborn through a LangChain agent with a modern UI.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import json
import plotly.express as px
import plotly.graph_objects as go
import random
import os
from typing import List, Dict, Any, Optional
from src.utils.dataframe_utils import prepare_dataframe_for_streamlit
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
import pandas as pd
from datetime import datetime
import traceback

from src.config.settings import USER_AVATAR, BOT_AVATAR
from src.core.chat import initialize_chat_history, add_message_to_history, get_messages_for_llm
from src.core.agents import create_agent_executor
from src.core.tools.tools import get_tools_by_context
from src.utils.auto_load import auto_load_processed_data
from src.utils.debug_utils import setup_debug_mode, toggle_debug_mode, debug_log, display_debug_sidebar, exception_handler
# from src.ui.components import display_chat_history, create_chat_input

# Set up Streamlit page config
st.set_page_config(
    page_title="Hack Your District 2025",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize debug mode
setup_debug_mode()

# Load custom CSS
with open(".streamlit/custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Display debug sidebar if debug mode is enabled
display_debug_sidebar()

# Add debug toggle in sidebar
with st.sidebar:
    if st.button("🐞 Toggle Debug Mode"):
        is_debug = toggle_debug_mode()
        st.success(f"Debug mode {'enabled' if is_debug else 'disabled'}")
        st.rerun()

# Create main container
with st.container():
    # Logo section
    st.markdown('<div class="logo-section">', unsafe_allow_html=True)
    st.image("src/ui/logos/tu-berlin-logo-long-red.svg")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Header section
    st.title("Hack Your District 2025")
    st.caption("Data Visualization Assistant")
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Log app initialization
    debug_log("Application initialized", context={"initial_state": "ready"})

# Initialize tools and agent
# Initialize different agent executors for different data contexts
if "standard_agent" not in st.session_state:
    # Create an agent executor for standard CSV data tools
    standard_tools = get_tools_by_context("standard")
    standard_agent = create_agent_executor(standard_tools)
    st.session_state.standard_agent = standard_agent

if "sensor_agent" not in st.session_state:
    # Create an agent executor for sensor data tools
    sensor_tools = get_tools_by_context("sensor")
    sensor_agent = create_agent_executor(sensor_tools)
    st.session_state.sensor_agent = sensor_agent

if "combined_agent" not in st.session_state:
    # Create an agent executor with all tools (for fallback)
    all_tools = get_tools_by_context("all")
    combined_agent = create_agent_executor(all_tools)
    st.session_state.combined_agent = combined_agent
    
# Set the default agent
if "current_agent" not in st.session_state:
    st.session_state.current_agent = "combined_agent"

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = initialize_chat_history()

if "dataset" not in st.session_state:
    st.session_state.dataset = None
    
    # Try to automatically load the most recent processed data
    auto_load_result = auto_load_processed_data()
    if auto_load_result:
        st.info(auto_load_result)
    
# Initialize user_input in session state if not present
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
    
# Initialize examples in session state to maintain consistent order
if "dataset_examples" not in st.session_state:
    st.session_state.dataset_examples = [
        "Show a histogram of the first numeric column",
        "Create a scatter plot comparing the first two numeric columns",
        "Plot a correlation heatmap for all numeric columns",
    ]
    random.shuffle(st.session_state.dataset_examples)
    
if "sensor_examples" not in st.session_state:
    st.session_state.sensor_examples = [
        "What variables are available in the sensor API and show their statistics",
        "Fetch the latest sensor data and show a summary",
        "Get the latest particulate matter readings from dust sensors"
    ]
    random.shuffle(st.session_state.sensor_examples)
    
if "map_examples" not in st.session_state:
    st.session_state.map_examples = [
        "Create an interactive map showing P1 (PM10) pollution levels across sensor locations",
        "Show a map visualization of P2 (PM2.5) concentrations with color gradient",
    ]
    random.shuffle(st.session_state.map_examples)

# Main content section
st.markdown('<div class="content-section">', unsafe_allow_html=True)
col1, col2 = st.columns([3, 2])

with col2:
    st.markdown("<div class='data-container'>", unsafe_allow_html=True)
    
    # Example suggestions
    st.markdown("<div class='data-container'>", unsafe_allow_html=True)
    st.subheader("Example Queries")
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Tabs for different types of examples
    data_tab, sensor_tab = st.tabs(["Dataset Examples", "Sensor.Community"])
    
    with data_tab:
        # Use dataset examples from session state to maintain consistency
        for i, example in enumerate(st.session_state.dataset_examples):
            if st.button(example, key=f"example_dataset_{i}"):
                # Store the example query in session state and force rerun
                st.session_state["user_input"] = example
                st.rerun()
    
    with sensor_tab:
        st.markdown("### Data Fetching & Visualization")
        
        # Use sensor examples from session state to maintain consistency
        for i, example in enumerate(st.session_state.sensor_examples):
            if st.button(example, key=f"example_general_{i}"):
                # Store the example query in session state and force rerun
                st.session_state["user_input"] = example
                st.rerun()
        
        # Add buttons for map examples
        st.markdown("### Map Visualizations")
        
        # Use map examples from session state to maintain consistency
        for i, example in enumerate(st.session_state.map_examples):
            if st.button(example, key=f"example_map_{i}"):
                # Store the example query in session state and force rerun
                st.session_state["user_input"] = example
                st.rerun()
                
        # Time Series example buttons removed as requested
    
    st.markdown("</div>", unsafe_allow_html=True)

     # Dataset upload section
    st.subheader("Dataset Upload")
    st.markdown("<hr>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload a CSV file for visualization", type=["csv"])
    
    st.markdown("""
    **Example Dataset**  
    Download an example air pollution dataset from [Kaggle](https://www.kaggle.com/datasets/sazidthe1/global-air-pollution-data)
    """)
    
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
        st.info("Welcome! You can ask me questions about the latests data available from sensor.community. Alternatively, you can upload a dataset and start asking questions about your data. I'll help you create beautiful visualizations and extract insights from your data.")
    
    # We don't need to initialize chat_input in session state
    # Streamlit manages this automatically
    
    # Process pre-selected example if exists in session state
    user_message = ""
    if "user_input" in st.session_state and st.session_state["user_input"]:
        user_message = st.session_state["user_input"]
        # Clear it to avoid repeated submission
        st.session_state["user_input"] = ""
    
    # Get input from chat field if provided
    # Note: We don't need to clear chat_input as Streamlit manages this automatically
    # and attempting to set it directly causes StreamlitValueAssignmentNotAllowedError
    if "chat_input" in st.session_state and st.session_state.chat_input:
        user_message = st.session_state.chat_input
    
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
                # Log user message for debugging
                debug_log(f"Processing user message: {user_message}", context={"message_length": len(user_message)})
                
                # Add dataset information as context if available
                context = ""
                if st.session_state.dataset is not None:
                    df = st.session_state.dataset
                    context = f"The user has uploaded a dataset with {len(df)} rows and {len(df.columns)} columns. "
                    context += f"The columns are: {', '.join(df.columns)}. "
                    debug_log("Dataset available", context={"rows": len(df), "columns": list(df.columns)})
                
                # Detect if this is likely a sensor visualization query
                is_sensor_visualization = ("sensor" in user_message.lower() or
                                         "pollution" in user_message.lower() or
                                         "pm10" in user_message.lower() or
                                         "pm2.5" in user_message.lower() or
                                         "p1" in user_message.lower() or 
                                         "p2" in user_message.lower())
                
                debug_log(f"Query classification: {'sensor' if is_sensor_visualization else 'standard'}", 
                         context={"is_sensor_visualization": is_sensor_visualization})
                
                if ("histogram" in user_message.lower() or 
                    "scatter" in user_message.lower() or 
                    "plot" in user_message.lower() or 
                    "heatmap" in user_message.lower() or
                    "correlation" in user_message.lower()) and st.session_state.dataset is None and not is_sensor_visualization:
                    
                    # Handle the case where visualization is requested without a dataset
                    full_response = "Please upload a dataset first before creating visualizations."
                    typing_container.empty()
                    st.error(full_response)
                    
                    # Add to chat history
                    st.session_state.messages = add_message_to_history(
                        st.session_state.messages, "assistant", full_response
                    )
                    
                else:
                    # Select the appropriate agent based on the query content
                    if is_sensor_visualization:
                        # Use the sensor-specific agent for sensor data queries
                        st.session_state.current_agent = "sensor_agent"
                        agent_to_use = st.session_state.sensor_agent
                        st.info("Using specialized sensor data visualization tools")
                        debug_log("Selected sensor agent", context={"agent_type": "sensor_agent"})
                    elif st.session_state.dataset is not None:
                        # Use the standard agent for CSV data queries when a dataset is loaded
                        st.session_state.current_agent = "standard_agent"
                        agent_to_use = st.session_state.standard_agent
                        debug_log("Selected standard agent", context={"agent_type": "standard_agent"})
                    else:
                        # Use the combined agent as a fallback
                        st.session_state.current_agent = "combined_agent"
                        agent_to_use = st.session_state.combined_agent
                        debug_log("Selected combined agent", context={"agent_type": "combined_agent"})
                    
                    # Run the selected agent with the callback handler
                    debug_log("Invoking agent", context={"input_length": len(context + user_message)})
                    try:
                        response = agent_to_use.invoke(
                            {
                                "input": context + user_message,
                                "chat_history": get_messages_for_llm()
                            },
                            {"callbacks": [st_callback]}
                        )
                        debug_log("Agent response received", context={"response_length": len(response["output"]) if "output" in response else 0})
                    except Exception as e:
                        debug_log(f"Agent invocation error: {str(e)}", level="ERROR", 
                                 context={"error_type": type(e).__name__, "traceback": traceback.format_exc()})
                        raise
                    
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
                
                # Log the error
                debug_log(f"Error processing message: {str(e)}", level="ERROR", 
                         context={"error_type": type(e).__name__, "traceback": traceback.format_exc()})
                
                # Show error message
                error_message = f"An error occurred: {str(e)}"
                st.error(error_message)
                
                # In debug mode, show more details
                if "debug_mode" in st.session_state and st.session_state.debug_mode:
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())
                
                # Add to chat history
                st.session_state.messages = add_message_to_history(
                    st.session_state.messages, "assistant", error_message
                )
    
    # Always show chat input field at the end of all content
    st.chat_input(
        "Ask a question about your data...", 
        key="chat_input"
    )
    
    st.markdown("</div>", unsafe_allow_html=True)
