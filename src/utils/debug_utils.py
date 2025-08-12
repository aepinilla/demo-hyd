"""
Debug utilities for Streamlit applications.

This module provides debugging tools and utilities for Streamlit applications.
"""

import streamlit as st
import inspect
import sys
import os
from typing import Any, Dict, List, Optional
import logging
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("streamlit_debug")

def setup_debug_mode():
    """
    Set up debug mode for the Streamlit application.
    This should be called at the start of your app.py file.
    """
    # Create a debug flag in session state if it doesn't exist
    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False
    
    # Create a log container in session state
    if "debug_logs" not in st.session_state:
        st.session_state.debug_logs = []

def toggle_debug_mode():
    """Toggle debug mode on/off"""
    if "debug_mode" in st.session_state:
        st.session_state.debug_mode = not st.session_state.debug_mode
        return st.session_state.debug_mode
    return False

def debug_log(message: str, level: str = "INFO", context: Dict[str, Any] = None):
    """
    Log a debug message and add it to the session state logs.
    
    Args:
        message: The message to log
        level: Log level (INFO, WARNING, ERROR, DEBUG)
        context: Additional context data to log
    """
    # Get caller information
    frame = inspect.currentframe().f_back
    filename = frame.f_code.co_filename
    lineno = frame.f_lineno
    function = frame.f_code.co_name
    
    # Format the log message
    log_entry = {
        "message": message,
        "level": level,
        "file": os.path.basename(filename),
        "line": lineno,
        "function": function,
        "context": context or {}
    }
    
    # Add to session state logs
    if "debug_logs" in st.session_state:
        st.session_state.debug_logs.append(log_entry)
        # Keep only the last 100 logs
        if len(st.session_state.debug_logs) > 100:
            st.session_state.debug_logs.pop(0)
    
    # Log to the logger
    log_message = f"{message} - {os.path.basename(filename)}:{lineno} in {function}"
    if context:
        log_message += f" - Context: {context}"
        
    if level == "ERROR":
        logger.error(log_message)
    elif level == "WARNING":
        logger.warning(log_message)
    elif level == "DEBUG":
        logger.debug(log_message)
    else:
        logger.info(log_message)

def display_debug_sidebar():
    """Display debug information in the sidebar if debug mode is enabled"""
    if "debug_mode" in st.session_state and st.session_state.debug_mode:
        with st.sidebar:
            st.subheader("🐞 Debug Information")
            st.markdown("---")
            
            # System info
            st.write("### System Info")
            st.write(f"Python version: {sys.version}")
            st.write(f"Streamlit version: {st.__version__}")
            
            # Session state
            st.write("### Session State")
            # Filter out large objects like dataframes
            filtered_state = {
                k: (f"{type(v).__name__} (size: {sys.getsizeof(v)})" 
                   if sys.getsizeof(v) > 1000 else v)
                for k, v in st.session_state.items()
                if k != "debug_logs"  # Don't show the logs themselves
            }
            st.json(filtered_state)
            
            # Debug logs
            st.write("### Debug Logs")
            if "debug_logs" in st.session_state and st.session_state.debug_logs:
                for log in reversed(st.session_state.debug_logs[-10:]):  # Show last 10 logs
                    color = "blue"
                    if log["level"] == "WARNING":
                        color = "orange"
                    elif log["level"] == "ERROR":
                        color = "red"
                    
                    st.markdown(
                        f"<div style='color:{color};'>"
                        f"<b>{log['level']}</b>: {log['message']}<br>"
                        f"<small>{log['file']}:{log['line']} in {log['function']}</small>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
            else:
                st.write("No logs yet")
            
            # Clear logs button
            if st.button("Clear Logs"):
                st.session_state.debug_logs = []
                st.rerun()

def exception_handler(func):
    """
    Decorator to catch and log exceptions.
    
    Usage:
        @exception_handler
        def my_function():
            # Your code here
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            debug_log(
                f"Exception: {error_msg}", 
                level="ERROR",
                context={"stack_trace": stack_trace}
            )
            # Re-raise the exception if not in debug mode
            if "debug_mode" not in st.session_state or not st.session_state.debug_mode:
                raise
            else:
                st.error(f"Error in {func.__name__}: {error_msg}")
                with st.expander("Stack Trace"):
                    st.code(stack_trace)
    return wrapper
