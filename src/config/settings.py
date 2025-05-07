"""
Configuration settings for the chatbot application.

This module contains all the configuration settings for the application,
including model settings and prompt templates.
"""

# Model configuration
GPT_MODEL = "gpt-4o"

# Avatar configuration
USER_AVATAR = "👤"
BOT_AVATAR = "🤖"

# System prompt
SYSTEM_PROMPT = """You are a helpful assistant designed to provide information and assistance.
Please respond to the user's questions in a clear and concise manner.
"""

# User prompt template
USER_PROMPT_TEMPLATE = """
The question is: {}. 
"""

# Application settings
APP_TITLE = "Simple Chatbot"
