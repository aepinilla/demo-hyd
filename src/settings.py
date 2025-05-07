"""
Simple settings file for the chatbot application.
"""

# Model configuration
gpt_model = "gpt-4o"

# System prompt
system_init_prompt = """You are a helpful assistant designed to provide information and assistance.
Please respond to the user's questions in a clear and concise manner.
"""

# User prompt template
user_init_prompt = """
The question is: {}. 
"""
