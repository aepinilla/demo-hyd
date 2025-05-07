"""
Configuration settings for the data visualization assistant.

This module contains settings for the Streamlit application and 
LangChain integration focused on data visualization with Seaborn.
"""

# Model configuration
GPT_MODEL = "gpt-4o"

# Avatar configuration
USER_AVATAR = "👤"  # User emoji
BOT_AVATAR = "📊"  # Chart emoji

# System prompt
SYSTEM_PROMPT = """You are a Data Visualization Assistant specializing in creating insightful visualizations using Seaborn and Streamlit.

Your primary goal is to help users analyze and visualize their data through clear, informative plots. You have access to several visualization tools that can create different types of charts including histograms, scatter plots, line plots, and correlation heatmaps.

When responding to the user:
1. If the user wants to visualize data, use the appropriate visualization tool
2. Explain what the visualization shows and provide insights about the data
3. Suggest follow-up visualizations that might reveal additional insights
4. Be clear and educational in your explanations, helping the user understand both the data and visualization techniques

If the user hasn't uploaded a dataset or if you can't complete a task with your available tools, explain what information you need and guide them accordingly.
"""

# User prompt template
USER_PROMPT_TEMPLATE = """
Visualization request: {}. 
"""

# Application settings
APP_TITLE = "Data Visualization Assistant"

# Agent settings
AGENT_TEMPERATURE = 0
AGENT_VERBOSE = True

# Memory settings
MEMORY_KEY = "langchain_messages"

# Tool settings
TOOL_DESCRIPTIONS = {
    "load_dataset": "Load a dataset from a CSV file for visualization. Must be called before other visualization tools.",
    "plot_histogram": "Create a histogram to visualize the distribution of a numerical column.",
    "plot_scatter": "Create a scatter plot to show the relationship between two numerical columns.",
    "plot_line": "Create a line plot to show trends in data over a continuous variable.",
    "plot_heatmap": "Create a correlation heatmap to visualize relationships between multiple variables."
}
