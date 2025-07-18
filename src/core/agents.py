"""
Agent implementations for the chatbot application.

This module handles the creation and execution of LangChain agents.
"""

import streamlit as st
from typing import List, Dict, Any, Optional
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.core.custom_parser import GracefulReActOutputParser

from src.config.settings import GPT_MODEL, SYSTEM_PROMPT

def create_agent_executor(tools, verbose=True):
    """
    Create an agent executor with the given tools.
    
    Args:
        tools (list): List of tools available to the agent
        verbose (bool): Whether to enable verbose mode
        
    Returns:
        AgentExecutor: An agent executor that can run the agent
    """
    try:
        # Try to use OpenAI with API key from environment variables
        llm = ChatOpenAI(
            model=GPT_MODEL, 
            temperature=0, 
            streaming=True
        )
    except Exception as e:
        # If OpenAI fails, use a simple fallback handler that returns a message about the error
        st.warning(f"OpenAI API connection failed: {str(e)}\nUsing fallback mode with limited functionality.")
        
        # Import a local fallback handler
        from src.core.fallback_handler import FallbackHandler
        llm = FallbackHandler()
    
    # Get tool names for the prompt
    tool_names = [tool.name for tool in tools]
    
    # Create a React-compatible prompt template with clearer instructions
    template = """
    {system_prompt}

    You have access to the following tools:

    {tools}

    IMPORTANT: You MUST use the following format correctly for the system to work:

    To use a tool:
    ```
    Thought: [your reasoning about what to do next]
    Action: [name of the tool to use - must be one of the available tools]
    Action Input: [input parameters for the tool - must be properly formatted]
    ```

    After you receive the tool output, you MUST continue with another action OR provide your final answer:

    When you have your final answer:
    ```
    Thought: [your reasoning about the final answer]
    Final Answer: [your complete response to the human's question]
    ```

    ALWAYS follow this format exactly. If you don't use a tool, go straight to Final Answer.
    Never write 'Thought:' without immediately following it with either 'Action:' or 'Final Answer:'.

    Begin!

    Previous conversation history:
    {chat_history}

    New human question: {input}
    {agent_scratchpad}
    """
        
    # Create the prompt with the required variables
    prompt = ChatPromptTemplate.from_template(template)
    
    # Create the agent with the proper prompt and custom output parser for better error handling
    custom_parser = GracefulReActOutputParser()
    
    agent = create_react_agent(
        llm=llm, 
        tools=tools, 
        prompt=prompt.partial(
            system_prompt=SYSTEM_PROMPT,
            tools="\n\n".join([f"{tool.name}: {tool.description}" for tool in tools]),
            tool_names=tool_names
        ),
        output_parser=custom_parser
    )
    
    # Create and return the agent executor with enhanced error handling
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        handle_parsing_errors=True,
        max_iterations=10,  # Prevent infinite loops
        max_execution_time=60,  # Timeout after 60 seconds
        early_stopping_method="force"  # Force stop on errors rather than trying to recover
    )

def run_agent(agent_executor, user_input, chat_history=None):
    """
    Run the agent with the given input.
    
    Args:
        agent_executor (AgentExecutor): The agent executor to run
        user_input (str): The user input
        chat_history (list, optional): The chat history
        
    Returns:
        dict: The agent's response
    """
    if chat_history is None:
        chat_history = []
    
    # Run the agent
    return agent_executor.invoke({
        "input": user_input,
        "chat_history": chat_history
    })
