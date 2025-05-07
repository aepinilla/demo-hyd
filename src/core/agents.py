"""
Agent implementations for the chatbot application.

This module handles the creation and execution of LangChain agents.
"""

import streamlit as st
from typing import List, Dict, Any, Optional
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

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
    # Use the API key from environment variables
    llm = ChatOpenAI(
        model=GPT_MODEL, 
        temperature=0, 
        streaming=True
    )
    
    # Get tool names for the prompt
    tool_names = [tool.name for tool in tools]
    
    # Create a React-compatible prompt template
    template = """
    {system_prompt}

    You have access to the following tools:

    {tools}

    To use a tool, please use the following format:
    ```
    Thought: I need to use a tool to help answer the user's question.
    Action: tool_name
    Action Input: the input to the tool
    ```

    When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:
    ```
    Thought: I am ready to answer the question
    Final Answer: the final answer to the human's question
    ```

    Begin!

    Previous conversation history:
    {chat_history}

    New human question: {input}
    {agent_scratchpad}
    """
        
    # Create the prompt with the required variables
    prompt = ChatPromptTemplate.from_template(template)
    
    # Create the agent with the proper prompt
    agent = create_react_agent(
        llm=llm, 
        tools=tools, 
        prompt=prompt.partial(
            system_prompt=SYSTEM_PROMPT,
            tools="\n\n".join([f"{tool.name}: {tool.description}" for tool in tools]),
            tool_names=tool_names
        )
    )
    
    # Create and return the agent executor
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        handle_parsing_errors=True
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
