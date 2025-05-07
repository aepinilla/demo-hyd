# Usage Guide

This document provides instructions on how to use the chatbot application.

## Basic Usage

The chatbot application provides a simple interface for interacting with an AI assistant. Here's how to use it:

1. Start the application:
   ```bash
   streamlit run app_new.py
   ```

2. Type your message in the chat input field at the bottom of the page and press Enter.

3. The assistant will respond to your message.

4. Continue the conversation by sending additional messages.

## Example Prompts

Here are some example prompts you can use to interact with the chatbot:

- "Hello, how are you today?"
- "Can you explain what this application does?"
- "Tell me a joke."
- "What's the weather like today?" (Note: The assistant doesn't have real-time data access)

## Advanced Features

The advanced features example (`examples/advanced_features.py`) includes additional functionality:

### File Upload

1. Click on the "Upload a file for context" button in the sidebar.
2. Select a text or CSV file from your computer.
3. The file will be uploaded and its contents will be used as context for the conversation.
4. Ask questions about the file content.

### Clearing Chat History

1. Click on the "Clear Chat History" button in the sidebar to start a new conversation.

## Project Structure

The project is organized into several modules:

- `src/config`: Configuration settings
- `src/core`: Core functionality (chat logic, LLM integration)
- `src/ui`: UI components for Streamlit
- `src/utils`: Utility functions
- `src/schemas`: Data models and schemas
- `examples`: Example scripts

## Extending the Application

To add new features to the application:

1. Identify the appropriate module for your feature.
2. Add your code to the relevant file or create a new file if needed.
3. Update the main application or create a new example to demonstrate your feature.
