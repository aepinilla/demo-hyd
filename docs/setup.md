# Setup Instructions

This document provides detailed setup instructions for the chatbot application.

## Prerequisites

- Python 3.8 or higher
- pip or uv package manager

## Installation Steps

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd demo
   ```

2. Install dependencies:
   ```bash
   uv sync
   ```

3. Set up environment variables:
   - Copy the `.env.example` file to `.env`
   - Edit the `.env` file to add your OpenAI API key
   ```bash
   cp env.example .env
   # Edit .env with your preferred editor
   ```

## Running the Application

To run the main application:

```bash
streamlit run app_new.py
```

## Running the Examples

The project includes example scripts that demonstrate specific features:

- Basic chat example:
  ```bash
  streamlit run examples/basic_chat.py
  ```

- Advanced features example:
  ```bash
  streamlit run examples/advanced_features.py
  ```

## Development Environment

For development, it's recommended to use a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```
