# Streamlit Chatbot Demo

A simple, modular Streamlit chatbot application designed for teaching purposes. This project demonstrates how to build a chatbot using Streamlit and OpenAI's API with a clean, well-organized structure.

## Project Structure

```
demo/
├── app_new.py              # Main application entry point
├── examples/               # Example scripts
│   ├── basic_chat.py       # Simple chat example
│   └── advanced_features.py # More complex features
├── docs/                   # Documentation
│   ├── setup.md            # Setup instructions
│   ├── usage.md            # Usage guide
│   └── troubleshooting.md  # Common issues and solutions
└── src/                    # Source code
    ├── config/             # Configuration
    ├── core/               # Core functionality
    ├── ui/                 # UI components
    ├── utils/              # Utility functions
    └── schemas/            # Data models
```

## Quick Start

1. Install the packages:
   ```bash
   uv sync
   ```

2. Edit the env.example file with your credentials:
   ```bash
   cp env.example .env
   # Edit .env with your OpenAI API key
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app_new.py
   ```

## Features

- Simple, clean UI with Streamlit
- OpenAI integration for natural language processing
- Modular, well-documented code structure
- Example scripts demonstrating different features
- Comprehensive documentation

## Example Prompts

- "Hello, how can you help me?"
- "Tell me about this application"
- "What can I use this chatbot for?"

## Documentation

For more detailed information, see the documentation in the `docs/` directory:

- [Setup Instructions](docs/setup.md)
- [Usage Guide](docs/usage.md)
- [Troubleshooting](docs/troubleshooting.md)

## For Teaching

This project is designed to be used for teaching purposes. The code is:

- **Modular**: Each component has a single responsibility
- **Well-documented**: Comprehensive comments and documentation
- **Easy to extend**: Clear separation of concerns
- **Progressive**: From basic to advanced examples

## License

MIT
