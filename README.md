# Data Visualization Assistant

A Streamlit-based data visualization assistant designed for teaching data analysis and visualization concepts. This educational tool demonstrates how to create interactive data visualizations using Streamlit, Seaborn, and LangChain.

## Project Structure

```
demo/
├── app.py                  # Main application entry point
├── examples/               # Example scripts & data
│   ├── sample_data.csv     # Sample dataset for demonstration
├── docs/                   # Documentation
│   ├── setup.md            # Setup instructions
│   ├── usage.md            # Usage guide
│   └── troubleshooting.md  # Common issues and solutions
└── src/                    # Source code
    ├── config/             # Configuration settings
    ├── core/               # LangChain agent integration
    │   ├── agents.py       # Agent creation and execution
    │   └── tools.py        # Visualization tools definition
    ├── ui/                 # UI components
    ├── utils/              # Utility functions
    │   └── visualization.py # Data visualization functions
    └── schemas/            # Data models
```

## Quick Start

1. Install the required packages:

   ```bash
   uv sync
   ```

2. Set up your environment variables:

   ```bash
   cp env.example .env
   # Edit .env with your OpenAI API key
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Features

- **Interactive UI**: Clean, user-friendly interface with Streamlit
- **Data Visualization**: Powerful visualizations using Seaborn
- **AI Assistance**: LangChain agent that helps interpret data and suggest visualizations
- **Educational Design**: Well-structured code for teaching data visualization concepts
- **Modular Architecture**: Easy to extend with new visualization types

## Visualization Capabilities

- **Histograms**: Visualize distribution of numerical data
- **Scatter Plots**: Explore relationships between two variables
- **Line Plots**: Analyze trends over time or sequences
- **Correlation Heatmaps**: Understand relationships between multiple variables

## Example Prompts

- "Load the sample dataset"
- "Create a histogram of age"
- "Show a scatter plot of income vs. education"
- "Generate a correlation heatmap for all numeric columns"
- "Plot a line chart showing trends over time"

## Documentation

For more detailed information, see the documentation in the `docs/` directory:

- [Setup Instructions](docs/setup.md)
- [Usage Guide](docs/usage.md)
- [Troubleshooting](docs/troubleshooting.md)

## For Teaching

This project is designed for educational purposes, focusing on data visualization concepts:

- **Exploratory Data Analysis**: Learn how to examine datasets visually
- **Statistical Insights**: Interpret data through visualizations
- **Interactive Visualization**: Understand how to create responsive data visuals
- **Data Storytelling**: Practice communicating insights through visual elements
