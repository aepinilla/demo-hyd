# Data Visualization Assistant

A Streamlit-based data visualization assistant designed for teaching data analysis and visualization concepts. This educational tool demonstrates how to create interactive data visualizations using Streamlit, Seaborn, and LangChain.

## Project Structure

```
.
├── .streamlit/            # Streamlit configuration
│   ├── config.toml        # Streamlit config settings
│   └── custom.css         # Custom CSS for the Streamlit app
├── app.py                 # Main application entry point
├── cloudbuild.yaml        # Cloud Build configuration
├── Dockerfile             # Docker configuration for containerization
├── env.example            # Example environment variables
├── launch_config.json     # Launch configuration
├── pyproject.toml         # Project dependencies and metadata
├── README.md              # Project documentation
├── requirements.txt       # Generated Python dependencies
├── src/                   # Source code
│   ├── config/            # Configuration settings
│   │   └── settings.py    # Application settings
│   ├── core/              # LangChain agent integration
│   │   ├── agents.py      # Agent creation and execution
│   │   ├── chat.py        # Chat functionality
│   │   ├── custom_parser.py # Custom parsing logic
│   │   ├── fallback_handler.py # Fallback logic for errors
│   │   ├── llm.py         # Language model integration
│   │   └── tools/         # Visualization tools
│   │       ├── sensor/    # Sensor-specific tools
│   │       │   ├── get_sensor_variable_stats.py
│   │       │   ├── plot_sensor_heatmap.py
│   │       │   ├── plot_sensor_histogram.py
│   │       │   ├── plot_sensor_scatter.py
│   │       │   ├── prepare_sensor_data_for_visualization.py
│   │       │   ├── sensor_tools.py
│   │       │   └── sync_sensor_data.py
│   │       ├── standard/  # Standard visualization tools
│   │       │   └── standard_tools.py
│   │       └── tools.py   # Tools registration and configuration
│   ├── schemas/           # Data models
│   │   └── models.py      # Data schemas and models
│   ├── ui/                # UI components
│   │   ├── components.py  # Streamlit UI components
│   │   └── logos/         # Logo assets
│   └── utils/             # Utility functions
│       ├── auto_load.py   # Automatic data loading
│       ├── data_bridge.py # Data transfer utilities
│       ├── dataframe_utils.py # DataFrame manipulation
│       ├── debug_utils.py # Debugging utilities
│       ├── load_processed_data.py # Data loading utilities
│       ├── remove_outliers.py # Data cleaning utilities
│       ├── sensor_api.py  # Sensor API integration
│       ├── time_series.py # Time series utilities
│       └── visualization.py # Data visualization functions
└── uv.lock                # Package lock file
```

## Quick Start

For detailed installation instructions on both macOS and Windows, see [INSTALLATION.md](INSTALLATION.md).

Quick setup:

1. Install the required packages:

   ```bash
   uv sync
   ```

2. Set up your environment variables:

   ```bash
   cp env.example .env
   # Edit .env with your OpenAI API key
   ```

3. Activate the virtual environment:
   ```bash
   # On macOS/Linux
   source .venv/bin/activate
   # On Windows
   .venv\Scripts\activate
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Features

- **Interactive UI**: Clean, user-friendly interface with Streamlit
- **Data Visualization**: Beautiful visualizations with Seaborn
- **AI Assistance**: LangChain agent that helps interpret data and suggest visualizations
- **Educational Design**: Minimal code for teaching AI agent development and data visualization concepts
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

This project is designed for educational purposes, focusing on AI agent development and data visualization concepts:

- **Exploratory Data Analysis**: Learn how to examine datasets visually
- **Statistical Insights**: Interpret data through visualizations
- **Interactive Visualization**: Understand how to create responsive data visuals
- **Data Storytelling**: Practice communicating insights through visual elements

## Docker

The Dockerfile is configured to automatically generate the requirements.txt file during the build process using the dependencies specified in pyproject.toml. This ensures that the Docker container always has the most up-to-date dependencies without requiring manual updates to requirements.txt.

To build the Docker image, run:

```bash
docker build -t hyd2025-demo .
```

To run the Docker container, run:

```bash
docker run -it --rm -p 8080:8080 hyd2025-demo
```

This will start the Streamlit app in a Docker container, and you can access it at `http://localhost:8080`.

## Data Cleaning Tools

### Outlier Removal

The application includes a tool for removing outliers from sensor data using the Interquartile Range (IQR) method. This is particularly useful for cleaning SDS011 sensor data which often contains extreme values in P1 (PM10) and P2 (PM2.5) measurements.

Example usage in the chatbot:

```
# First load the sensor data
fetch_latest_sensor_data: {"data_type": "SDS011"}
load_dataset: "sensor_data"

# Then remove outliers
remove_outliers: {"columns": "P1,P2", "iqr_multiplier": 1.5, "drop_outliers": true}

# Now visualize the cleaned data
plot_histogram: {"column": "P1", "bins": 30}
```

Parameters:

- `columns`: List of columns or 'all' for all numeric columns
- `iqr_multiplier`: Multiplier for IQR to determine outlier threshold (default: 1.5)
- `drop_outliers`: Whether to drop outliers (true) or replace them (false)
- `fill_method`: Method to fill outliers if drop_outliers is false. Options: 'mean', 'median', 'mode', 'nearest', null

### Loading Processed Data

After removing outliers, the cleaned data is automatically saved to disk in the `data/processed` directory. The application will automatically load the most recent processed dataset when it starts.

You can also manually load processed data using the `load_processed_data` tool:

Example usage in the chatbot:

```
# Load the most recent processed dataset
load_processed_data: {"latest": true}

# Or load a specific file by name
load_processed_data: {"filename": "cleaned_data_20250718_144500.csv"}

# List all available processed files
load_processed_data: {"latest": false}
```

Parameters:

- `filename`: Optional specific filename to load
- `latest`: If true and no filename provided, loads the most recent file. If false, lists all available files.

## Docker Container Management

```bash
# List all running containers
docker ps

# Stop a running container
docker stop hyd2025-demo

# Remove a stopped container
docker rm hyd2025-demo
```
