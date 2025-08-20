# Installation Guide for Data Visualization Assistant

Below are the step-by-step instructions for installing and running the project on both macOS and Windows.

## macOS Installation

1. **Install Homebrew (if not already installed)**
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Install Python 3.11**
   ```bash
   brew install python@3.11
   ```

3. **Install uv package manager**
   ```bash
   brew install astral-sh/tap/uv
   ```

4. **Clone the repository (if you haven't already)**
   ```bash
   git clone git@github.com:aepinilla/demo-hyd.git && cd demo-hyd
   ```

5. **Set up environment variables**
   ```bash
   cp env.example .env
   ```

6. **Edit .env with your OpenAI API key**
   ```bash
   # Edit .env file and add your API key
   OPENAI_API_KEY="your-api-key-here"
   ```

7. **Install project dependencies using uv**
   ```bash
   uv sync
   ```

8. **Activate the virtual environment**
   ```bash
   source .venv/bin/activate
   ```

9. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

## Windows Installation

1. **Install Chocolatey (if not already installed)**
   ```powershell
   # Run in an Administrator PowerShell
   Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
   ```

2. **Install Python 3.11**
   ```powershell
   choco install python311 -y
   ```

3. **Install uv package manager**
   ```powershell
   pip install uv
   ```

4. **Clone the repository (if you haven't already)**
   ```powershell
   git clone git@github.com:aepinilla/demo-hyd.git && cd demo-hyd
   ```

5. **Set up environment variables**
   ```powershell
   copy env.example .env
   ```

6. **Edit .env with your OpenAI API key**
   ```powershell
   # Edit .env file and add your API key
   OPENAI_API_KEY="your-api-key-here"
   ```

7. **Install project dependencies using uv**
   ```powershell
   uv sync
   ```

8. **Activate the virtual environment**
   ```powershell
   .venv\Scripts\Activate.ps1
   ```

9. **Run the Streamlit app**
   ```powershell
   streamlit run app.py
   ```

## Troubleshooting

- **Python not found**: Restart your terminal or run `refreshenv` in PowerShell on Windows
- **OpenAI API errors**: Verify your API key is correctly set in the .env file

The application will be accessible at `http://localhost:8501` in your browser.
