# Use the official lightweight Python image.
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Upgrade pip and install dependencies
RUN pip install --upgrade pip

# Copy only requirements.txt first for better layer caching
COPY requirements.txt .

# Install dependencies using pip
RUN pip install -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port Streamlit will run on
EXPOSE 8080

# Set the port for Streamlit
ENV PORT=8080

# Run Streamlit with JSON array format (recommended for production)
# Using hardcoded port value instead of environment variable
CMD ["streamlit", "run", "app.py", "--server.port", "8080", "--server.address", "0.0.0.0"]