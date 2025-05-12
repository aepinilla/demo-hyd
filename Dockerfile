# Use the official lightweight Python image.
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install uv (fast Python package manager)
RUN pip install --upgrade pip && pip install uv

# Create requirements.txt
RUN uv pip freeze > requirements.txt

# Copy only requirements.txt first for better layer caching
COPY requirements.txt .

# Install dependencies using uv with --system flag
RUN uv pip install --system -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port Streamlit will run on
EXPOSE 8080

# Streamlit-specific config to allow Cloud Run to bind to $PORT
ENV PORT 8080

# Run Streamlit
CMD streamlit run app.py --server.port $PORT --server.address 0.0.0.0