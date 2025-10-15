# NETWORK Frequency Analysis System - Docker Deployment

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data and visualizations directories
RUN mkdir -p data visualizations logs

# Set environment variables
ENV PYTHONPATH=/app
ENV NETWORK_DATA_DIR=/app/data
ENV NETWORK_LOG_LEVEL=INFO

# Expose any ports if needed (GUI app)
EXPOSE 8000

# Default command - run main analysis
CMD ["python3", "main.py"]

# Alternative commands for different deployment modes:
# CMD ["python3", "live_console_monitor.py"]        # Live monitoring
# CMD ["python3", "background_monitor.py", "--instance", "1"]  # Background monitoring
# CMD ["python3", "enhanced_monitor.py"]            # Enhanced monitoring
# CMD ["python3", "gui_app.py"]                      # GUI application
