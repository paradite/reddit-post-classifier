# Use Python 3.11.10 as specified in pyproject.toml
FROM python:3.11.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy API server requirements and install Python dependencies
COPY api-requirements.txt .

# Install dependencies using pip
RUN pip install --no-cache-dir -r api-requirements.txt

# Copy the model files and server code
COPY best_model_run12_epoch_9.pt .
COPY best_regressor_run1_epoch_4.pt .
COPY api-server.py .

# Expose the port the server runs on
EXPOSE 8080

# Command to run the server
CMD ["python", "api-server.py"] 