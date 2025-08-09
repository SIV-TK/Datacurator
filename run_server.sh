#!/bin/bash

# Data Curator Web Server Startup Script
# Created: August 9, 2025

echo "Starting Data Curator Web Server..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Check if the required directories exist
if [ ! -d "src/web" ]; then
    echo "Error: src/web directory not found. Are you in the project root directory?"
    exit 1
fi

# Make sure the config directory exists and has the default.yaml file
if [ ! -f "config/default.yaml" ]; then
    echo "Warning: config/default.yaml not found. Using default configuration."
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Run the FastAPI application using uvicorn
echo "Launching web server..."
uvicorn src.web.app:app --host 0.0.0.0 --port 8000 --reload 2>&1 | tee logs/server_$(date +%Y%m%d).log

# Deactivate virtual environment on exit
if [ -n "$VIRTUAL_ENV" ]; then
    echo "Deactivating virtual environment..."
    deactivate
fi
