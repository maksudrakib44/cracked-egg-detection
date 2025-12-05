#!/bin/bash
# Setup script for Streamlit Cloud

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Creating necessary directories..."
mkdir -p runs_video

echo "Setup complete!"