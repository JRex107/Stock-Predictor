#!/bin/bash

# Stock Market Forecasting System - Launcher Script

echo "================================"
echo "Stock Market Forecasting System"
echo "================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Creating one..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check if dependencies are installed
if ! python -c "import streamlit" &> /dev/null; then
    echo "Dependencies not installed. Installing..."
    pip install -r requirements.txt
    echo "✓ Dependencies installed"
fi

# Check for .env file
if [ ! -f ".env" ]; then
    echo ""
    echo "⚠️  WARNING: .env file not found!"
    echo "Creating .env from template..."
    cp .env.example .env
    echo ""
    echo "Please edit .env and add your NewsAPI key for sentiment analysis."
    echo "Get a free key at: https://newsapi.org/register"
    echo ""
fi

# Launch dashboard
echo ""
echo "Launching dashboard..."
echo "The dashboard will open in your browser at http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

streamlit run dashboard/app.py
