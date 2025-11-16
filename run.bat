@echo off

REM Stock Market Forecasting System - Launcher Script (Windows)

echo ================================
echo Stock Market Forecasting System
echo ================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo Virtual environment not found. Creating one...
    python -m venv venv
    echo [OK] Virtual environment created
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Check for .env file
if not exist ".env" (
    echo.
    echo WARNING: .env file not found!
    echo Creating .env from template...
    copy .env.example .env
    echo.
    echo Please edit .env and add your NewsAPI key for sentiment analysis.
    echo Get a free key at: https://newsapi.org/register
    echo.
)

REM Install dependencies if needed
pip show streamlit >nul 2>&1
if errorlevel 1 (
    echo Dependencies not installed. Installing...
    pip install -r requirements.txt
    echo [OK] Dependencies installed
)

REM Launch dashboard
echo.
echo Launching dashboard...
echo The dashboard will open in your browser at http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.

streamlit run dashboard\app.py

pause
