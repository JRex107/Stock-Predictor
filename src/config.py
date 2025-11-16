"""
Configuration settings for the Stock Forecasting System
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CONSTITUENTS_DIR = DATA_DIR / "constituents"
MODELS_DIR = PROJECT_ROOT / "models"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, CONSTITUENTS_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# API Keys
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "")

# Data settings
CACHE_DURATION_HOURS = int(os.getenv("CACHE_DURATION_HOURS", "24"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "5"))
TOP_N_PER_INDEX = int(os.getenv("TOP_N_PER_INDEX", "50"))
MIN_MARKET_CAP_MILLIONS = float(os.getenv("MIN_MARKET_CAP_MILLIONS", "100"))

# Index configurations
INDICES = {
    "SP500": {
        "name": "S&P 500",
        "symbol": "^GSPC",
        "constituents_url": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    },
    "NASDAQ100": {
        "name": "NASDAQ-100",
        "symbol": "^NDX",
        "constituents_url": "https://en.wikipedia.org/wiki/Nasdaq-100"
    },
    "FTSE100": {
        "name": "FTSE 100",
        "symbol": "^FTSE",
        "constituents_url": "https://en.wikipedia.org/wiki/FTSE_100_Index"
    },
    "FTSE250": {
        "name": "FTSE 250",
        "symbol": "^FTMC",
        "constituents_url": "https://en.wikipedia.org/wiki/FTSE_250_Index"
    }
}

# Feature engineering parameters
FEATURE_PARAMS = {
    "returns_periods": [1, 5, 21, 63, 126, 252],  # days (1D, 1W, 1M, 3M, 6M, 1Y)
    "volatility_window": 21,  # 1 month
    "ma_windows": [20, 50, 200],  # Moving average windows
    "volume_window": 20,  # Volume average window
    "rsi_period": 14,  # RSI period
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9
}

# Scoring model weights (configurable)
SCORING_WEIGHTS = {
    "conservative": {
        "momentum": 0.15,
        "volatility": 0.25,  # Lower volatility preferred
        "trend": 0.20,
        "value": 0.25,
        "sentiment": 0.15
    },
    "balanced": {
        "momentum": 0.25,
        "volatility": 0.15,
        "trend": 0.25,
        "value": 0.20,
        "sentiment": 0.15
    },
    "aggressive": {
        "momentum": 0.35,
        "volatility": 0.10,
        "trend": 0.30,
        "value": 0.10,
        "sentiment": 0.15
    }
}

# Recommendation thresholds
RECOMMENDATION_THRESHOLDS = {
    "conservative": {
        "strong_buy": 0.75,
        "buy": 0.60,
        "hold": 0.40,
        "sell": 0.25
    },
    "balanced": {
        "strong_buy": 0.70,
        "buy": 0.55,
        "hold": 0.45,
        "sell": 0.30
    },
    "aggressive": {
        "strong_buy": 0.65,
        "buy": 0.50,
        "hold": 0.50,
        "sell": 0.35
    }
}

# Portfolio constraints
PORTFOLIO_CONSTRAINTS = {
    "max_positions": 20,
    "max_position_size": 0.10,  # 10% max per position
    "min_position_size": 0.03,  # 3% min per position
    "max_sector_concentration": 0.30,  # 30% max per sector
    "rebalance_threshold": 0.05  # 5% drift before rebalancing
}

# Backtesting parameters
BACKTEST_PARAMS = {
    "lookback_years": 2,
    "rebalance_frequency": "M",  # M=Monthly, W=Weekly, Q=Quarterly
    "portfolio_size": 20,
    "initial_capital": 100000,
    "commission": 0.001  # 0.1% per trade
}

# News and sentiment settings
NEWS_SETTINGS = {
    "lookback_days": 7,
    "max_articles_per_stock": 10,
    "sentiment_cache_hours": 24,
    "min_relevance_score": 0.5
}

# Risk-free rate (for Sharpe ratio calculation)
RISK_FREE_RATE = 0.04  # 4% annual

# Disclaimer text
DISCLAIMER = """
⚠️ **IMPORTANT DISCLAIMER** ⚠️

This tool is for **research and educational purposes only** and does **NOT constitute financial advice**.

- Past performance is not indicative of future results
- All investments carry risk, including potential loss of principal
- The recommendations provided are based on automated analysis and may not account for your personal financial situation
- Always conduct your own research and consult with a qualified financial advisor before making investment decisions
- The creators of this tool assume no liability for any financial losses incurred from using this system

By using this tool, you acknowledge that you understand these risks.
"""
