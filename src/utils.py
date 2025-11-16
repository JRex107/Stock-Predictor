"""
Utility functions for the stock forecasting system
"""
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Any
import pickle
import json

import pandas as pd


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with consistent formatting

    Args:
        name: Logger name
        level: Logging level

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def is_cache_valid(cache_path: Path, hours: int = 24) -> bool:
    """
    Check if a cache file exists and is recent enough

    Args:
        cache_path: Path to cache file
        hours: Maximum age in hours

    Returns:
        True if cache is valid, False otherwise
    """
    if not cache_path.exists():
        return False

    file_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
    age = datetime.now() - file_time

    return age < timedelta(hours=hours)


def save_cache(data: Any, cache_path: Path, format: str = "pickle") -> None:
    """
    Save data to cache file

    Args:
        data: Data to save
        cache_path: Path to cache file
        format: Format to use ('pickle', 'json', 'csv')
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "pickle":
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
    elif format == "json":
        with open(cache_path, 'w') as f:
            json.dump(data, f, indent=2)
    elif format == "csv" and isinstance(data, pd.DataFrame):
        data.to_csv(cache_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_cache(cache_path: Path, format: str = "pickle") -> Optional[Any]:
    """
    Load data from cache file

    Args:
        cache_path: Path to cache file
        format: Format to use ('pickle', 'json', 'csv')

    Returns:
        Cached data or None if not found
    """
    if not cache_path.exists():
        return None

    try:
        if format == "pickle":
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        elif format == "json":
            with open(cache_path, 'r') as f:
                return json.load(f)
        elif format == "csv":
            return pd.read_csv(cache_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    except Exception as e:
        logging.error(f"Error loading cache from {cache_path}: {e}")
        return None


def retry_with_backoff(func, max_retries: int = 3, initial_delay: float = 1.0):
    """
    Retry a function with exponential backoff

    Args:
        func: Function to retry
        max_retries: Maximum number of retries
        initial_delay: Initial delay in seconds

    Returns:
        Function result

    Raises:
        Last exception if all retries fail
    """
    delay = initial_delay
    last_exception = None

    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
            else:
                raise last_exception


def safe_division(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if division fails

    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division fails

    Returns:
        Result or default
    """
    try:
        if denominator == 0 or pd.isna(denominator):
            return default
        result = numerator / denominator
        return result if not pd.isna(result) else default
    except:
        return default


def normalize_ticker(ticker: str) -> str:
    """
    Normalize ticker symbol (remove special characters, uppercase)

    Args:
        ticker: Raw ticker symbol

    Returns:
        Normalized ticker
    """
    # Remove common suffixes for UK stocks
    ticker = ticker.upper().strip()
    ticker = ticker.replace('.L', '')  # London Stock Exchange
    return ticker


def format_large_number(num: float, precision: int = 2) -> str:
    """
    Format large numbers with K/M/B suffixes

    Args:
        num: Number to format
        precision: Decimal precision

    Returns:
        Formatted string
    """
    if pd.isna(num):
        return "N/A"

    abs_num = abs(num)
    sign = "-" if num < 0 else ""

    if abs_num >= 1e9:
        return f"{sign}{abs_num/1e9:.{precision}f}B"
    elif abs_num >= 1e6:
        return f"{sign}{abs_num/1e6:.{precision}f}M"
    elif abs_num >= 1e3:
        return f"{sign}{abs_num/1e3:.{precision}f}K"
    else:
        return f"{sign}{abs_num:.{precision}f}"


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.04) -> float:
    """
    Calculate annualized Sharpe ratio

    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate

    Returns:
        Sharpe ratio
    """
    if len(returns) == 0 or returns.std() == 0:
        return 0.0

    excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
    return (excess_returns.mean() / returns.std()) * (252 ** 0.5)


def calculate_max_drawdown(returns: pd.Series) -> float:
    """
    Calculate maximum drawdown from returns series

    Args:
        returns: Series of returns

    Returns:
        Maximum drawdown (negative value)
    """
    if len(returns) == 0:
        return 0.0

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max

    return drawdown.min()
