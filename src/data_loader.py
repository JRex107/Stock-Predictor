"""
Data loader module for fetching stock market data using yfinance
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import yfinance as yf
from tqdm import tqdm

from .config import (
    RAW_DATA_DIR,
    CACHE_DURATION_HOURS,
    MAX_WORKERS,
    INDICES
)
from .utils import (
    setup_logger,
    is_cache_valid,
    save_cache,
    load_cache,
    retry_with_backoff
)


logger = setup_logger(__name__)


class DataLoader:
    """
    Handles fetching and caching of stock market data
    """

    def __init__(self, cache_duration_hours: int = CACHE_DURATION_HOURS):
        """
        Initialize DataLoader

        Args:
            cache_duration_hours: Hours to cache data
        """
        self.cache_duration = cache_duration_hours
        self.cache_dir = RAW_DATA_DIR
        logger.info(f"DataLoader initialized with {cache_duration_hours}h cache")

    def get_stock_data(
        self,
        ticker: str,
        period: str = "2y",
        interval: str = "1d",
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Get historical stock data for a ticker

        Args:
            ticker: Stock ticker symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            use_cache: Whether to use cached data

        Returns:
            DataFrame with OHLCV data or None if failed
        """
        cache_path = self.cache_dir / f"{ticker}_{period}_{interval}.csv"

        # Check cache
        if use_cache and is_cache_valid(cache_path, self.cache_duration):
            logger.debug(f"Loading {ticker} from cache")
            data = load_cache(cache_path, format="csv")
            if data is not None and not data.empty:
                data.index = pd.to_datetime(data.index)
                # Ensure column names are capitalized (yfinance standard format)
                # Handle both lowercase and capitalized versions
                column_mapping = {}
                for col in data.columns:
                    col_lower = col.lower()
                    if col_lower in ['open', 'high', 'low', 'close', 'volume', 'dividends', 'stock splits']:
                        if col_lower == 'stock splits':
                            column_mapping[col] = 'Stock Splits'
                        else:
                            column_mapping[col] = col_lower.capitalize()
                if column_mapping:
                    data = data.rename(columns=column_mapping)
                return data

        # Fetch fresh data
        try:
            logger.debug(f"Fetching {ticker} from yfinance")

            def fetch():
                stock = yf.Ticker(ticker)
                return stock.history(period=period, interval=interval)

            data = retry_with_backoff(fetch, max_retries=3)

            if data is not None and not data.empty:
                # Remove timezone info to avoid serialization issues
                if isinstance(data.index, pd.DatetimeIndex) and data.index.tz is not None:
                    data.index = data.index.tz_localize(None)

                # Save to cache
                save_cache(data, cache_path, format="csv")
                return data
            else:
                logger.warning(f"No data returned for {ticker}")
                return None

        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            return None

    def get_stock_info(
        self,
        ticker: str,
        use_cache: bool = True
    ) -> Optional[Dict]:
        """
        Get fundamental information about a stock

        Args:
            ticker: Stock ticker symbol
            use_cache: Whether to use cached data

        Returns:
            Dictionary with stock info or None if failed
        """
        cache_path = self.cache_dir / f"{ticker}_info.pickle"

        # Check cache
        if use_cache and is_cache_valid(cache_path, self.cache_duration):
            logger.debug(f"Loading {ticker} info from cache")
            return load_cache(cache_path, format="pickle")

        # Fetch fresh data
        try:
            logger.debug(f"Fetching {ticker} info from yfinance")

            def fetch():
                stock = yf.Ticker(ticker)
                return stock.info

            info = retry_with_backoff(fetch, max_retries=3)

            if info:
                # Save to cache
                save_cache(info, cache_path, format="pickle")
                return info
            else:
                logger.warning(f"No info returned for {ticker}")
                return None

        except Exception as e:
            logger.error(f"Error fetching {ticker} info: {e}")
            return None

    def get_multiple_stocks(
        self,
        tickers: List[str],
        period: str = "2y",
        interval: str = "1d",
        use_cache: bool = True,
        show_progress: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Get historical data for multiple stocks in parallel

        Args:
            tickers: List of ticker symbols
            period: Data period
            interval: Data interval
            use_cache: Whether to use cached data
            show_progress: Whether to show progress bar

        Returns:
            Dictionary mapping ticker to DataFrame
        """
        results = {}

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all tasks
            future_to_ticker = {
                executor.submit(
                    self.get_stock_data,
                    ticker,
                    period,
                    interval,
                    use_cache
                ): ticker
                for ticker in tickers
            }

            # Collect results with progress bar
            iterator = as_completed(future_to_ticker)
            if show_progress:
                iterator = tqdm(iterator, total=len(tickers), desc="Fetching stock data")

            for future in iterator:
                ticker = future_to_ticker[future]
                try:
                    data = future.result()
                    if data is not None and not data.empty:
                        results[ticker] = data
                except Exception as e:
                    logger.error(f"Error processing {ticker}: {e}")

        logger.info(f"Successfully fetched {len(results)}/{len(tickers)} stocks")
        return results

    def get_index_data(
        self,
        index_key: str,
        period: str = "2y",
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Get historical data for an index

        Args:
            index_key: Index key from config (e.g., 'SP500', 'FTSE100')
            period: Data period
            use_cache: Whether to use cached data

        Returns:
            DataFrame with index data or None if failed
        """
        if index_key not in INDICES:
            logger.error(f"Unknown index: {index_key}")
            return None

        index_symbol = INDICES[index_key]["symbol"]
        return self.get_stock_data(index_symbol, period=period, use_cache=use_cache)

    def get_current_price(self, ticker: str) -> Optional[float]:
        """
        Get current/latest price for a ticker

        Args:
            ticker: Stock ticker symbol

        Returns:
            Current price or None if failed
        """
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period="1d")
            if not data.empty:
                return data['Close'].iloc[-1]
            return None
        except Exception as e:
            logger.error(f"Error fetching current price for {ticker}: {e}")
            return None

    def get_multiple_info(
        self,
        tickers: List[str],
        use_cache: bool = True,
        show_progress: bool = True
    ) -> Dict[str, Dict]:
        """
        Get fundamental info for multiple stocks in parallel

        Args:
            tickers: List of ticker symbols
            use_cache: Whether to use cached data
            show_progress: Whether to show progress bar

        Returns:
            Dictionary mapping ticker to info dict
        """
        results = {}

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_ticker = {
                executor.submit(
                    self.get_stock_info,
                    ticker,
                    use_cache
                ): ticker
                for ticker in tickers
            }

            iterator = as_completed(future_to_ticker)
            if show_progress:
                iterator = tqdm(iterator, total=len(tickers), desc="Fetching stock info")

            for future in iterator:
                ticker = future_to_ticker[future]
                try:
                    info = future.result()
                    if info:
                        results[ticker] = info
                except Exception as e:
                    logger.error(f"Error processing {ticker} info: {e}")

        logger.info(f"Successfully fetched info for {len(results)}/{len(tickers)} stocks")
        return results

    def validate_ticker(self, ticker: str) -> bool:
        """
        Check if a ticker is valid by attempting to fetch data

        Args:
            ticker: Stock ticker symbol

        Returns:
            True if valid, False otherwise
        """
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period="5d")
            return not data.empty
        except:
            return False


# Convenience function
def create_data_loader(cache_duration_hours: int = CACHE_DURATION_HOURS) -> DataLoader:
    """
    Create and return a DataLoader instance

    Args:
        cache_duration_hours: Hours to cache data

    Returns:
        DataLoader instance
    """
    return DataLoader(cache_duration_hours=cache_duration_hours)
