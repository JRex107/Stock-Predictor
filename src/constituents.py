"""
Module for fetching and managing index constituents
"""
import logging
from typing import List, Dict, Optional
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .config import (
    CONSTITUENTS_DIR,
    INDICES,
    TOP_N_PER_INDEX,
    MIN_MARKET_CAP_MILLIONS
)
from .utils import (
    setup_logger,
    is_cache_valid,
    save_cache,
    load_cache
)
from .data_loader import DataLoader


logger = setup_logger(__name__)


class ConstituentsManager:
    """
    Manages fetching and caching of index constituents
    """

    def __init__(self, cache_duration_hours: int = 24 * 7):  # Cache for 1 week
        """
        Initialize ConstituentsManager

        Args:
            cache_duration_hours: Hours to cache constituent lists
        """
        self.cache_duration = cache_duration_hours
        self.cache_dir = CONSTITUENTS_DIR
        self.data_loader = DataLoader()

        # User-Agent header to avoid Wikipedia 403 errors
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        logger.info("ConstituentsManager initialized")

    def get_sp500_constituents(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch S&P 500 constituents from Wikipedia

        Args:
            use_cache: Whether to use cached data

        Returns:
            DataFrame with columns: Symbol, Security, Sector, Sub-Industry, etc.
        """
        cache_path = self.cache_dir / "sp500_constituents.csv"

        if use_cache and is_cache_valid(cache_path, self.cache_duration):
            logger.info("Loading S&P 500 constituents from cache")
            return load_cache(cache_path, format="csv")

        try:
            logger.info("Fetching S&P 500 constituents from Wikipedia")
            url = INDICES["SP500"]["constituents_url"]

            # Fetch with proper headers to avoid 403 errors
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()

            # Read the first table from Wikipedia
            tables = pd.read_html(response.text)
            df = tables[0]

            # Handle multi-level columns if present
            if isinstance(df.columns, pd.MultiIndex):
                # Try different levels to find the one with proper column names
                for level in range(df.columns.nlevels):
                    test_cols = df.columns.get_level_values(level)
                    # Check if this level has string column names (not just numbers)
                    if any(isinstance(col, str) and len(str(col)) > 1 for col in test_cols):
                        df.columns = test_cols
                        logger.info(f"Using MultiIndex level {level} for column names")
                        break
                else:
                    # If all levels are numeric or empty, flatten to last level
                    df.columns = df.columns.get_level_values(-1)

            # If columns are still numeric, try using first row as headers
            if all(isinstance(col, (int, float)) for col in df.columns):
                logger.warning("Columns are numeric, attempting to use first row as headers")
                df.columns = df.iloc[0]
                df = df[1:].reset_index(drop=True)

            logger.info(f"S&P 500 table columns: {list(df.columns)}")

            # Try to find the right columns flexibly
            ticker_col = None
            name_col = None
            sector_col = None
            industry_col = None

            for col in df.columns:
                col_lower = str(col).lower()
                if 'symbol' in col_lower or 'ticker' in col_lower:
                    ticker_col = col
                elif 'security' in col_lower or 'company' in col_lower or (col_lower == 'name' or 'name' in col_lower):
                    name_col = col
                elif 'sector' in col_lower and 'sub' not in col_lower:
                    sector_col = col
                elif ('sub' in col_lower and 'industry' in col_lower) or col_lower == 'industry':
                    industry_col = col

            if not ticker_col:
                raise ValueError(f"Could not find ticker column in {list(df.columns)}")

            # Build result dataframe with available columns
            result = pd.DataFrame()
            result['Ticker'] = df[ticker_col]
            result['Name'] = df[name_col] if name_col else ''
            result['Sector'] = df[sector_col] if sector_col else ''
            result['Industry'] = df[industry_col] if industry_col else ''
            result['Index'] = 'S&P 500'

            # Clean up ticker symbols (remove any whitespace)
            result['Ticker'] = result['Ticker'].astype(str).str.strip()

            save_cache(result, cache_path, format="csv")
            logger.info(f"Fetched {len(result)} S&P 500 constituents")

            return result

        except Exception as e:
            logger.error(f"Error fetching S&P 500 constituents: {e}")
            return pd.DataFrame()

    def get_nasdaq100_constituents(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch NASDAQ-100 constituents from Wikipedia

        Args:
            use_cache: Whether to use cached data

        Returns:
            DataFrame with constituent data
        """
        cache_path = self.cache_dir / "nasdaq100_constituents.csv"

        if use_cache and is_cache_valid(cache_path, self.cache_duration):
            logger.info("Loading NASDAQ-100 constituents from cache")
            return load_cache(cache_path, format="csv")

        try:
            logger.info("Fetching NASDAQ-100 constituents from Wikipedia")
            url = INDICES["NASDAQ100"]["constituents_url"]

            # Fetch with proper headers to avoid 403 errors
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()

            tables = pd.read_html(response.text)

            # Try different table indices to find the constituents table
            df = None
            for table_idx in [3, 4, 5]:
                if table_idx < len(tables):
                    temp_df = tables[table_idx]
                    # Handle multi-level columns if present
                    if isinstance(temp_df.columns, pd.MultiIndex):
                        # Try different levels to find the one with proper column names
                        for level in range(temp_df.columns.nlevels):
                            test_cols = temp_df.columns.get_level_values(level)
                            if any(isinstance(col, str) and len(str(col)) > 1 for col in test_cols):
                                temp_df.columns = test_cols
                                break
                        else:
                            temp_df.columns = temp_df.columns.get_level_values(-1)

                    # If columns are numeric, try using first row as headers
                    if all(isinstance(col, (int, float)) for col in temp_df.columns):
                        temp_df.columns = temp_df.iloc[0]
                        temp_df = temp_df[1:].reset_index(drop=True)

                    # Check if this looks like a constituents table (has ticker/company column)
                    col_str = ' '.join(str(col).lower() for col in temp_df.columns)
                    if ('ticker' in col_str or 'symbol' in col_str) and ('company' in col_str or 'name' in col_str):
                        df = temp_df
                        logger.info(f"Found NASDAQ-100 constituents in table {table_idx}")
                        break

            if df is None:
                raise ValueError("Could not find NASDAQ-100 constituents table")

            logger.info(f"NASDAQ-100 table columns: {list(df.columns)}")

            # Try to find the right columns flexibly
            ticker_col = None
            name_col = None
            sector_col = None
            industry_col = None

            for col in df.columns:
                col_lower = str(col).lower()
                if 'symbol' in col_lower or 'ticker' in col_lower:
                    ticker_col = col
                elif 'company' in col_lower or 'security' in col_lower or col_lower == 'name':
                    name_col = col
                elif 'sector' in col_lower and 'sub' not in col_lower:
                    sector_col = col
                elif ('sub' in col_lower and 'industry' in col_lower) or col_lower == 'industry':
                    industry_col = col

            if not ticker_col:
                raise ValueError(f"Could not find ticker column in {list(df.columns)}")

            # Build result dataframe with available columns
            result = pd.DataFrame()
            result['Ticker'] = df[ticker_col]
            result['Name'] = df[name_col] if name_col else ''
            result['Sector'] = df[sector_col] if sector_col else 'Technology'  # Default for NASDAQ
            result['Industry'] = df[industry_col] if industry_col else ''
            result['Index'] = 'NASDAQ-100'

            # Clean up ticker symbols
            result['Ticker'] = result['Ticker'].astype(str).str.strip()

            save_cache(result, cache_path, format="csv")
            logger.info(f"Fetched {len(result)} NASDAQ-100 constituents")

            return result

        except Exception as e:
            logger.error(f"Error fetching NASDAQ-100 constituents: {e}")
            return pd.DataFrame()

    def get_ftse100_constituents(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch FTSE 100 constituents from Wikipedia

        Args:
            use_cache: Whether to use cached data

        Returns:
            DataFrame with constituent data
        """
        cache_path = self.cache_dir / "ftse100_constituents.csv"

        if use_cache and is_cache_valid(cache_path, self.cache_duration):
            logger.info("Loading FTSE 100 constituents from cache")
            return load_cache(cache_path, format="csv")

        try:
            logger.info("Fetching FTSE 100 constituents from Wikipedia")
            url = INDICES["FTSE100"]["constituents_url"]

            # Fetch with proper headers to avoid 403 errors
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()

            tables = pd.read_html(response.text)

            # Try different table indices to find the constituents table
            df = None
            for table_idx in [2, 3, 4]:
                if table_idx < len(tables):
                    temp_df = tables[table_idx]
                    # Handle multi-level columns if present
                    if isinstance(temp_df.columns, pd.MultiIndex):
                        # Try different levels to find the one with proper column names
                        for level in range(temp_df.columns.nlevels):
                            test_cols = temp_df.columns.get_level_values(level)
                            if any(isinstance(col, str) and len(str(col)) > 1 for col in test_cols):
                                temp_df.columns = test_cols
                                break
                        else:
                            temp_df.columns = temp_df.columns.get_level_values(-1)

                    # If columns are numeric, try using first row as headers
                    if all(isinstance(col, (int, float)) for col in temp_df.columns):
                        temp_df.columns = temp_df.iloc[0]
                        temp_df = temp_df[1:].reset_index(drop=True)

                    # Check if this looks like a constituents table
                    if len(temp_df) > 50:  # FTSE 100 should have ~100 rows
                        col_str = ' '.join(str(col).lower() for col in temp_df.columns)
                        if ('ticker' in col_str or 'epic' in col_str) and ('company' in col_str or 'name' in col_str):
                            df = temp_df
                            logger.info(f"Found FTSE 100 constituents in table {table_idx}")
                            break

            if df is None:
                raise ValueError("Could not find FTSE 100 constituents table")

            logger.info(f"FTSE 100 table columns: {list(df.columns)}")

            # Try to find the right columns flexibly
            ticker_col = None
            name_col = None
            sector_col = None

            for col in df.columns:
                col_lower = str(col).lower()
                if 'ticker' in col_lower or 'epic' in col_lower or 'symbol' in col_lower:
                    ticker_col = col
                elif 'company' in col_lower or (col_lower == 'name' or 'name' in col_lower):
                    name_col = col
                elif 'sector' in col_lower:
                    sector_col = col

            if not ticker_col:
                raise ValueError(f"Could not find ticker column in {list(df.columns)}")

            # Build result dataframe with available columns
            result = pd.DataFrame()

            # Add .L suffix for London Stock Exchange tickers
            tickers = df[ticker_col].astype(str).str.strip()
            # Only add .L if not already present
            result['Ticker'] = tickers.apply(lambda x: x if x.endswith('.L') else f"{x}.L")

            result['Name'] = df[name_col] if name_col else ''
            result['Sector'] = df[sector_col] if sector_col else ''
            result['Industry'] = ''  # FTSE typically doesn't have industry level detail
            result['Index'] = 'FTSE 100'

            save_cache(result, cache_path, format="csv")
            logger.info(f"Fetched {len(result)} FTSE 100 constituents")

            return result

        except Exception as e:
            logger.error(f"Error fetching FTSE 100 constituents: {e}")
            return pd.DataFrame()

    def get_ftse250_constituents(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch FTSE 250 constituents from Wikipedia

        Args:
            use_cache: Whether to use cached data

        Returns:
            DataFrame with constituent data
        """
        cache_path = self.cache_dir / "ftse250_constituents.csv"

        if use_cache and is_cache_valid(cache_path, self.cache_duration):
            logger.info("Loading FTSE 250 constituents from cache")
            return load_cache(cache_path, format="csv")

        try:
            logger.info("Fetching FTSE 250 constituents from Wikipedia")
            url = INDICES["FTSE250"]["constituents_url"]

            # Fetch with proper headers to avoid 403 errors
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()

            tables = pd.read_html(response.text)

            # Try different table indices to find the constituents table
            df = None
            for table_idx in [2, 3, 4]:
                if table_idx < len(tables):
                    temp_df = tables[table_idx]
                    # Handle multi-level columns if present
                    if isinstance(temp_df.columns, pd.MultiIndex):
                        # Try different levels to find the one with proper column names
                        for level in range(temp_df.columns.nlevels):
                            test_cols = temp_df.columns.get_level_values(level)
                            if any(isinstance(col, str) and len(str(col)) > 1 for col in test_cols):
                                temp_df.columns = test_cols
                                break
                        else:
                            temp_df.columns = temp_df.columns.get_level_values(-1)

                    # If columns are numeric, try using first row as headers
                    if all(isinstance(col, (int, float)) for col in temp_df.columns):
                        temp_df.columns = temp_df.iloc[0]
                        temp_df = temp_df[1:].reset_index(drop=True)

                    # Check if this looks like a constituents table
                    if len(temp_df) > 100:  # FTSE 250 should have ~250 rows
                        col_str = ' '.join(str(col).lower() for col in temp_df.columns)
                        if ('ticker' in col_str or 'epic' in col_str) and ('company' in col_str or 'name' in col_str):
                            df = temp_df
                            logger.info(f"Found FTSE 250 constituents in table {table_idx}")
                            break

            if df is None:
                raise ValueError("Could not find FTSE 250 constituents table")

            logger.info(f"FTSE 250 table columns: {list(df.columns)}")

            # Try to find the right columns flexibly
            ticker_col = None
            name_col = None
            sector_col = None

            for col in df.columns:
                col_lower = str(col).lower()
                if 'ticker' in col_lower or 'epic' in col_lower or 'symbol' in col_lower:
                    ticker_col = col
                elif 'company' in col_lower or (col_lower == 'name' or 'name' in col_lower):
                    name_col = col
                elif 'sector' in col_lower:
                    sector_col = col

            if not ticker_col:
                raise ValueError(f"Could not find ticker column in {list(df.columns)}")

            # Build result dataframe with available columns
            result = pd.DataFrame()

            # Add .L suffix for London Stock Exchange tickers
            tickers = df[ticker_col].astype(str).str.strip()
            # Only add .L if not already present
            result['Ticker'] = tickers.apply(lambda x: x if x.endswith('.L') else f"{x}.L")

            result['Name'] = df[name_col] if name_col else ''
            result['Sector'] = df[sector_col] if sector_col else ''
            result['Industry'] = ''  # FTSE typically doesn't have industry level detail
            result['Index'] = 'FTSE 250'

            save_cache(result, cache_path, format="csv")
            logger.info(f"Fetched {len(result)} FTSE 250 constituents")

            return result

        except Exception as e:
            logger.error(f"Error fetching FTSE 250 constituents: {e}")
            return pd.DataFrame()

    def get_constituents(
        self,
        indices: List[str],
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Get constituents for specified indices

        Args:
            indices: List of index keys (e.g., ['SP500', 'NASDAQ100'])
            use_cache: Whether to use cached data

        Returns:
            Combined DataFrame of all constituents
        """
        all_constituents = []

        for index_key in indices:
            if index_key == 'SP500':
                df = self.get_sp500_constituents(use_cache)
            elif index_key == 'NASDAQ100':
                df = self.get_nasdaq100_constituents(use_cache)
            elif index_key == 'FTSE100':
                df = self.get_ftse100_constituents(use_cache)
            elif index_key == 'FTSE250':
                df = self.get_ftse250_constituents(use_cache)
            else:
                logger.warning(f"Unknown index: {index_key}")
                continue

            if not df.empty:
                all_constituents.append(df)

        if all_constituents:
            combined = pd.concat(all_constituents, ignore_index=True)
            logger.info(f"Total constituents across all indices: {len(combined)}")
            return combined
        else:
            return pd.DataFrame()

    def filter_by_market_cap(
        self,
        constituents: pd.DataFrame,
        top_n: int = TOP_N_PER_INDEX,
        min_market_cap: float = MIN_MARKET_CAP_MILLIONS
    ) -> pd.DataFrame:
        """
        Filter constituents by market cap, taking top N from each index

        Args:
            constituents: DataFrame of constituents
            top_n: Number of top stocks to keep per index
            min_market_cap: Minimum market cap in millions

        Returns:
            Filtered DataFrame
        """
        if constituents.empty:
            return constituents

        # Validate required columns exist
        if 'Ticker' not in constituents.columns:
            logger.error(f"Constituents DataFrame missing 'Ticker' column. Columns: {list(constituents.columns)}")
            return pd.DataFrame()

        if 'Index' not in constituents.columns:
            logger.error(f"Constituents DataFrame missing 'Index' column. Columns: {list(constituents.columns)}")
            return pd.DataFrame()

        logger.info(f"Filtering to top {top_n} stocks per index by market cap")

        # Fetch market cap info for all tickers
        tickers = constituents['Ticker'].tolist()
        info_dict = self.data_loader.get_multiple_info(tickers, use_cache=True)

        # Add market cap to dataframe
        market_caps = []
        for ticker in constituents['Ticker']:
            if ticker in info_dict:
                market_cap = info_dict[ticker].get('marketCap', 0)
                market_caps.append(market_cap / 1e6)  # Convert to millions
            else:
                market_caps.append(0)

        constituents['MarketCap'] = market_caps

        # Filter by minimum market cap
        constituents = constituents[constituents['MarketCap'] >= min_market_cap]

        # Take top N from each index
        filtered = []
        for index_name in constituents['Index'].unique():
            index_df = constituents[constituents['Index'] == index_name]
            index_df = index_df.nlargest(top_n, 'MarketCap')
            filtered.append(index_df)

        result = pd.concat(filtered, ignore_index=True)
        logger.info(f"Filtered to {len(result)} stocks total")

        return result


# Convenience function
def get_index_constituents(
    indices: List[str],
    top_n: int = TOP_N_PER_INDEX,
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Get and filter constituents for specified indices

    Args:
        indices: List of index keys
        top_n: Number of top stocks per index
        use_cache: Whether to use cached data

    Returns:
        DataFrame of filtered constituents
    """
    manager = ConstituentsManager()
    constituents = manager.get_constituents(indices, use_cache=use_cache)

    if not constituents.empty:
        constituents = manager.filter_by_market_cap(constituents, top_n=top_n)

    return constituents
