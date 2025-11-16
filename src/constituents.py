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

            # Read the first table from Wikipedia
            tables = pd.read_html(url)
            df = tables[0]

            # Rename columns to standard format
            df = df.rename(columns={
                'Symbol': 'Ticker',
                'Security': 'Name',
                'GICS Sector': 'Sector',
                'GICS Sub-Industry': 'Industry'
            })

            # Select relevant columns
            df = df[['Ticker', 'Name', 'Sector', 'Industry']]
            df['Index'] = 'S&P 500'

            save_cache(df, cache_path, format="csv")
            logger.info(f"Fetched {len(df)} S&P 500 constituents")

            return df

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

            tables = pd.read_html(url)
            df = tables[3]  # NASDAQ-100 table is usually the 4th table

            # Rename columns
            df = df.rename(columns={
                'Ticker': 'Ticker',
                'Company': 'Name',
                'GICS Sector': 'Sector',
                'GICS Sub-Industry': 'Industry'
            })

            # Handle different column names
            if 'Company' not in df.columns and 'Security' in df.columns:
                df = df.rename(columns={'Security': 'Name'})

            # Select relevant columns
            cols = ['Ticker', 'Name']
            if 'Sector' in df.columns:
                cols.append('Sector')
            else:
                df['Sector'] = 'Technology'  # Default for NASDAQ

            if 'Industry' in df.columns:
                cols.append('Industry')
            else:
                df['Industry'] = 'Unknown'

            df = df[cols]
            df['Index'] = 'NASDAQ-100'

            save_cache(df, cache_path, format="csv")
            logger.info(f"Fetched {len(df)} NASDAQ-100 constituents")

            return df

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

            tables = pd.read_html(url)
            df = tables[3]  # Constituents table

            # Rename columns
            df = df.rename(columns={
                'Ticker': 'Ticker',
                'Company': 'Name',
                'FTSE Industry Classification Benchmark sector[10]': 'Sector'
            })

            # Handle different column formats
            if 'Company' not in df.columns and 'Name' not in df.columns:
                # Try to find the right column
                for col in df.columns:
                    if 'company' in col.lower() or 'name' in col.lower():
                        df = df.rename(columns={col: 'Name'})
                        break

            # Add .L suffix for London Stock Exchange
            if 'Ticker' in df.columns:
                df['Ticker'] = df['Ticker'].astype(str) + '.L'

            # Select relevant columns
            cols = ['Ticker', 'Name']
            if 'Sector' in df.columns:
                cols.append('Sector')
            else:
                df['Sector'] = 'Unknown'

            df['Industry'] = 'Unknown'
            cols.extend(['Sector', 'Industry'])

            df = df[cols]
            df['Index'] = 'FTSE 100'

            save_cache(df, cache_path, format="csv")
            logger.info(f"Fetched {len(df)} FTSE 100 constituents")

            return df

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

            tables = pd.read_html(url)
            df = tables[3]  # Constituents table

            # Similar processing as FTSE 100
            df = df.rename(columns={
                'Ticker': 'Ticker',
                'Company': 'Name',
                'FTSE Industry Classification Benchmark sector': 'Sector'
            })

            if 'Company' not in df.columns and 'Name' not in df.columns:
                for col in df.columns:
                    if 'company' in col.lower() or 'name' in col.lower():
                        df = df.rename(columns={col: 'Name'})
                        break

            if 'Ticker' in df.columns:
                df['Ticker'] = df['Ticker'].astype(str) + '.L'

            cols = ['Ticker', 'Name']
            if 'Sector' in df.columns:
                cols.append('Sector')
            else:
                df['Sector'] = 'Unknown'

            df['Industry'] = 'Unknown'
            cols.extend(['Sector', 'Industry'])

            df = df[cols]
            df['Index'] = 'FTSE 250'

            save_cache(df, cache_path, format="csv")
            logger.info(f"Fetched {len(df)} FTSE 250 constituents")

            return df

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
