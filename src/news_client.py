"""
News client for fetching financial news from various sources
"""
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time

import requests
from newsapi import NewsApiClient

from .config import (
    NEWSAPI_KEY,
    RAW_DATA_DIR,
    NEWS_SETTINGS
)
from .utils import (
    setup_logger,
    is_cache_valid,
    save_cache,
    load_cache
)


logger = setup_logger(__name__)


class NewsClient:
    """
    Handles fetching news from various sources
    """

    def __init__(self, api_key: str = None):
        """
        Initialize NewsClient

        Args:
            api_key: NewsAPI key (uses config if None)
        """
        self.api_key = api_key or NEWSAPI_KEY
        self.cache_dir = RAW_DATA_DIR / "news"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if self.api_key:
            try:
                self.client = NewsApiClient(api_key=self.api_key)
                logger.info("NewsAPI client initialized")
            except Exception as e:
                logger.error(f"Error initializing NewsAPI: {e}")
                self.client = None
        else:
            logger.warning("No NewsAPI key provided - news features will be limited")
            self.client = None

    def get_stock_news(
        self,
        ticker: str,
        company_name: str = None,
        lookback_days: int = None,
        max_articles: int = None,
        use_cache: bool = True
    ) -> List[Dict]:
        """
        Get news articles for a stock

        Args:
            ticker: Stock ticker symbol
            company_name: Company name for search
            lookback_days: Days to look back
            max_articles: Maximum articles to return
            use_cache: Whether to use cached data

        Returns:
            List of article dictionaries
        """
        lookback_days = lookback_days or NEWS_SETTINGS['lookback_days']
        max_articles = max_articles or NEWS_SETTINGS['max_articles_per_stock']

        cache_path = self.cache_dir / f"{ticker}_news.pickle"

        # Check cache
        cache_hours = NEWS_SETTINGS['sentiment_cache_hours']
        if use_cache and is_cache_valid(cache_path, cache_hours):
            logger.debug(f"Loading news for {ticker} from cache")
            cached_data = load_cache(cache_path, format="pickle")
            if cached_data:
                return cached_data[:max_articles]

        # Fetch fresh news
        if not self.client:
            logger.warning(f"No NewsAPI client - cannot fetch news for {ticker}")
            return []

        try:
            # Calculate date range
            to_date = datetime.now()
            from_date = to_date - timedelta(days=lookback_days)

            # Build search query
            search_query = ticker
            if company_name:
                # Use company name for better results
                search_query = f'"{company_name}" OR {ticker}'

            logger.debug(f"Fetching news for {ticker}: {search_query}")

            # Fetch from NewsAPI
            response = self.client.get_everything(
                q=search_query,
                from_param=from_date.strftime('%Y-%m-%d'),
                to=to_date.strftime('%Y-%m-%d'),
                language='en',
                sort_by='relevancy',
                page_size=min(max_articles, 100)
            )

            articles = []
            if response['status'] == 'ok':
                for article in response['articles']:
                    articles.append({
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'content': article.get('content', ''),
                        'url': article.get('url', ''),
                        'source': article.get('source', {}).get('name', 'Unknown'),
                        'publishedAt': article.get('publishedAt', ''),
                        'ticker': ticker
                    })

                logger.info(f"Fetched {len(articles)} articles for {ticker}")

                # Save to cache
                save_cache(articles, cache_path, format="pickle")

                return articles[:max_articles]
            else:
                logger.warning(f"NewsAPI error for {ticker}: {response.get('message', 'Unknown')}")
                return []

        except Exception as e:
            logger.error(f"Error fetching news for {ticker}: {e}")
            return []

    def get_market_news(
        self,
        topics: List[str] = None,
        lookback_days: int = 7,
        max_articles: int = 20,
        use_cache: bool = True
    ) -> List[Dict]:
        """
        Get general market news

        Args:
            topics: List of topics/keywords
            lookback_days: Days to look back
            max_articles: Maximum articles to return
            use_cache: Whether to use cached data

        Returns:
            List of article dictionaries
        """
        topics = topics or ['stock market', 'economy', 'trading']
        cache_key = "_".join(sorted(topics))
        cache_path = self.cache_dir / f"market_{cache_key}_news.pickle"

        # Check cache
        if use_cache and is_cache_valid(cache_path, NEWS_SETTINGS['sentiment_cache_hours']):
            logger.debug("Loading market news from cache")
            cached_data = load_cache(cache_path, format="pickle")
            if cached_data:
                return cached_data[:max_articles]

        if not self.client:
            logger.warning("No NewsAPI client - cannot fetch market news")
            return []

        try:
            to_date = datetime.now()
            from_date = to_date - timedelta(days=lookback_days)

            # Build query
            query = " OR ".join([f'"{topic}"' for topic in topics])

            logger.debug(f"Fetching market news: {query}")

            response = self.client.get_everything(
                q=query,
                from_param=from_date.strftime('%Y-%m-%d'),
                to=to_date.strftime('%Y-%m-%d'),
                language='en',
                sort_by='publishedAt',
                page_size=min(max_articles, 100)
            )

            articles = []
            if response['status'] == 'ok':
                for article in response['articles']:
                    articles.append({
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'content': article.get('content', ''),
                        'url': article.get('url', ''),
                        'source': article.get('source', {}).get('name', 'Unknown'),
                        'publishedAt': article.get('publishedAt', ''),
                    })

                logger.info(f"Fetched {len(articles)} market news articles")
                save_cache(articles, cache_path, format="pickle")
                return articles[:max_articles]
            else:
                logger.warning(f"NewsAPI error: {response.get('message', 'Unknown')}")
                return []

        except Exception as e:
            logger.error(f"Error fetching market news: {e}")
            return []

    def get_batch_news(
        self,
        tickers: List[str],
        company_names: Dict[str, str] = None,
        delay_between_requests: float = 1.0
    ) -> Dict[str, List[Dict]]:
        """
        Get news for multiple tickers (with rate limiting)

        Args:
            tickers: List of tickers
            company_names: Mapping of ticker to company name
            delay_between_requests: Delay in seconds between API calls

        Returns:
            Dictionary mapping ticker to articles
        """
        company_names = company_names or {}
        results = {}

        for ticker in tickers:
            company_name = company_names.get(ticker)
            articles = self.get_stock_news(ticker, company_name=company_name)
            results[ticker] = articles

            # Rate limiting
            if delay_between_requests > 0:
                time.sleep(delay_between_requests)

        logger.info(f"Fetched news for {len(results)} stocks")
        return results


def create_news_client(api_key: str = None) -> NewsClient:
    """
    Create and return a NewsClient instance

    Args:
        api_key: NewsAPI key

    Returns:
        NewsClient instance
    """
    return NewsClient(api_key=api_key)
