"""
Recommendation engine that orchestrates analysis and generates stock recommendations
"""
import logging
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from .config import PORTFOLIO_CONSTRAINTS, MAX_WORKERS
from .data_loader import DataLoader
from .constituents import ConstituentsManager
from .features import FeatureEngine
from .news_client import NewsClient
from .sentiment import SentimentAnalyzer
from .scoring import StockScorer
from .utils import setup_logger

logger = setup_logger(__name__)


class RecommendationEngine:
    """
    Main engine for generating stock recommendations
    """

    def __init__(
        self,
        risk_profile: str = "balanced",
        use_sentiment: bool = True,
        use_finbert: bool = False
    ):
        """
        Initialize RecommendationEngine

        Args:
            risk_profile: Risk profile ('conservative', 'balanced', 'aggressive')
            use_sentiment: Whether to use sentiment analysis
            use_finbert: Whether to use FinBERT (slower but more accurate)
        """
        self.risk_profile = risk_profile
        self.use_sentiment = use_sentiment

        # Initialize components
        self.data_loader = DataLoader()
        self.constituents_manager = ConstituentsManager()
        self.feature_engine = FeatureEngine()
        self.news_client = NewsClient()
        self.sentiment_analyzer = SentimentAnalyzer(use_finbert=use_finbert)
        self.scorer = StockScorer(risk_profile=risk_profile)

        logger.info(f"RecommendationEngine initialized ({risk_profile}, sentiment={use_sentiment})")

    def analyze_stock(
        self,
        ticker: str,
        company_name: str = None,
        index_data: pd.DataFrame = None
    ) -> Optional[Dict]:
        """
        Perform complete analysis on a single stock

        Args:
            ticker: Stock ticker
            company_name: Company name (for news search)
            index_data: Index data for relative performance

        Returns:
            Dictionary with complete analysis or None if failed
        """
        try:
            # 1. Get price data
            price_data = self.data_loader.get_stock_data(ticker, period="2y")
            if price_data is None or price_data.empty:
                logger.warning(f"No price data for {ticker}")
                return None

            # 2. Get fundamentals
            fundamentals = self.data_loader.get_stock_info(ticker)
            if not fundamentals:
                fundamentals = {}

            # 3. Calculate features
            data_with_features = self.feature_engine.calculate_all_features(price_data)
            latest_features = self.feature_engine.get_latest_features(data_with_features)

            # 4. Get news and sentiment
            sentiment_data = {}
            if self.use_sentiment:
                articles = self.news_client.get_stock_news(ticker, company_name=company_name)
                if articles:
                    sentiment_data = self.sentiment_analyzer.get_sentiment_summary(ticker, articles)
                else:
                    sentiment_data = {'total_articles': 0}

            # 5. Calculate relative performance
            relative_perf = 0
            if index_data is not None and not index_data.empty:
                relative_perf = self.feature_engine.calculate_relative_performance(
                    price_data, index_data, period=126
                )

            # 6. Calculate score and recommendation
            score_result = self.scorer.calculate_overall_score(
                latest_features,
                fundamentals,
                sentiment_data,
                use_sentiment=self.use_sentiment
            )

            # 7. Generate explanation
            explanation = self._generate_explanation(
                ticker,
                score_result,
                fundamentals,
                sentiment_data,
                relative_perf
            )

            # 8. Compile result
            result = {
                'ticker': ticker,
                'name': company_name or fundamentals.get('longName', ticker),
                'sector': fundamentals.get('sector', 'Unknown'),
                'industry': fundamentals.get('industry', 'Unknown'),
                'market_cap': fundamentals.get('marketCap', 0),
                'current_price': price_data['Close'].iloc[-1],
                'score': score_result['overall_score'],
                'action': score_result['action'],
                'confidence': score_result['confidence'],
                'explanation': explanation,
                'component_scores': score_result['component_scores'],
                'relative_performance_6m': relative_perf,
                'sentiment': sentiment_data.get('overall_sentiment', 'neutral'),
                'sentiment_score': sentiment_data.get('average_score', 0),
                'recent_return_1m': latest_features.get('return_21d', 0),
                'recent_return_3m': latest_features.get('return_63d', 0),
                'volatility': latest_features.get('volatility_annual', 0),
                'pe_ratio': fundamentals.get('trailingPE', None),
                'dividend_yield': fundamentals.get('dividendYield', None),
            }

            return result

        except Exception as e:
            logger.error(f"Error analyzing {ticker}: {e}")
            return None

    def _generate_explanation(
        self,
        ticker: str,
        score_result: Dict,
        fundamentals: Dict,
        sentiment_data: Dict,
        relative_perf: float
    ) -> str:
        """
        Generate human-readable explanation for recommendation

        Args:
            ticker: Stock ticker
            score_result: Scoring results
            fundamentals: Fundamental data
            sentiment_data: Sentiment data
            relative_perf: Relative performance vs index

        Returns:
            Explanation string
        """
        action = score_result['action']
        explanations = score_result['explanations']

        # Start with action
        parts = []

        # Add key drivers
        if action in ['STRONG BUY', 'BUY']:
            parts.append(f"**{action}**: Strong opportunity based on:")
        elif action == 'HOLD':
            parts.append(f"**{action}**: Moderate opportunity:")
        elif action == 'WATCH':
            parts.append(f"**{action}**: Proceed with caution:")
        else:
            parts.append(f"**{action}**: Consider reducing position:")

        # Add component explanations
        for key in ['momentum', 'trend', 'value', 'sentiment']:
            exp = explanations.get(key)
            if exp and (key != 'sentiment' or self.use_sentiment):
                parts.append(f"• {exp}")

        # Add relative performance
        if relative_perf != 0:
            perf_pct = relative_perf * 100
            if abs(perf_pct) > 5:
                if perf_pct > 0:
                    parts.append(f"• Outperforming index by {perf_pct:.1f}% (6M)")
                else:
                    parts.append(f"• Underperforming index by {abs(perf_pct):.1f}% (6M)")

        # Add risk factors
        volatility_score = score_result['component_scores'].get('volatility', 0.5)
        if volatility_score < 0.4 and self.risk_profile == 'conservative':
            parts.append("• ⚠️ High volatility - higher risk")

        explanation = "\n".join(parts)
        return explanation

    def analyze_portfolio(
        self,
        tickers: List[str],
        indices: List[str],
        top_n: int = 50,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Analyze a portfolio of stocks

        Args:
            tickers: List of tickers (if empty, will fetch from indices)
            indices: List of index keys to analyze
            top_n: Top N stocks per index
            show_progress: Show progress bar

        Returns:
            DataFrame with analysis results
        """
        # Get constituents if tickers not provided
        if not tickers:
            logger.info(f"Fetching constituents for {indices}")
            constituents_df = self.constituents_manager.get_constituents(indices)

            if not constituents_df.empty:
                constituents_df = self.constituents_manager.filter_by_market_cap(
                    constituents_df, top_n=top_n
                )
                tickers = constituents_df['Ticker'].tolist()
                # Create mapping of ticker to company name
                ticker_to_name = dict(zip(
                    constituents_df['Ticker'],
                    constituents_df['Name']
                ))
            else:
                logger.error("Could not fetch constituents")
                return pd.DataFrame()
        else:
            ticker_to_name = {}

        logger.info(f"Analyzing {len(tickers)} stocks")

        # Get index data for relative performance
        index_data_dict = {}
        for index_key in indices:
            index_data = self.data_loader.get_index_data(index_key, period="2y")
            if index_data is not None:
                index_data_dict[index_key] = index_data

        # Use first index for relative performance
        index_data = next(iter(index_data_dict.values())) if index_data_dict else None

        # Analyze all stocks in parallel
        results = []

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_ticker = {
                executor.submit(
                    self.analyze_stock,
                    ticker,
                    ticker_to_name.get(ticker),
                    index_data
                ): ticker
                for ticker in tickers
            }

            iterator = as_completed(future_to_ticker)
            if show_progress:
                iterator = tqdm(iterator, total=len(tickers), desc="Analyzing stocks")

            for future in iterator:
                ticker = future_to_ticker[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Error processing {ticker}: {e}")

        # Convert to DataFrame
        if results:
            df = pd.DataFrame(results)
            df = df.sort_values('score', ascending=False)
            logger.info(f"Successfully analyzed {len(df)} stocks")
            return df
        else:
            logger.warning("No successful analyses")
            return pd.DataFrame()

    def get_recommendations(
        self,
        analysis_df: pd.DataFrame,
        num_buy: int = 20,
        num_watch: int = 10,
        existing_portfolio: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate buy/watch/sell recommendations from analysis

        Args:
            analysis_df: DataFrame with analysis results
            num_buy: Number of buy recommendations
            num_watch: Number of watch recommendations
            existing_portfolio: List of currently held tickers

        Returns:
            Dictionary with 'buy', 'watch', 'sell' DataFrames
        """
        if analysis_df.empty:
            return {'buy': pd.DataFrame(), 'watch': pd.DataFrame(), 'sell': pd.DataFrame()}

        existing_portfolio = existing_portfolio or []

        # Separate existing holdings from new opportunities
        held_df = analysis_df[analysis_df['ticker'].isin(existing_portfolio)]
        new_df = analysis_df[~analysis_df['ticker'].isin(existing_portfolio)]

        # Buy recommendations: top scoring new stocks
        buy_df = new_df[new_df['action'].isin(['STRONG BUY', 'BUY'])].head(num_buy)

        # Watch recommendations: moderate scoring new stocks
        watch_df = new_df[new_df['action'].isin(['HOLD', 'WATCH'])].head(num_watch)

        # Sell recommendations: poorly performing holdings
        if not held_df.empty:
            sell_df = held_df[held_df['action'].isin(['SELL', 'WATCH'])]
        else:
            sell_df = pd.DataFrame()

        logger.info(f"Generated {len(buy_df)} BUY, {len(watch_df)} WATCH, {len(sell_df)} SELL recommendations")

        return {
            'buy': buy_df,
            'watch': watch_df,
            'sell': sell_df,
            'all_holdings': held_df
        }


def create_recommendation_engine(
    risk_profile: str = "balanced",
    use_sentiment: bool = True,
    use_finbert: bool = False
) -> RecommendationEngine:
    """
    Create and return a RecommendationEngine instance

    Args:
        risk_profile: Risk profile
        use_sentiment: Whether to use sentiment
        use_finbert: Whether to use FinBERT

    Returns:
        RecommendationEngine instance
    """
    return RecommendationEngine(
        risk_profile=risk_profile,
        use_sentiment=use_sentiment,
        use_finbert=use_finbert
    )
