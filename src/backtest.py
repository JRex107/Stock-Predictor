"""
Backtesting framework for validating the stock recommendation strategy
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings

import pandas as pd
import numpy as np

from .config import BACKTEST_PARAMS, RISK_FREE_RATE
from .data_loader import DataLoader
from .features import FeatureEngine
from .scoring import StockScorer
from .utils import (
    setup_logger,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    safe_division
)

warnings.filterwarnings('ignore')
logger = setup_logger(__name__)


class Backtester:
    """
    Backtesting framework for stock recommendations
    """

    def __init__(
        self,
        risk_profile: str = "balanced",
        portfolio_size: int = None,
        rebalance_freq: str = None,
        initial_capital: float = None,
        commission: float = None
    ):
        """
        Initialize Backtester

        Args:
            risk_profile: Risk profile for scoring
            portfolio_size: Number of stocks in portfolio
            rebalance_freq: Rebalancing frequency ('M'=monthly, 'W'=weekly)
            initial_capital: Initial capital in dollars
            commission: Commission per trade (as decimal, e.g., 0.001 = 0.1%)
        """
        self.risk_profile = risk_profile
        self.portfolio_size = portfolio_size or BACKTEST_PARAMS['portfolio_size']
        self.rebalance_freq = rebalance_freq or BACKTEST_PARAMS['rebalance_frequency']
        self.initial_capital = initial_capital or BACKTEST_PARAMS['initial_capital']
        self.commission = commission or BACKTEST_PARAMS['commission']

        self.data_loader = DataLoader()
        self.feature_engine = FeatureEngine()
        self.scorer = StockScorer(risk_profile=risk_profile)

        logger.info(f"Backtester initialized: {self.portfolio_size} stocks, {self.rebalance_freq} rebalance")

    def score_stocks_at_date(
        self,
        tickers: List[str],
        date: pd.Timestamp,
        lookback_period: str = "1y"
    ) -> pd.DataFrame:
        """
        Score all stocks at a specific date

        Args:
            tickers: List of tickers to score
            date: Date to score at
            lookback_period: Lookback period for historical data

        Returns:
            DataFrame with ticker and score
        """
        scores = []

        for ticker in tickers:
            try:
                # Get historical data up to this date
                data = self.data_loader.get_stock_data(ticker, period="5y", use_cache=True)
                if data is None or data.empty:
                    continue

                # Filter to data available up to this date
                data = data[data.index <= date]

                if len(data) < 100:  # Need minimum history
                    continue

                # Calculate features
                data_with_features = self.feature_engine.calculate_all_features(data)
                latest_features = self.feature_engine.get_latest_features(data_with_features)

                # Get fundamentals (we'll use current fundamentals for simplicity)
                fundamentals = self.data_loader.get_stock_info(ticker)
                if not fundamentals:
                    fundamentals = {}

                # Score (without sentiment for backtesting simplicity)
                score_result = self.scorer.calculate_overall_score(
                    latest_features,
                    fundamentals,
                    {},  # No sentiment
                    use_sentiment=False
                )

                scores.append({
                    'ticker': ticker,
                    'score': score_result['overall_score'],
                    'action': score_result['action']
                })

            except Exception as e:
                logger.debug(f"Error scoring {ticker} at {date}: {e}")
                continue

        return pd.DataFrame(scores)

    def select_portfolio(
        self,
        scores_df: pd.DataFrame,
        n_stocks: int
    ) -> List[str]:
        """
        Select top N stocks from scores

        Args:
            scores_df: DataFrame with scores
            n_stocks: Number of stocks to select

        Returns:
            List of selected tickers
        """
        if scores_df.empty:
            return []

        # Sort by score and take top N
        top_stocks = scores_df.nlargest(n_stocks, 'score')
        return top_stocks['ticker'].tolist()

    def calculate_portfolio_value(
        self,
        holdings: Dict[str, float],
        date: pd.Timestamp,
        price_data: Dict[str, pd.DataFrame]
    ) -> float:
        """
        Calculate total portfolio value at a date

        Args:
            holdings: Dictionary of ticker to shares held
            date: Date to value at
            price_data: Dictionary of ticker to price DataFrame

        Returns:
            Total portfolio value
        """
        total_value = 0.0

        for ticker, shares in holdings.items():
            if ticker not in price_data:
                continue

            data = price_data[ticker]
            # Get price at or before this date
            data_at_date = data[data.index <= date]

            if data_at_date.empty:
                continue

            price = data_at_date['Close'].iloc[-1]
            total_value += shares * price

        return total_value

    def run_backtest(
        self,
        tickers: List[str],
        start_date: str = None,
        end_date: str = None,
        benchmark_ticker: str = "^GSPC"
    ) -> Dict:
        """
        Run complete backtest

        Args:
            tickers: Universe of tickers to trade
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            benchmark_ticker: Benchmark index ticker

        Returns:
            Dictionary with backtest results
        """
        # Set default dates
        if not end_date:
            end_date = datetime.now()
        else:
            end_date = pd.to_datetime(end_date)

        if not start_date:
            lookback_years = BACKTEST_PARAMS['lookback_years']
            start_date = end_date - timedelta(days=lookback_years * 365)
        else:
            start_date = pd.to_datetime(start_date)

        logger.info(f"Running backtest from {start_date.date()} to {end_date.date()}")
        logger.info(f"Universe: {len(tickers)} stocks")

        # Get price data for all tickers
        logger.info("Fetching price data...")
        price_data = self.data_loader.get_multiple_stocks(
            tickers,
            period="5y",
            use_cache=True,
            show_progress=True
        )

        # Get benchmark data
        benchmark_data = self.data_loader.get_stock_data(
            benchmark_ticker,
            period="5y",
            use_cache=True
        )

        # Generate rebalancing dates
        date_range = pd.date_range(start=start_date, end=end_date, freq=self.rebalance_freq)
        rebalance_dates = [d for d in date_range if d <= end_date]

        logger.info(f"Rebalancing on {len(rebalance_dates)} dates")

        # Track portfolio
        holdings = {}  # ticker -> shares
        cash = self.initial_capital
        portfolio_values = []
        transactions = []

        # Run through each rebalance date
        for i, rebal_date in enumerate(rebalance_dates):
            logger.info(f"Rebalancing {i+1}/{len(rebalance_dates)}: {rebal_date.date()}")

            # Calculate current portfolio value
            portfolio_value = self.calculate_portfolio_value(holdings, rebal_date, price_data)
            total_value = portfolio_value + cash

            portfolio_values.append({
                'date': rebal_date,
                'portfolio_value': portfolio_value,
                'cash': cash,
                'total_value': total_value
            })

            # Score all stocks at this date
            available_tickers = [t for t in tickers if t in price_data]
            scores_df = self.score_stocks_at_date(available_tickers, rebal_date)

            if scores_df.empty:
                logger.warning(f"No scores at {rebal_date}")
                continue

            # Select new portfolio
            new_portfolio = self.select_portfolio(scores_df, self.portfolio_size)

            # Rebalance: sell everything and buy new portfolio equally weighted
            # Sell all holdings
            for ticker, shares in holdings.items():
                if ticker not in price_data:
                    continue

                data = price_data[ticker]
                data_at_date = data[data.index <= rebal_date]

                if not data_at_date.empty:
                    sell_price = data_at_date['Close'].iloc[-1]
                    proceeds = shares * sell_price * (1 - self.commission)
                    cash += proceeds

                    transactions.append({
                        'date': rebal_date,
                        'ticker': ticker,
                        'action': 'SELL',
                        'shares': shares,
                        'price': sell_price,
                        'value': proceeds
                    })

            holdings = {}

            # Buy new portfolio
            position_size = (cash + portfolio_value) / len(new_portfolio) if new_portfolio else 0

            for ticker in new_portfolio:
                if ticker not in price_data:
                    continue

                data = price_data[ticker]
                data_at_date = data[data.index <= rebal_date]

                if not data_at_date.empty:
                    buy_price = data_at_date['Close'].iloc[-1]
                    shares_to_buy = (position_size * (1 - self.commission)) / buy_price
                    cost = shares_to_buy * buy_price

                    if cost <= cash:
                        holdings[ticker] = shares_to_buy
                        cash -= cost

                        transactions.append({
                            'date': rebal_date,
                            'ticker': ticker,
                            'action': 'BUY',
                            'shares': shares_to_buy,
                            'price': buy_price,
                            'value': cost
                        })

        # Calculate final value
        final_date = end_date
        final_portfolio_value = self.calculate_portfolio_value(holdings, final_date, price_data)
        final_total_value = final_portfolio_value + cash

        # Convert to DataFrames
        portfolio_df = pd.DataFrame(portfolio_values)
        portfolio_df.set_index('date', inplace=True)

        transactions_df = pd.DataFrame(transactions)

        # Calculate returns
        portfolio_df['returns'] = portfolio_df['total_value'].pct_change()

        # Calculate benchmark returns
        benchmark_aligned = benchmark_data.reindex(portfolio_df.index, method='ffill')
        benchmark_returns = benchmark_aligned['Close'].pct_change()

        # Calculate metrics
        total_return = (final_total_value - self.initial_capital) / self.initial_capital
        cagr = self._calculate_cagr(self.initial_capital, final_total_value, (end_date - start_date).days / 365)
        sharpe = calculate_sharpe_ratio(portfolio_df['returns'].dropna(), RISK_FREE_RATE)
        max_dd = calculate_max_drawdown(portfolio_df['returns'].dropna())

        # Benchmark metrics
        benchmark_total_return = (benchmark_aligned['Close'].iloc[-1] - benchmark_aligned['Close'].iloc[0]) / benchmark_aligned['Close'].iloc[0]
        benchmark_sharpe = calculate_sharpe_ratio(benchmark_returns.dropna(), RISK_FREE_RATE)

        results = {
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': self.initial_capital,
            'final_value': final_total_value,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'cagr': cagr,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'max_drawdown_pct': max_dd * 100,
            'num_trades': len(transactions_df),
            'portfolio_history': portfolio_df,
            'transactions': transactions_df,
            'benchmark_return': benchmark_total_return,
            'benchmark_return_pct': benchmark_total_return * 100,
            'benchmark_sharpe': benchmark_sharpe,
            'outperformance': total_return - benchmark_total_return,
            'outperformance_pct': (total_return - benchmark_total_return) * 100
        }

        logger.info(f"Backtest complete: {total_return*100:.2f}% return vs {benchmark_total_return*100:.2f}% benchmark")

        return results

    def _calculate_cagr(self, start_value: float, end_value: float, years: float) -> float:
        """
        Calculate Compound Annual Growth Rate

        Args:
            start_value: Starting value
            end_value: Ending value
            years: Number of years

        Returns:
            CAGR as decimal
        """
        if years <= 0 or start_value <= 0:
            return 0.0

        return (end_value / start_value) ** (1 / years) - 1


def create_backtester(
    risk_profile: str = "balanced",
    portfolio_size: int = None
) -> Backtester:
    """
    Create and return a Backtester instance

    Args:
        risk_profile: Risk profile
        portfolio_size: Portfolio size

    Returns:
        Backtester instance
    """
    return Backtester(risk_profile=risk_profile, portfolio_size=portfolio_size)
