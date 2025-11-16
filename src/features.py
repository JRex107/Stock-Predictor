"""
Feature engineering module for calculating technical indicators and features
"""
import logging
from typing import Dict, List, Optional
import warnings

import pandas as pd
import numpy as np
import pandas_ta as ta

from .config import FEATURE_PARAMS
from .utils import setup_logger, safe_division


warnings.filterwarnings('ignore')
logger = setup_logger(__name__)


class FeatureEngine:
    """
    Calculates technical indicators and features for stock analysis
    """

    def __init__(self, params: Dict = None):
        """
        Initialize FeatureEngine

        Args:
            params: Feature parameters (uses defaults from config if None)
        """
        self.params = params or FEATURE_PARAMS
        logger.info("FeatureEngine initialized")

    def calculate_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate returns over various periods

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with return columns added
        """
        df = data.copy()

        for period in self.params['returns_periods']:
            col_name = f'return_{period}d'
            df[col_name] = df['Close'].pct_change(periods=period)

        return df

    def calculate_volatility(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volatility measures

        Args:
            data: DataFrame with price data

        Returns:
            DataFrame with volatility columns added
        """
        df = data.copy()
        window = self.params['volatility_window']

        # Standard deviation of returns
        df['volatility'] = df['Close'].pct_change().rolling(window=window).std()

        # Annualized volatility
        df['volatility_annual'] = df['volatility'] * np.sqrt(252)

        # ATR (Average True Range)
        df['atr'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

        return df

    def calculate_moving_averages(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate moving averages and crossovers

        Args:
            data: DataFrame with price data

        Returns:
            DataFrame with MA columns added
        """
        df = data.copy()

        for window in self.params['ma_windows']:
            # Simple Moving Average
            df[f'sma_{window}'] = df['Close'].rolling(window=window).mean()

            # Exponential Moving Average
            df[f'ema_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()

            # Distance from MA (as percentage)
            df[f'dist_from_sma_{window}'] = (
                (df['Close'] - df[f'sma_{window}']) / df[f'sma_{window}']
            )

        # Golden Cross / Death Cross signals
        if 50 in self.params['ma_windows'] and 200 in self.params['ma_windows']:
            df['golden_cross'] = (
                (df['sma_50'] > df['sma_200']) &
                (df['sma_50'].shift(1) <= df['sma_200'].shift(1))
            ).astype(int)

            df['death_cross'] = (
                (df['sma_50'] < df['sma_200']) &
                (df['sma_50'].shift(1) >= df['sma_200'].shift(1))
            ).astype(int)

        return df

    def calculate_momentum_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate momentum indicators

        Args:
            data: DataFrame with price data

        Returns:
            DataFrame with momentum columns added
        """
        df = data.copy()

        # RSI (Relative Strength Index)
        df['rsi'] = ta.rsi(df['Close'], length=self.params['rsi_period'])

        # MACD
        macd = ta.macd(
            df['Close'],
            fast=self.params['macd_fast'],
            slow=self.params['macd_slow'],
            signal=self.params['macd_signal']
        )
        if macd is not None:
            df['macd'] = macd[f"MACD_{self.params['macd_fast']}_{self.params['macd_slow']}_{self.params['macd_signal']}"]
            df['macd_signal'] = macd[f"MACDs_{self.params['macd_fast']}_{self.params['macd_slow']}_{self.params['macd_signal']}"]
            df['macd_hist'] = macd[f"MACDh_{self.params['macd_fast']}_{self.params['macd_slow']}_{self.params['macd_signal']}"]

        # Stochastic Oscillator
        stoch = ta.stoch(df['High'], df['Low'], df['Close'])
        if stoch is not None:
            df['stoch_k'] = stoch['STOCHk_14_3_3']
            df['stoch_d'] = stoch['STOCHd_14_3_3']

        return df

    def calculate_volume_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume-based indicators

        Args:
            data: DataFrame with volume data

        Returns:
            DataFrame with volume columns added
        """
        df = data.copy()
        window = self.params['volume_window']

        # Average volume
        df['avg_volume'] = df['Volume'].rolling(window=window).mean()

        # Volume ratio (current volume vs average)
        df['volume_ratio'] = safe_division(df['Volume'], df['avg_volume'], default=1.0)

        # On-Balance Volume (OBV)
        df['obv'] = ta.obv(df['Close'], df['Volume'])

        # Volume-Weighted Average Price (VWAP) - for intraday, approximate with daily
        df['vwap'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()

        return df

    def calculate_trend_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trend strength indicators

        Args:
            data: DataFrame with price data

        Returns:
            DataFrame with trend columns added
        """
        df = data.copy()

        # ADX (Average Directional Index) - trend strength
        adx = ta.adx(df['High'], df['Low'], df['Close'])
        if adx is not None:
            df['adx'] = adx['ADX_14']
            df['di_plus'] = adx['DMP_14']
            df['di_minus'] = adx['DMN_14']

        # Bollinger Bands
        bbands = ta.bbands(df['Close'], length=20, std=2)
        if bbands is not None:
            df['bb_upper'] = bbands['BBU_20_2.0']
            df['bb_middle'] = bbands['BBM_20_2.0']
            df['bb_lower'] = bbands['BBL_20_2.0']
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        return df

    def calculate_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all features for a stock

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with all features added
        """
        if data is None or data.empty:
            return pd.DataFrame()

        try:
            df = data.copy()

            # Calculate all feature groups
            df = self.calculate_returns(df)
            df = self.calculate_volatility(df)
            df = self.calculate_moving_averages(df)
            df = self.calculate_momentum_indicators(df)
            df = self.calculate_volume_indicators(df)
            df = self.calculate_trend_indicators(df)

            logger.debug(f"Calculated {len(df.columns)} features")
            return df

        except Exception as e:
            logger.error(f"Error calculating features: {e}")
            return data

    def get_latest_features(
        self,
        data: pd.DataFrame,
        include_columns: Optional[List[str]] = None
    ) -> Dict:
        """
        Get the latest values of calculated features

        Args:
            data: DataFrame with features calculated
            include_columns: Specific columns to include (None = all)

        Returns:
            Dictionary of latest feature values
        """
        if data.empty:
            return {}

        latest = data.iloc[-1]

        if include_columns:
            features = {col: latest[col] for col in include_columns if col in latest.index}
        else:
            # Exclude OHLCV columns, keep only calculated features
            base_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            features = {
                col: latest[col]
                for col in latest.index
                if col not in base_cols
            }

        # Convert numpy types to Python types for JSON serialization
        features = {
            k: float(v) if isinstance(v, (np.integer, np.floating)) else v
            for k, v in features.items()
        }

        return features

    def calculate_relative_performance(
        self,
        stock_data: pd.DataFrame,
        index_data: pd.DataFrame,
        period: int = 252
    ) -> float:
        """
        Calculate stock's relative performance vs index

        Args:
            stock_data: Stock price data
            index_data: Index price data
            period: Period in days

        Returns:
            Relative performance (positive = outperformance)
        """
        if stock_data.empty or index_data.empty:
            return 0.0

        try:
            # Align data
            aligned = pd.DataFrame({
                'stock': stock_data['Close'],
                'index': index_data['Close']
            }).dropna()

            if len(aligned) < period:
                period = len(aligned)

            # Calculate returns
            stock_return = (aligned['stock'].iloc[-1] / aligned['stock'].iloc[-period]) - 1
            index_return = (aligned['index'].iloc[-1] / aligned['index'].iloc[-period]) - 1

            return stock_return - index_return

        except Exception as e:
            logger.error(f"Error calculating relative performance: {e}")
            return 0.0


def create_feature_engine(params: Dict = None) -> FeatureEngine:
    """
    Create and return a FeatureEngine instance

    Args:
        params: Feature parameters

    Returns:
        FeatureEngine instance
    """
    return FeatureEngine(params=params)
