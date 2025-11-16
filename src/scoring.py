"""
Scoring model for ranking and evaluating stocks
"""
import logging
from typing import Dict, List, Optional, Tuple
import warnings

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from .config import (
    SCORING_WEIGHTS,
    RECOMMENDATION_THRESHOLDS,
    RISK_FREE_RATE
)
from .utils import setup_logger, safe_division

warnings.filterwarnings('ignore')
logger = setup_logger(__name__)


class StockScorer:
    """
    Scores stocks based on multiple factors
    """

    def __init__(self, risk_profile: str = "balanced"):
        """
        Initialize StockScorer

        Args:
            risk_profile: Risk profile ('conservative', 'balanced', 'aggressive')
        """
        self.risk_profile = risk_profile
        self.weights = SCORING_WEIGHTS.get(risk_profile, SCORING_WEIGHTS['balanced'])
        self.thresholds = RECOMMENDATION_THRESHOLDS.get(risk_profile, RECOMMENDATION_THRESHOLDS['balanced'])
        logger.info(f"StockScorer initialized with {risk_profile} profile")

    def calculate_momentum_score(self, features: Dict) -> Tuple[float, str]:
        """
        Calculate momentum score

        Args:
            features: Dictionary of feature values

        Returns:
            Tuple of (score, explanation)
        """
        # Use multiple return periods
        returns = []
        weights = [0.1, 0.2, 0.3, 0.25, 0.15]  # Weight recent returns more
        periods = ['return_21d', 'return_63d', 'return_126d', 'return_252d']

        for i, period in enumerate(periods):
            ret = features.get(period, 0)
            if not pd.isna(ret):
                returns.append(ret)
            else:
                returns.append(0)

        # Weighted average
        if returns:
            momentum = sum(r * w for r, w in zip(returns, weights[:len(returns)]))
        else:
            momentum = 0

        # Normalize to 0-1 scale (assume -50% to +100% range)
        normalized = (momentum + 0.5) / 1.5
        normalized = max(0, min(1, normalized))

        # Generate explanation
        recent_return = features.get('return_21d', 0) * 100
        yearly_return = features.get('return_252d', 0) * 100

        if momentum > 0.15:
            explanation = f"Strong upward momentum ({yearly_return:.1f}% yearly)"
        elif momentum > 0:
            explanation = f"Positive momentum ({yearly_return:.1f}% yearly)"
        elif momentum > -0.15:
            explanation = f"Weak momentum ({yearly_return:.1f}% yearly)"
        else:
            explanation = f"Negative momentum ({yearly_return:.1f}% yearly)"

        return normalized, explanation

    def calculate_volatility_score(self, features: Dict) -> Tuple[float, str]:
        """
        Calculate volatility score (lower volatility = higher score for conservative)

        Args:
            features: Dictionary of feature values

        Returns:
            Tuple of (score, explanation)
        """
        volatility = features.get('volatility_annual', 0)

        # Normalize volatility (assume 0-100% annual volatility range)
        # For conservative: lower volatility is better
        # For aggressive: we care less about volatility
        if self.risk_profile == 'conservative':
            # Invert: low volatility = high score
            normalized = max(0, min(1, 1 - (volatility / 1.0)))
        elif self.risk_profile == 'aggressive':
            # Don't penalize volatility as much
            normalized = max(0, min(1, 1 - (volatility / 2.0)))
        else:
            # Balanced
            normalized = max(0, min(1, 1 - (volatility / 1.5)))

        vol_pct = volatility * 100
        if volatility < 0.2:
            explanation = f"Low volatility ({vol_pct:.1f}%)"
        elif volatility < 0.4:
            explanation = f"Moderate volatility ({vol_pct:.1f}%)"
        else:
            explanation = f"High volatility ({vol_pct:.1f}%)"

        return normalized, explanation

    def calculate_trend_score(self, features: Dict) -> Tuple[float, str]:
        """
        Calculate trend score based on MA and technical indicators

        Args:
            features: Dictionary of feature values

        Returns:
            Tuple of (score, explanation)
        """
        scores = []
        signals = []

        # Moving average position
        dist_sma_20 = features.get('dist_from_sma_20', 0)
        dist_sma_50 = features.get('dist_from_sma_50', 0)
        dist_sma_200 = features.get('dist_from_sma_200', 0)

        # Score based on position relative to MAs
        # Positive = above MA (bullish), negative = below MA (bearish)
        ma_score = (
            0.3 * (1 if dist_sma_20 > 0 else 0) +
            0.4 * (1 if dist_sma_50 > 0 else 0) +
            0.3 * (1 if dist_sma_200 > 0 else 0)
        )
        scores.append(ma_score)

        if dist_sma_20 > 0 and dist_sma_50 > 0:
            signals.append("above key MAs")

        # ADX (trend strength)
        adx = features.get('adx', 0)
        if not pd.isna(adx):
            # Strong trend if ADX > 25
            adx_score = min(1.0, adx / 50)
            scores.append(adx_score)
            if adx > 25:
                signals.append("strong trend")

        # MACD
        macd_hist = features.get('macd_hist', 0)
        if not pd.isna(macd_hist):
            macd_score = 1 if macd_hist > 0 else 0
            scores.append(macd_score)
            if macd_hist > 0:
                signals.append("MACD bullish")

        # Golden/Death cross
        if features.get('golden_cross', 0) == 1:
            scores.append(1.0)
            signals.append("golden cross")
        elif features.get('death_cross', 0) == 1:
            scores.append(0.0)
            signals.append("death cross")

        # Average all trend scores
        final_score = np.mean(scores) if scores else 0.5

        explanation = "Trend: " + (", ".join(signals) if signals else "neutral")

        return final_score, explanation

    def calculate_value_score(self, fundamentals: Dict) -> Tuple[float, str]:
        """
        Calculate value score based on fundamentals

        Args:
            fundamentals: Dictionary of fundamental data

        Returns:
            Tuple of (score, explanation)
        """
        scores = []
        signals = []

        # P/E ratio (lower is better, to a point)
        pe = fundamentals.get('trailingPE', 0)
        if pe and not pd.isna(pe) and pe > 0:
            # Normalize P/E (assume 0-50 range, with 15 being ideal)
            if pe < 15:
                pe_score = 1.0
            elif pe < 25:
                pe_score = 0.7
            elif pe < 35:
                pe_score = 0.4
            else:
                pe_score = 0.2
            scores.append(pe_score)
            signals.append(f"P/E {pe:.1f}")

        # P/B ratio
        pb = fundamentals.get('priceToBook', 0)
        if pb and not pd.isna(pb) and pb > 0:
            if pb < 1:
                pb_score = 1.0
            elif pb < 3:
                pb_score = 0.7
            elif pb < 5:
                pb_score = 0.4
            else:
                pb_score = 0.2
            scores.append(pb_score)

        # Dividend yield (higher is better for conservative)
        div_yield = fundamentals.get('dividendYield', 0)
        if div_yield and not pd.isna(div_yield):
            div_score = min(1.0, div_yield / 0.05)  # 5% = perfect
            if self.risk_profile == 'conservative':
                scores.append(div_score * 1.5)  # Weight dividends more
            else:
                scores.append(div_score)
            if div_yield > 0.02:
                signals.append(f"{div_yield*100:.1f}% yield")

        # Default score if no data
        if not scores:
            return 0.5, "Value: insufficient data"

        final_score = np.mean(scores)
        explanation = "Value: " + (", ".join(signals) if signals else "fair")

        return final_score, explanation

    def calculate_sentiment_score(self, sentiment_data: Dict) -> Tuple[float, str]:
        """
        Calculate sentiment score from news

        Args:
            sentiment_data: Dictionary with sentiment analysis

        Returns:
            Tuple of (score, explanation)
        """
        if not sentiment_data or sentiment_data.get('total_articles', 0) == 0:
            return 0.5, "Sentiment: no news data"

        avg_score = sentiment_data.get('average_score', 0)
        overall = sentiment_data.get('overall_sentiment', 'neutral')
        total = sentiment_data.get('total_articles', 0)
        positive_count = sentiment_data.get('positive_count', 0)
        negative_count = sentiment_data.get('negative_count', 0)

        # Normalize score from -1,1 to 0,1
        normalized = (avg_score + 1) / 2

        explanation = f"Sentiment: {overall} ({positive_count}+/{negative_count}- from {total} articles)"

        return normalized, explanation

    def calculate_overall_score(
        self,
        features: Dict,
        fundamentals: Dict,
        sentiment_data: Dict,
        use_sentiment: bool = True
    ) -> Dict:
        """
        Calculate overall score and generate recommendation

        Args:
            features: Technical features
            fundamentals: Fundamental data
            sentiment_data: Sentiment analysis
            use_sentiment: Whether to include sentiment in score

        Returns:
            Dictionary with scores and recommendation
        """
        # Calculate individual component scores
        momentum_score, momentum_exp = self.calculate_momentum_score(features)
        volatility_score, volatility_exp = self.calculate_volatility_score(features)
        trend_score, trend_exp = self.calculate_trend_score(features)
        value_score, value_exp = self.calculate_value_score(fundamentals)
        sentiment_score, sentiment_exp = self.calculate_sentiment_score(sentiment_data)

        # Calculate weighted overall score
        if use_sentiment:
            overall_score = (
                self.weights['momentum'] * momentum_score +
                self.weights['volatility'] * volatility_score +
                self.weights['trend'] * trend_score +
                self.weights['value'] * value_score +
                self.weights['sentiment'] * sentiment_score
            )
        else:
            # Redistribute sentiment weight to other factors
            total_weight = sum([
                self.weights['momentum'],
                self.weights['volatility'],
                self.weights['trend'],
                self.weights['value']
            ])
            overall_score = (
                (self.weights['momentum'] / total_weight) * momentum_score +
                (self.weights['volatility'] / total_weight) * volatility_score +
                (self.weights['trend'] / total_weight) * trend_score +
                (self.weights['value'] / total_weight) * value_score
            )

        # Determine recommendation
        if overall_score >= self.thresholds['strong_buy']:
            action = 'STRONG BUY'
            confidence = 'high'
        elif overall_score >= self.thresholds['buy']:
            action = 'BUY'
            confidence = 'medium'
        elif overall_score >= self.thresholds['hold']:
            action = 'HOLD'
            confidence = 'medium'
        elif overall_score >= self.thresholds['sell']:
            action = 'WATCH'
            confidence = 'low'
        else:
            action = 'SELL'
            confidence = 'medium'

        return {
            'overall_score': overall_score,
            'action': action,
            'confidence': confidence,
            'component_scores': {
                'momentum': momentum_score,
                'volatility': volatility_score,
                'trend': trend_score,
                'value': value_score,
                'sentiment': sentiment_score if use_sentiment else None
            },
            'explanations': {
                'momentum': momentum_exp,
                'volatility': volatility_exp,
                'trend': trend_exp,
                'value': value_exp,
                'sentiment': sentiment_exp if use_sentiment else None
            }
        }


def create_stock_scorer(risk_profile: str = "balanced") -> StockScorer:
    """
    Create and return a StockScorer instance

    Args:
        risk_profile: Risk profile

    Returns:
        StockScorer instance
    """
    return StockScorer(risk_profile=risk_profile)
