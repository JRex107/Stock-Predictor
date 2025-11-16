"""
Explainer module for generating natural language explanations
"""
import logging
from typing import Dict, List
import pandas as pd

from .utils import setup_logger, format_large_number

logger = setup_logger(__name__)


class RecommendationExplainer:
    """
    Generates human-readable explanations for recommendations
    """

    def __init__(self):
        """Initialize RecommendationExplainer"""
        logger.info("RecommendationExplainer initialized")

    def explain_stock(self, stock_data: Dict) -> str:
        """
        Generate detailed explanation for a single stock

        Args:
            stock_data: Dictionary with stock analysis data

        Returns:
            Formatted explanation string
        """
        ticker = stock_data.get('ticker', 'Unknown')
        name = stock_data.get('name', ticker)
        action = stock_data.get('action', 'HOLD')
        score = stock_data.get('score', 0)
        sector = stock_data.get('sector', 'Unknown')

        # Start explanation
        explanation = f"### {name} ({ticker})\n\n"
        explanation += f"**Sector:** {sector}  \n"
        explanation += f"**Overall Score:** {score:.2f}/1.00  \n"
        explanation += f"**Recommendation:** {action}  \n\n"

        # Add main recommendation text
        if stock_data.get('explanation'):
            explanation += stock_data['explanation'] + "\n\n"

        # Add key metrics
        explanation += "**Key Metrics:**\n"

        price = stock_data.get('current_price')
        if price:
            explanation += f"- Current Price: ${price:.2f}\n"

        market_cap = stock_data.get('market_cap')
        if market_cap:
            explanation += f"- Market Cap: ${format_large_number(market_cap)}\n"

        pe = stock_data.get('pe_ratio')
        if pe and not pd.isna(pe):
            explanation += f"- P/E Ratio: {pe:.2f}\n"

        div_yield = stock_data.get('dividend_yield')
        if div_yield and not pd.isna(div_yield):
            explanation += f"- Dividend Yield: {div_yield*100:.2f}%\n"

        # Add performance
        return_1m = stock_data.get('recent_return_1m', 0)
        return_3m = stock_data.get('recent_return_3m', 0)
        explanation += f"\n**Recent Performance:**\n"
        explanation += f"- 1 Month: {return_1m*100:+.2f}%\n"
        explanation += f"- 3 Months: {return_3m*100:+.2f}%\n"

        rel_perf = stock_data.get('relative_performance_6m', 0)
        if rel_perf != 0:
            explanation += f"- vs Index (6M): {rel_perf*100:+.2f}%\n"

        return explanation

    def explain_portfolio(
        self,
        recommendations: Dict[str, pd.DataFrame]
    ) -> str:
        """
        Generate portfolio-level explanation

        Args:
            recommendations: Dictionary with buy/watch/sell DataFrames

        Returns:
            Formatted explanation string
        """
        buy_df = recommendations.get('buy', pd.DataFrame())
        watch_df = recommendations.get('watch', pd.DataFrame())
        sell_df = recommendations.get('sell', pd.DataFrame())

        explanation = "## Portfolio Recommendations Summary\n\n"

        # Buy recommendations
        if not buy_df.empty:
            explanation += f"### 🟢 BUY Recommendations ({len(buy_df)} stocks)\n\n"
            explanation += "These stocks show strong potential based on our analysis:\n\n"

            for idx, row in buy_df.head(5).iterrows():
                explanation += f"**{row['ticker']}** - {row['name']}\n"
                explanation += f"  Score: {row['score']:.2f} | "
                explanation += f"Action: {row['action']} | "
                explanation += f"Sector: {row['sector']}\n"

                # Brief reason
                explanations = row.get('component_scores', {})
                if explanations:
                    top_factor = max(explanations.items(), key=lambda x: x[1] if x[1] else 0)
                    explanation += f"  Key strength: {top_factor[0]} (score: {top_factor[1]:.2f})\n"

                explanation += "\n"

            if len(buy_df) > 5:
                explanation += f"*...and {len(buy_df) - 5} more*\n\n"

        # Watch recommendations
        if not watch_df.empty:
            explanation += f"### 🟡 WATCH List ({len(watch_df)} stocks)\n\n"
            explanation += "Monitor these stocks for potential opportunities:\n\n"

            for idx, row in watch_df.head(3).iterrows():
                explanation += f"- **{row['ticker']}** ({row['name']}) - Score: {row['score']:.2f}\n"

            if len(watch_df) > 3:
                explanation += f"- *...and {len(watch_df) - 3} more*\n"

            explanation += "\n"

        # Sell recommendations
        if not sell_df.empty:
            explanation += f"### 🔴 SELL Recommendations ({len(sell_df)} holdings)\n\n"
            explanation += "Consider reducing or exiting these positions:\n\n"

            for idx, row in sell_df.iterrows():
                explanation += f"- **{row['ticker']}** ({row['name']}) - Score: {row['score']:.2f}\n"
                explanation += f"  Reason: {row.get('explanation', 'Underperforming')}\n\n"

        return explanation

    def explain_sector_allocation(self, analysis_df: pd.DataFrame) -> str:
        """
        Explain sector allocation in recommendations

        Args:
            analysis_df: DataFrame with analysis results

        Returns:
            Formatted explanation string
        """
        if analysis_df.empty:
            return ""

        # Count by sector
        sector_counts = analysis_df['sector'].value_counts()

        explanation = "### Sector Distribution\n\n"

        for sector, count in sector_counts.head(10).items():
            pct = (count / len(analysis_df)) * 100
            explanation += f"- {sector}: {count} stocks ({pct:.1f}%)\n"

        return explanation


def create_explainer() -> RecommendationExplainer:
    """
    Create and return a RecommendationExplainer instance

    Returns:
        RecommendationExplainer instance
    """
    return RecommendationExplainer()
