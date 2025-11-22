"""
Streamlit Dashboard for Stock Market Forecasting System
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import logging

from src.config import (
    DISCLAIMER,
    INDICES,
    TOP_N_PER_INDEX,
    SCORING_WEIGHTS,
    NEWSAPI_KEY
)
from src.recommender import create_recommendation_engine
from src.backtest import create_backtester
from src.data_loader import create_data_loader
from src.explainer import create_explainer
from src.utils import setup_logger, format_large_number

# Configure logging
logger = setup_logger(__name__, level=logging.INFO)

# Page config
st.set_page_config(
    page_title="Stock Forecasting System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 500;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .buy-tag {
        background-color: #28a745;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: 600;
    }
    .sell-tag {
        background-color: #dc3545;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: 600;
    }
    .hold-tag {
        background-color: #ffc107;
        color: black;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'analysis_df' not in st.session_state:
        st.session_state.analysis_df = pd.DataFrame()
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = {}
    if 'backtest_results' not in st.session_state:
        st.session_state.backtest_results = None


def render_sidebar():
    """Render sidebar controls"""
    st.sidebar.markdown("## ⚙️ Configuration")

    # Market selection
    st.sidebar.markdown("### Markets to Analyze")
    indices = []
    if st.sidebar.checkbox("S&P 500", value=True, key="cb_sp500"):
        indices.append('SP500')
    if st.sidebar.checkbox("NASDAQ-100", value=True, key="cb_nasdaq100"):
        indices.append('NASDAQ100')

    ftse_option = st.sidebar.selectbox(
        "FTSE Index",
        ["None", "FTSE 100", "FTSE 250", "Both"],
        index=1
    )
    if ftse_option == "FTSE 100" or ftse_option == "Both":
        indices.append('FTSE100')
    if ftse_option == "FTSE 250" or ftse_option == "Both":
        indices.append('FTSE250')

    # Risk profile
    st.sidebar.markdown("### Risk Profile")
    risk_profile = st.sidebar.radio(
        "Select risk profile:",
        ["Conservative", "Balanced", "Aggressive"],
        index=1
    ).lower()

    # Top N stocks
    st.sidebar.markdown("### Analysis Scope")
    top_n = st.sidebar.slider(
        "Top N stocks per index:",
        min_value=10,
        max_value=100,
        value=TOP_N_PER_INDEX,
        step=10
    )

    # Sentiment analysis
    use_sentiment = st.sidebar.checkbox(
        "Use news sentiment analysis",
        value=bool(NEWSAPI_KEY),
        disabled=not bool(NEWSAPI_KEY)
    )

    if not NEWSAPI_KEY:
        st.sidebar.warning("⚠️ No NewsAPI key configured. Sentiment analysis disabled.")

    # Existing portfolio
    st.sidebar.markdown("### Analyze Specific Stocks (Optional)")
    st.sidebar.caption("Leave empty to analyze all stocks from selected indices, or enter specific tickers to analyze only those.")
    portfolio_input = st.sidebar.text_area(
        "Enter tickers (one per line):",
        placeholder="AAPL\nMSFT\nGOOGL",
        height=100,
        key="portfolio_input"
    )
    existing_portfolio = [t.strip().upper() for t in portfolio_input.split('\n') if t.strip()]

    return {
        'indices': indices,
        'risk_profile': risk_profile,
        'top_n': top_n,
        'use_sentiment': use_sentiment,
        'existing_portfolio': existing_portfolio
    }


def render_overview(analysis_df: pd.DataFrame, config: dict):
    """Render overview section"""
    st.markdown('<div class="sub-header">📊 Analysis Overview</div>', unsafe_allow_html=True)

    if analysis_df.empty:
        st.info("Run analysis to see results")
        return

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Stocks Analyzed", len(analysis_df))

    with col2:
        buy_count = len(analysis_df[analysis_df['action'].isin(['STRONG BUY', 'BUY'])])
        st.metric("Buy Opportunities", buy_count)

    with col3:
        avg_score = analysis_df['score'].mean()
        st.metric("Average Score", f"{avg_score:.2f}")

    with col4:
        indices_str = ", ".join([INDICES[i]['name'] for i in config['indices']])
        st.metric("Markets", len(config['indices']))
        st.caption(indices_str)

    # Score distribution
    st.markdown("#### Score Distribution")
    fig = px.histogram(
        analysis_df,
        x='score',
        nbins=20,
        title="Distribution of Stock Scores",
        labels={'score': 'Score', 'count': 'Number of Stocks'}
    )
    st.plotly_chart(fig, use_container_width=True)

    # Sector distribution
    if 'sector' in analysis_df.columns:
        st.markdown("#### Top Sectors")
        sector_counts = analysis_df['sector'].value_counts().head(10)
        fig = px.bar(
            x=sector_counts.index,
            y=sector_counts.values,
            labels={'x': 'Sector', 'y': 'Count'},
            title="Stock Count by Sector"
        )
        st.plotly_chart(fig, use_container_width=True)


def render_recommendations(recommendations: dict, config: dict):
    """Render recommendations section"""
    st.markdown('<div class="sub-header">💡 Recommendations</div>', unsafe_allow_html=True)

    if not recommendations:
        st.info("Run analysis to see recommendations")
        return

    # Tabs for different recommendation types
    tab1, tab2, tab3 = st.tabs(["🟢 BUY", "🟡 WATCH", "🔴 SELL/Review"])

    with tab1:
        render_recommendation_table(recommendations.get('buy', pd.DataFrame()), "BUY")

    with tab2:
        render_recommendation_table(recommendations.get('watch', pd.DataFrame()), "WATCH")

    with tab3:
        render_recommendation_table(recommendations.get('sell', pd.DataFrame()), "SELL")


def render_recommendation_table(df: pd.DataFrame, rec_type: str):
    """Render recommendation table"""
    if df.empty:
        st.info(f"No {rec_type} recommendations")
        return

    st.markdown(f"### Top {rec_type} Recommendations ({len(df)} stocks)")

    # Display table
    display_df = df[[
        'ticker', 'name', 'sector', 'score', 'action',
        'current_price', 'recent_return_1m', 'sentiment'
    ]].copy()

    # Format columns
    display_df['score'] = display_df['score'].apply(lambda x: f"{x:.3f}")
    display_df['current_price'] = display_df['current_price'].apply(lambda x: f"${x:.2f}")
    display_df['recent_return_1m'] = display_df['recent_return_1m'].apply(lambda x: f"{x*100:+.2f}%")

    display_df.columns = ['Ticker', 'Name', 'Sector', 'Score', 'Action', 'Price', '1M Return', 'Sentiment']

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Expandable explanations
    with st.expander("📄 View Detailed Explanations"):
        for idx, row in df.iterrows():
            st.markdown(f"**{row['ticker']} - {row['name']}**")
            st.markdown(row['explanation'])
            st.markdown("---")


def render_stock_detail():
    """Render individual stock detail view"""
    st.markdown('<div class="sub-header">🔍 Stock Detail View</div>', unsafe_allow_html=True)

    if st.session_state.analysis_df.empty:
        st.info("Run analysis first to view stock details")
        return

    # Select stock
    tickers = sorted(st.session_state.analysis_df['ticker'].tolist())
    selected_ticker = st.selectbox("Select a stock:", tickers)

    if not selected_ticker:
        return

    # Get stock data
    stock_data = st.session_state.analysis_df[
        st.session_state.analysis_df['ticker'] == selected_ticker
    ].iloc[0].to_dict()

    # Display stock info
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Current Price", f"${stock_data['current_price']:.2f}")
        st.metric("Score", f"{stock_data['score']:.3f}")

    with col2:
        st.metric("Action", stock_data['action'])
        market_cap = stock_data.get('market_cap', 0)
        st.metric("Market Cap", format_large_number(market_cap))

    with col3:
        st.metric("Sector", stock_data['sector'])
        st.metric("Sentiment", stock_data.get('sentiment', 'N/A'))

    # Show explanation
    st.markdown("#### Why this recommendation?")
    st.markdown(stock_data.get('explanation', 'No explanation available'))

    # Load and plot price history
    data_loader = create_data_loader()
    price_data = data_loader.get_stock_data(selected_ticker, period="1y")

    if price_data is not None and not price_data.empty:
        st.markdown("#### Price History (1 Year)")

        # Check if required columns exist
        required_cols = ['Open', 'High', 'Low', 'Close']
        if all(col in price_data.columns for col in required_cols):
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=price_data.index,
                open=price_data['Open'],
                high=price_data['High'],
                low=price_data['Low'],
                close=price_data['Close'],
                name='Price'
            ))

            # Add moving averages
            fig.add_trace(go.Scatter(
                x=price_data.index,
                y=price_data['Close'].rolling(20).mean(),
                name='20-day MA',
                line=dict(color='orange', width=1)
            ))

            fig.add_trace(go.Scatter(
                x=price_data.index,
                y=price_data['Close'].rolling(50).mean(),
                name='50-day MA',
                line=dict(color='blue', width=1)
            ))

            fig.update_layout(
                title=f"{selected_ticker} Price Chart",
                yaxis_title="Price ($)",
                xaxis_title="Date",
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"Price data columns missing. Found: {list(price_data.columns)}")
            logger.error(f"Missing OHLC columns for {selected_ticker}. Columns: {list(price_data.columns)}")


def render_backtest(config: dict):
    """Render backtest section"""
    st.markdown('<div class="sub-header">📈 Backtest Results</div>', unsafe_allow_html=True)

    if st.button("Run Backtest", type="primary"):
        with st.spinner("Running backtest... This may take several minutes..."):
            try:
                backtester = create_backtester(
                    risk_profile=config['risk_profile'],
                    portfolio_size=20
                )

                # Get tickers from analysis
                if not st.session_state.analysis_df.empty:
                    tickers = st.session_state.analysis_df['ticker'].tolist()
                else:
                    st.error("Run analysis first to get stock universe")
                    return

                # Run backtest
                results = backtester.run_backtest(tickers)
                st.session_state.backtest_results = results

                st.success("Backtest complete!")

            except Exception as e:
                st.error(f"Backtest error: {str(e)}")
                logger.error(f"Backtest error: {e}", exc_info=True)
                return

    # Display results
    if st.session_state.backtest_results:
        results = st.session_state.backtest_results

        # Metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Return",
                f"{results['total_return_pct']:.2f}%",
                delta=f"{results['outperformance_pct']:+.2f}% vs benchmark"
            )

        with col2:
            st.metric("CAGR", f"{results['cagr']*100:.2f}%")

        with col3:
            st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")

        with col4:
            st.metric("Max Drawdown", f"{results['max_drawdown_pct']:.2f}%")

        # Portfolio value chart
        st.markdown("#### Portfolio Value Over Time")

        portfolio_history = results['portfolio_history']

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=portfolio_history.index,
            y=portfolio_history['total_value'],
            name='Portfolio Value',
            line=dict(color='blue', width=2)
        ))

        # Add initial capital line
        fig.add_hline(
            y=results['initial_capital'],
            line_dash="dash",
            line_color="gray",
            annotation_text="Initial Capital"
        )

        fig.update_layout(
            title="Portfolio Performance",
            yaxis_title="Portfolio Value ($)",
            xaxis_title="Date",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        # Returns distribution
        st.markdown("#### Returns Distribution")
        fig = px.histogram(
            portfolio_history['returns'].dropna(),
            nbins=50,
            title="Distribution of Daily Returns",
            labels={'value': 'Daily Return', 'count': 'Frequency'}
        )
        st.plotly_chart(fig, use_container_width=True)


def main():
    """Main application"""
    initialize_session_state()

    # Header
    st.markdown('<div class="main-header">📈 Stock Market Forecasting & Recommendation System</div>',
                unsafe_allow_html=True)

    # Disclaimer
    with st.expander("⚠️ Important Disclaimer - Please Read"):
        st.markdown(DISCLAIMER)

    # Sidebar
    config = render_sidebar()

    # Main actions
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
            if not config['indices']:
                st.error("Please select at least one index")
                return

            with st.spinner("Analyzing stocks... This may take a few minutes..."):
                try:
                    # Create engine
                    engine = create_recommendation_engine(
                        risk_profile=config['risk_profile'],
                        use_sentiment=config['use_sentiment'],
                        use_finbert=False  # Use TextBlob for speed
                    )

                    # Run analysis
                    # If user provided existing portfolio, only analyze those stocks
                    # Otherwise, analyze top N stocks from selected indices
                    tickers_to_analyze = config['existing_portfolio'] if config['existing_portfolio'] else []

                    analysis_df = engine.analyze_portfolio(
                        tickers=tickers_to_analyze,
                        indices=config['indices'],
                        top_n=config['top_n'],
                        show_progress=True
                    )

                    if analysis_df.empty:
                        st.error("No stocks analyzed successfully")
                        return

                    # Get recommendations
                    recommendations = engine.get_recommendations(
                        analysis_df,
                        num_buy=20,
                        num_watch=10,
                        existing_portfolio=config['existing_portfolio']
                    )

                    # Store in session state
                    st.session_state.analysis_df = analysis_df
                    st.session_state.recommendations = recommendations
                    st.session_state.analysis_complete = True

                    st.success(f"✅ Analysis complete! Analyzed {len(analysis_df)} stocks.")

                except Exception as e:
                    st.error(f"Analysis error: {str(e)}")
                    logger.error(f"Analysis error: {e}", exc_info=True)
                    return

    with col2:
        if st.session_state.analysis_complete:
            st.success(f"Last analysis: {len(st.session_state.analysis_df)} stocks | "
                      f"Profile: {config['risk_profile'].title()}")

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview",
        "💡 Recommendations",
        "🔍 Stock Details",
        "📈 Backtest"
    ])

    with tab1:
        render_overview(st.session_state.analysis_df, config)

    with tab2:
        render_recommendations(st.session_state.recommendations, config)

    with tab3:
        render_stock_detail()

    with tab4:
        render_backtest(config)

    # Footer
    st.markdown("---")
    st.markdown(
        "Built with Streamlit | Data: yfinance | News: NewsAPI | "
        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )


if __name__ == "__main__":
    main()
