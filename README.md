# 📈 Stock Market Forecasting & Recommendation System

An end-to-end Python-based system for analyzing stocks from major indices (FTSE, NASDAQ, S&P) and generating actionable BUY/WATCH/SELL recommendations using technical analysis, fundamentals, and news sentiment.

## ✨ Features

- **Multi-Market Analysis**: Analyze stocks from FTSE 100/250, NASDAQ-100, and S&P 500
- **Active Investing Strategy**: Identifies BUY and SELL opportunities for capital growth
- **Risk Profiles**: Configurable strategies for Conservative, Balanced, or Aggressive investors
- **Comprehensive Scoring**: Combines:
  - Technical indicators (moving averages, RSI, MACD, Bollinger Bands)
  - Momentum and trend analysis
  - Value metrics (P/E, P/B, dividend yield)
  - News sentiment analysis (optional)
- **Natural Language Explanations**: Each recommendation includes a clear explanation
- **Interactive Dashboard**: Beautiful Streamlit interface with:
  - Overview and analytics
  - Stock recommendations with explanations
  - Individual stock detail views
  - Backtesting framework
- **Portfolio Tracking**: Support for existing portfolio evaluation
- **Backtesting**: Validate strategies with historical data

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) NewsAPI key for sentiment analysis

### Installation

1. **Clone the repository**
```bash
cd /path/to/Project---2
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

Note: If you encounter issues installing `ta-lib` on Windows, you can skip it - the system will use `pandas-ta` as a fallback.

4. **Set up environment variables**
```bash
cp .env.example .env
```

Edit `.env` and add your NewsAPI key (optional but recommended):
```bash
NEWSAPI_KEY=your_api_key_here
```

**Get a free NewsAPI key:**
- Visit https://newsapi.org/register
- Sign up for a free account (100 requests/day)
- Copy your API key to `.env`

### Running the Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## 📖 How to Use

### 1. Configure Your Analysis

In the sidebar:
- **Select Markets**: Choose FTSE 100/250, NASDAQ-100, and/or S&P 500
- **Risk Profile**: Select Conservative, Balanced, or Aggressive
- **Analysis Scope**: Choose how many stocks to analyze per index (10-100)
- **Sentiment Analysis**: Toggle news sentiment (requires NewsAPI key)
- **Existing Portfolio**: (Optional) Enter tickers you currently hold

### 2. Run Analysis

Click the "🚀 Run Analysis" button. The system will:
- Fetch constituent lists for selected indices
- Download price data and fundamentals
- Calculate technical indicators
- Fetch and analyze news (if enabled)
- Score and rank all stocks
- Generate recommendations

**Note**: First run takes longer (5-10 minutes) as data is downloaded. Subsequent runs use cached data and are faster.

### 3. View Recommendations

Navigate through the tabs:

**📊 Overview**
- Summary statistics
- Score distribution
- Sector breakdown

**💡 Recommendations**
- **BUY**: Top-scored stocks to purchase
- **WATCH**: Moderate opportunities to monitor
- **SELL**: Poorly performing holdings to consider exiting

**🔍 Stock Details**
- Select any analyzed stock
- View detailed metrics and explanations
- See price charts with technical indicators
- Review recent news sentiment

**📈 Backtest**
- Test the strategy on historical data
- See performance metrics (return, Sharpe ratio, drawdown)
- Compare against benchmark index

## 🏗️ Project Structure

```
stock-forecasting-system/
│
├── data/
│   ├── raw/              # Cached price data and news
│   ├── processed/        # Processed features
│   └── constituents/     # Index constituent lists
│
├── models/               # Saved models (if using ML)
│
├── src/
│   ├── config.py         # Configuration settings
│   ├── data_loader.py    # Market data fetching (yfinance)
│   ├── constituents.py   # Index constituent management
│   ├── news_client.py    # News API integration
│   ├── sentiment.py      # Sentiment analysis (TextBlob/FinBERT)
│   ├── features.py       # Feature engineering & indicators
│   ├── scoring.py        # Stock scoring model
│   ├── recommender.py    # Recommendation engine
│   ├── backtest.py       # Backtesting framework
│   ├── explainer.py      # Natural language explanations
│   └── utils.py          # Helper functions
│
├── dashboard/
│   └── app.py            # Streamlit dashboard
│
├── .env                  # Environment variables (API keys)
├── .env.example          # Example environment file
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## ⚙️ Configuration

### Risk Profiles

The system supports three risk profiles with different scoring weights:

**Conservative**
- Lower volatility preferred
- Higher weight on value metrics
- Emphasizes stability and dividends

**Balanced** (Default)
- Equal consideration of growth and stability
- Balanced weights across all factors

**Aggressive**
- Momentum and trend focused
- Less concerned with volatility
- Maximizes growth potential

### Scoring Weights

Edit `src/config.py` to customize scoring weights:

```python
SCORING_WEIGHTS = {
    "balanced": {
        "momentum": 0.25,
        "volatility": 0.15,
        "trend": 0.25,
        "value": 0.20,
        "sentiment": 0.15
    }
}
```

### Recommendation Thresholds

Adjust buy/sell thresholds in `src/config.py`:

```python
RECOMMENDATION_THRESHOLDS = {
    "balanced": {
        "strong_buy": 0.70,  # Score >= 0.70
        "buy": 0.55,         # Score >= 0.55
        "hold": 0.45,        # Score >= 0.45
        "sell": 0.30         # Score < 0.30
    }
}
```

## 🔧 Advanced Usage

### Using the Python API

You can use the system programmatically:

```python
from src.recommender import create_recommendation_engine
from src.config import INDICES

# Create engine
engine = create_recommendation_engine(
    risk_profile="balanced",
    use_sentiment=True
)

# Analyze stocks
analysis_df = engine.analyze_portfolio(
    tickers=[],  # Empty = fetch from indices
    indices=['SP500', 'NASDAQ100'],
    top_n=50
)

# Get recommendations
recommendations = engine.get_recommendations(
    analysis_df,
    num_buy=20,
    num_watch=10
)

# View buy recommendations
print(recommendations['buy'][['ticker', 'name', 'score', 'action']])
```

### Running Backtests

```python
from src.backtest import create_backtester

# Create backtester
backtester = create_backtester(
    risk_profile="balanced",
    portfolio_size=20
)

# Run backtest
results = backtester.run_backtest(
    tickers=list_of_tickers,
    start_date="2022-01-01",
    end_date="2024-01-01"
)

# View results
print(f"Total Return: {results['total_return_pct']:.2f}%")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown_pct']:.2f}%")
```

## 📊 Technical Indicators

The system calculates:

**Momentum**
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Stochastic Oscillator

**Trend**
- Simple Moving Averages (20, 50, 200 day)
- Exponential Moving Averages
- ADX (Average Directional Index)
- Bollinger Bands

**Volume**
- Average volume
- On-Balance Volume (OBV)
- Volume-Weighted Average Price (VWAP)

**Returns**
- Multiple timeframes: 1D, 1W, 1M, 3M, 6M, 1Y
- Relative performance vs index

## 📰 News & Sentiment

When enabled (with NewsAPI key):
- Fetches recent articles for each stock
- Analyzes sentiment using TextBlob
- Aggregates sentiment across articles
- Weights recent news more heavily
- Includes sentiment in overall score

## ⚠️ Important Disclaimers

**This tool is for research and educational purposes only.**

- NOT financial advice
- Past performance ≠ future results
- All investments carry risk of loss
- Recommendations are automated and may not suit your situation
- Always do your own research
- Consult a qualified financial advisor before investing

The creators assume NO liability for any losses incurred from using this system.

## 🐛 Troubleshooting

### "No module named 'talib'"
TA-Lib can be difficult to install on some systems. The system will automatically fall back to `pandas-ta` which works on all platforms.

### "NewsAPI error: apiKeyInvalid"
Check that your NewsAPI key is correctly set in `.env`. You can run the system without news sentiment - just uncheck the option in the dashboard.

### "No data returned for ticker XXX"
Some tickers may not have data available in yfinance. This is normal - the system will skip them and continue with others.

### Analysis is slow
- First run is always slower (downloading data)
- Reduce "Top N stocks per index" in sidebar
- Disable sentiment analysis for faster processing
- Data is cached for 24 hours by default

### "SSL Certificate Error"
If you encounter SSL errors with yfinance or news APIs:
```bash
pip install --upgrade certifi
```

## 📝 Data Sources

- **Market Data**: Yahoo Finance via `yfinance`
- **Index Constituents**: Wikipedia (via `pandas.read_html`)
- **News**: NewsAPI.org (free tier: 100 requests/day)
- **Sentiment**: TextBlob (default) or FinBERT (optional)

## 🔄 Data Refresh

- **Price Data**: Refreshed on every dashboard refresh
- **News & Sentiment**: Cached for 24 hours
- **Index Constituents**: Cached for 7 days
- **Stock Fundamentals**: Cached for 24 hours

## 🛠️ Development

### Running Tests
```bash
pytest tests/
```

### Code Structure
- **Modular design**: Each component is independent
- **Caching**: Aggressive caching for performance
- **Error handling**: Graceful failures with logging
- **Parallel processing**: Multi-threaded data fetching

## 📚 Further Reading

- [Investopedia - Technical Analysis](https://www.investopedia.com/terms/t/technicalanalysis.asp)
- [Investopedia - Fundamental Analysis](https://www.investopedia.com/terms/f/fundamentalanalysis.asp)
- [yfinance Documentation](https://github.com/ranaroussi/yfinance)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 📄 License

This project is provided as-is for educational purposes.

## 🤝 Contributing

This is a personal project, but suggestions and improvements are welcome!

---

**Happy Investing! Remember: This is a tool to assist your research, not a replacement for due diligence.** 📈
