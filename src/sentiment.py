"""
Sentiment analysis module for financial news
"""
import logging
from typing import List, Dict, Optional
import warnings

import pandas as pd
import numpy as np
from textblob import TextBlob

from .utils import setup_logger

warnings.filterwarnings('ignore')
logger = setup_logger(__name__)


class SentimentAnalyzer:
    """
    Analyzes sentiment of financial news articles
    """

    def __init__(self, use_finbert: bool = False):
        """
        Initialize SentimentAnalyzer

        Args:
            use_finbert: Whether to use FinBERT model (slower but more accurate)
        """
        self.use_finbert = use_finbert
        self.finbert_model = None
        self.finbert_tokenizer = None

        if use_finbert:
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                import torch

                logger.info("Loading FinBERT model...")
                model_name = "ProsusAI/finbert"
                self.finbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.finbert_model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.finbert_model.to(self.device)
                logger.info(f"FinBERT loaded on {self.device}")
            except Exception as e:
                logger.warning(f"Could not load FinBERT, falling back to TextBlob: {e}")
                self.use_finbert = False

        if not self.use_finbert:
            logger.info("Using TextBlob for sentiment analysis")

    def analyze_text_textblob(self, text: str) -> Dict:
        """
        Analyze sentiment using TextBlob

        Args:
            text: Text to analyze

        Returns:
            Dictionary with sentiment scores
        """
        if not text or pd.isna(text):
            return {
                'sentiment': 'neutral',
                'score': 0.0,
                'polarity': 0.0,
                'subjectivity': 0.0
            }

        try:
            blob = TextBlob(str(text))
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity

            # Classify sentiment
            if polarity > 0.1:
                sentiment = 'positive'
            elif polarity < -0.1:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'

            return {
                'sentiment': sentiment,
                'score': polarity,
                'polarity': polarity,
                'subjectivity': subjectivity
            }

        except Exception as e:
            logger.error(f"Error in TextBlob analysis: {e}")
            return {
                'sentiment': 'neutral',
                'score': 0.0,
                'polarity': 0.0,
                'subjectivity': 0.0
            }

    def analyze_text_finbert(self, text: str) -> Dict:
        """
        Analyze sentiment using FinBERT

        Args:
            text: Text to analyze

        Returns:
            Dictionary with sentiment scores
        """
        if not text or pd.isna(text):
            return {
                'sentiment': 'neutral',
                'score': 0.0,
                'confidence': 0.0
            }

        try:
            import torch

            # Truncate text if too long
            max_length = 512
            text = str(text)[:max_length]

            # Tokenize and predict
            inputs = self.finbert_tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(self.device)

            with torch.no_grad():
                outputs = self.finbert_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # FinBERT outputs: [positive, negative, neutral]
            probs = predictions[0].cpu().numpy()
            sentiment_map = {0: 'positive', 1: 'negative', 2: 'neutral'}
            sentiment_idx = np.argmax(probs)
            sentiment = sentiment_map[sentiment_idx]
            confidence = float(probs[sentiment_idx])

            # Calculate score: -1 (negative) to +1 (positive)
            score = float(probs[0] - probs[1])

            return {
                'sentiment': sentiment,
                'score': score,
                'confidence': confidence,
                'positive_prob': float(probs[0]),
                'negative_prob': float(probs[1]),
                'neutral_prob': float(probs[2])
            }

        except Exception as e:
            logger.error(f"Error in FinBERT analysis: {e}")
            return {
                'sentiment': 'neutral',
                'score': 0.0,
                'confidence': 0.0
            }

    def analyze_text(self, text: str) -> Dict:
        """
        Analyze sentiment using configured method

        Args:
            text: Text to analyze

        Returns:
            Dictionary with sentiment scores
        """
        if self.use_finbert:
            return self.analyze_text_finbert(text)
        else:
            return self.analyze_text_textblob(text)

    def analyze_article(self, article: Dict) -> Dict:
        """
        Analyze sentiment of a news article

        Args:
            article: Article dictionary with 'title', 'description', 'content'

        Returns:
            Article dictionary with sentiment added
        """
        # Combine title, description, and content
        text_parts = []
        if article.get('title'):
            text_parts.append(article['title'])
        if article.get('description'):
            text_parts.append(article['description'])
        if article.get('content'):
            # Take first 500 chars of content
            text_parts.append(article['content'][:500])

        combined_text = " ".join(text_parts)

        # Analyze sentiment
        sentiment_result = self.analyze_text(combined_text)

        # Add to article
        article_with_sentiment = article.copy()
        article_with_sentiment.update({
            'sentiment': sentiment_result['sentiment'],
            'sentiment_score': sentiment_result['score'],
            'analyzed_text_length': len(combined_text)
        })

        if self.use_finbert:
            article_with_sentiment['sentiment_confidence'] = sentiment_result.get('confidence', 0)
        else:
            article_with_sentiment['sentiment_polarity'] = sentiment_result.get('polarity', 0)
            article_with_sentiment['sentiment_subjectivity'] = sentiment_result.get('subjectivity', 0)

        return article_with_sentiment

    def analyze_articles(self, articles: List[Dict]) -> List[Dict]:
        """
        Analyze sentiment for multiple articles

        Args:
            articles: List of article dictionaries

        Returns:
            List of articles with sentiment added
        """
        analyzed = []
        for article in articles:
            analyzed.append(self.analyze_article(article))

        return analyzed

    def aggregate_sentiment(
        self,
        articles: List[Dict],
        weight_by_recency: bool = True
    ) -> Dict:
        """
        Aggregate sentiment across multiple articles

        Args:
            articles: List of articles with sentiment
            weight_by_recency: Weight recent articles more heavily

        Returns:
            Dictionary with aggregated sentiment
        """
        if not articles:
            return {
                'overall_sentiment': 'neutral',
                'average_score': 0.0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'total_articles': 0
            }

        # Count sentiments
        sentiments = [a.get('sentiment', 'neutral') for a in articles]
        positive_count = sentiments.count('positive')
        negative_count = sentiments.count('negative')
        neutral_count = sentiments.count('neutral')

        # Calculate weighted average score
        scores = []
        weights = []

        for i, article in enumerate(articles):
            score = article.get('sentiment_score', 0)
            scores.append(score)

            # Weight by recency (more recent = higher weight)
            if weight_by_recency:
                # Exponential decay: recent articles get weight 1, oldest get weight 0.5
                weight = 1.0 - (0.5 * i / len(articles))
            else:
                weight = 1.0

            weights.append(weight)

        scores = np.array(scores)
        weights = np.array(weights)
        weighted_avg = np.average(scores, weights=weights)

        # Determine overall sentiment
        if weighted_avg > 0.1:
            overall = 'positive'
        elif weighted_avg < -0.1:
            overall = 'negative'
        else:
            overall = 'neutral'

        return {
            'overall_sentiment': overall,
            'average_score': float(weighted_avg),
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'total_articles': len(articles),
            'sentiment_ratio': (positive_count - negative_count) / len(articles) if articles else 0
        }

    def get_sentiment_summary(self, ticker: str, articles: List[Dict]) -> Dict:
        """
        Get complete sentiment summary for a stock

        Args:
            ticker: Stock ticker
            articles: List of articles

        Returns:
            Dictionary with complete sentiment analysis
        """
        # Analyze all articles
        analyzed_articles = self.analyze_articles(articles)

        # Aggregate sentiment
        aggregated = self.aggregate_sentiment(analyzed_articles)

        # Add ticker and article samples
        summary = {
            'ticker': ticker,
            **aggregated,
            'recent_headlines': [
                {
                    'title': a.get('title', ''),
                    'sentiment': a.get('sentiment', 'neutral'),
                    'score': a.get('sentiment_score', 0),
                    'source': a.get('source', 'Unknown'),
                    'date': a.get('publishedAt', '')
                }
                for a in analyzed_articles[:5]  # Top 5 articles
            ]
        }

        return summary


def create_sentiment_analyzer(use_finbert: bool = False) -> SentimentAnalyzer:
    """
    Create and return a SentimentAnalyzer instance

    Args:
        use_finbert: Whether to use FinBERT model

    Returns:
        SentimentAnalyzer instance
    """
    return SentimentAnalyzer(use_finbert=use_finbert)
