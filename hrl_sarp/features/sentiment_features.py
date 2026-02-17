"""
File: sentiment_features.py
Module: features
Description: FinBERT-India inference pipeline for financial news sentiment scoring.
             Processes batch news articles and computes daily sentiment scores per
             sector and per stock, feeding the Macro and Micro agent state vectors.
Design Decisions:
    - FinBERT (pre-trained or fine-tuned on Indian corpus) for domain-specific accuracy.
    - Batch inference with GPU support for throughput.
    - Exponential decay weighting: recent articles have more influence.
    - Sector and stock-level aggregation for both agent hierarchies.
References:
    - Araci (2019): "FinBERT: Financial Sentiment Analysis with Pre-Trained Language Models"
    - Hugging Face Transformers: https://huggingface.co/ProsusAI/finbert
Author: HRL-SARP Framework
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


class SentimentFeatures:
    """
    FinBERT-based sentiment scoring for Indian financial news.

    Pipeline:
        1. Load pre-trained FinBERT (or fine-tuned FinBERT-India checkpoint)
        2. Tokenize news articles in batches
        3. Run inference → per-article sentiment {positive, negative, neutral}
        4. Aggregate to daily sector/stock sentiment scores

    Sentiment score ∈ [-1, 1]:
        - +1: Strongly positive sentiment
        -  0: Neutral
        - -1: Strongly negative sentiment

    Aggregation uses exponential time decay:
        weight_i = exp(-λ * hours_since_article)
        sector_sentiment = Σ(weight_i * score_i) / Σ(weight_i)

    This ensures recent news has more influence on the feature.
    """

    # ── FinBERT label mapping ────────────────────────────────────────
    LABEL_MAP = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        decay_lambda: float = 0.02,
    ) -> None:
        """
        Initialise FinBERT sentiment pipeline.

        Args:
            model_name: HuggingFace model name (default ProsusAI/finbert).
            checkpoint_path: Path to fine-tuned FinBERT-India checkpoint (overrides model_name).
            device: "cuda" or "cpu". Auto-detects if None.
            decay_lambda: Exponential decay rate for time-weighted aggregation.
        """
        self.decay_lambda = decay_lambda

        # ── Device selection ─────────────────────────────────────────
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Load model and tokenizer ─────────────────────────────────
        model_to_load = checkpoint_path or model_name
        logger.info("Loading FinBERT from: %s on %s", model_to_load, self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_to_load)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_to_load)
        self.model.to(self.device)
        self.model.eval()  # Set to inference mode

        # ── Get label order from model config ────────────────────────
        if hasattr(self.model.config, "id2label"):
            self.id2label = self.model.config.id2label
        else:
            self.id2label = {0: "positive", 1: "negative", 2: "neutral"}

        logger.info("FinBERT loaded successfully | labels=%s", self.id2label)

    # ══════════════════════════════════════════════════════════════════
    # SINGLE ARTICLE SCORING
    # ══════════════════════════════════════════════════════════════════
    @torch.no_grad()
    def score_text(self, text: str) -> Dict[str, float]:
        """
        Score a single text for sentiment.

        Args:
            text: Raw news article text.
        Returns:
            Dict with keys {positive, negative, neutral, sentiment_score}.
            sentiment_score = positive_prob - negative_prob ∈ [-1, 1].
        """
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512, padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

        result = {}
        for idx, prob in enumerate(probs):
            label = self.id2label.get(idx, f"label_{idx}").lower()
            result[label] = float(prob)

        # Composite sentiment score: positive probability - negative probability
        result["sentiment_score"] = result.get("positive", 0) - result.get("negative", 0)
        return result

    # ══════════════════════════════════════════════════════════════════
    # BATCH SCORING
    # ══════════════════════════════════════════════════════════════════
    @torch.no_grad()
    def score_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, float]]:
        """
        Score multiple texts in batches for efficiency.

        Uses batch inference with padding to maximise GPU throughput.

        Args:
            texts: List of news article texts.
            batch_size: Number of texts per batch.
        Returns:
            List of sentiment dictionaries (same format as score_text).
        """
        all_results: List[Dict[str, float]] = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            inputs = self.tokenizer(
                batch_texts, return_tensors="pt", truncation=True,
                max_length=512, padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()

            for prob_row in probs:
                result = {}
                for idx, prob in enumerate(prob_row):
                    label = self.id2label.get(idx, f"label_{idx}").lower()
                    result[label] = float(prob)
                result["sentiment_score"] = result.get("positive", 0) - result.get("negative", 0)
                all_results.append(result)

            logger.debug("Scored batch %d–%d of %d", i, min(i + batch_size, len(texts)), len(texts))

        return all_results

    # ══════════════════════════════════════════════════════════════════
    # TIME-WEIGHTED AGGREGATION
    # ══════════════════════════════════════════════════════════════════
    def aggregate_sentiment(
        self,
        scores: List[float],
        timestamps: List[datetime],
        reference_time: Optional[datetime] = None,
    ) -> float:
        """
        Compute time-weighted aggregate sentiment score.

        Uses exponential decay so recent articles have more influence:
            weight_i = exp(-λ * hours_since_article_i)
            aggregate = Σ(weight * score) / Σ(weight)

        This is important because:
            - A negative news article from 7 days ago has less impact than one from today
            - Markets adjust to information within 1-3 days

        Args:
            scores: List of sentiment scores ∈ [-1, 1].
            timestamps: Corresponding publication timestamps.
            reference_time: Current time for decay computation. Defaults to now.
        Returns:
            Weighted aggregate sentiment ∈ [-1, 1].
        """
        if not scores:
            return 0.0

        ref = reference_time or datetime.now()
        weights = []
        for ts in timestamps:
            hours_diff = max(0, (ref - ts).total_seconds() / 3600)
            weight = np.exp(-self.decay_lambda * hours_diff)
            weights.append(weight)

        weights_arr = np.array(weights)
        scores_arr = np.array(scores)
        total_weight = weights_arr.sum()

        if total_weight == 0:
            return 0.0

        return float(np.dot(weights_arr, scores_arr) / total_weight)

    # ══════════════════════════════════════════════════════════════════
    # SECTOR-LEVEL SENTIMENT
    # ══════════════════════════════════════════════════════════════════
    def compute_sector_sentiment(
        self,
        articles_df: pd.DataFrame,
        reference_time: Optional[datetime] = None,
    ) -> Dict[str, float]:
        """
        Compute daily sentiment scores per sector from news articles.

        Args:
            articles_df: News articles DataFrame with columns
                         [title, content, sectors, published_at].
            reference_time: Time for decay weighting.
        Returns:
            Dict mapping sector_name → sentiment_score ∈ [-1, 1].
        """
        if articles_df.empty:
            return {}

        # ── Score all articles ───────────────────────────────────────
        texts = (articles_df["title"] + " " + articles_df["content"]).tolist()
        sentiment_results = self.score_batch(texts)

        # ── Aggregate per sector ─────────────────────────────────────
        sector_scores: Dict[str, List[Tuple[float, datetime]]] = {}

        for idx, row in articles_df.iterrows():
            score = sentiment_results[articles_df.index.get_loc(idx) if isinstance(idx, str) else idx]["sentiment_score"]
            pub_time = row.get("published_at", datetime.now())
            if isinstance(pub_time, str):
                pub_time = datetime.fromisoformat(pub_time)

            sectors = row.get("sectors", [])
            if isinstance(sectors, str):
                sectors = [sectors]

            for sector in sectors:
                if sector not in sector_scores:
                    sector_scores[sector] = []
                sector_scores[sector].append((score, pub_time))

        # ── Time-weighted aggregation per sector ─────────────────────
        result: Dict[str, float] = {}
        for sector, pairs in sector_scores.items():
            scores = [p[0] for p in pairs]
            times = [p[1] for p in pairs]
            result[sector] = self.aggregate_sentiment(scores, times, reference_time)

        logger.info("Sector sentiment computed: %s", {k: f"{v:.3f}" for k, v in result.items()})
        return result

    # ══════════════════════════════════════════════════════════════════
    # STOCK-LEVEL SENTIMENT
    # ══════════════════════════════════════════════════════════════════
    def compute_stock_sentiment(
        self,
        articles_df: pd.DataFrame,
        reference_time: Optional[datetime] = None,
    ) -> Dict[str, float]:
        """
        Compute daily sentiment per stock from news articles.

        Same as sector sentiment but uses stock symbol tagging.

        Args:
            articles_df: News DataFrame with "symbols" column.
        Returns:
            Dict mapping symbol → sentiment_score ∈ [-1, 1].
        """
        if articles_df.empty:
            return {}

        texts = (articles_df["title"] + " " + articles_df["content"]).tolist()
        sentiment_results = self.score_batch(texts)

        stock_scores: Dict[str, List[Tuple[float, datetime]]] = {}

        for i, (_, row) in enumerate(articles_df.iterrows()):
            score = sentiment_results[i]["sentiment_score"]
            pub_time = row.get("published_at", datetime.now())
            if isinstance(pub_time, str):
                pub_time = datetime.fromisoformat(pub_time)

            symbols = row.get("symbols", [])
            if isinstance(symbols, str):
                symbols = [symbols]

            for symbol in symbols:
                if symbol not in stock_scores:
                    stock_scores[symbol] = []
                stock_scores[symbol].append((score, pub_time))

        result: Dict[str, float] = {}
        for symbol, pairs in stock_scores.items():
            scores = [p[0] for p in pairs]
            times = [p[1] for p in pairs]
            result[symbol] = self.aggregate_sentiment(scores, times, reference_time)

        return result
