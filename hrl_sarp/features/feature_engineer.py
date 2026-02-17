"""
File: feature_engineer.py
Module: features
Description: Master orchestrator that assembles all feature modules (technical, fundamental,
    sentiment, macro, sector) into ready-to-use macro_state and micro_state tensors.
Design Decisions: Single entry point avoids scattered feature logic. Timestamp gating
    prevents lookahead bias. Outputs are torch tensors for direct RL consumption.
References: Feature store pattern (Uber Michelangelo), point-in-time joins
Author: HRL-SARP Framework
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
import yaml

from features.technical_features import TechnicalFeatureComputer
from features.fundamental_features import FundamentalFeatureComputer
from features.sentiment_features import SentimentFeatureComputer
from features.macro_features import MacroFeatureComputer
from features.sector_features import SectorFeatureComputer

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════
# FEATURE ENGINEER — MASTER ASSEMBLER
# ══════════════════════════════════════════════════════════════════════


class FeatureEngineer:
    """Assembles all feature modules into macro_state and micro_state tensors."""

    def __init__(self, config_path: str = "config/data_config.yaml") -> None:
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.macro_feature_names: List[str] = self.config["features"]["macro_state"]
        self.stock_feature_names: List[str] = self.config["features"]["stock_features"]
        self.sector_mapping: Dict[str, List[str]] = self.config["sectors"]["mapping"]

        # All sectors in canonical order
        self.sector_names: List[str] = list(self.sector_mapping.keys())
        self.num_sectors: int = len(self.sector_names)

        # Reverse mapping: stock → sector
        self.stock_to_sector: Dict[str, str] = {}
        for sector, stocks in self.sector_mapping.items():
            for stock in stocks:
                self.stock_to_sector[stock] = sector

        # Feature sub-modules
        self.technical = TechnicalFeatureComputer()
        self.fundamental = FundamentalFeatureComputer()
        self.sentiment = SentimentFeatureComputer()
        self.macro = MacroFeatureComputer()
        self.sector = SectorFeatureComputer()

        # Cache for the current date's features (avoid recomputation)
        self._cache_date: Optional[str] = None
        self._cache: Dict[str, pd.DataFrame] = {}

        logger.info(
            "FeatureEngineer initialised | macro_dim=%d | stock_dim=%d | sectors=%d",
            len(self.macro_feature_names),
            len(self.stock_feature_names),
            self.num_sectors,
        )

    # ── Macro State Assembly ─────────────────────────────────────────

    def build_macro_state(
        self,
        date: str,
        ohlcv_data: Dict[str, pd.DataFrame],
        macro_data: pd.DataFrame,
        sentiment_scores: Optional[Dict[str, float]] = None,
        portfolio_drawdown: float = 0.0,
        prev_regime: int = 2,
    ) -> torch.Tensor:
        """Build 18D macro state vector for the Macro agent.

        Args:
            date: Current date (YYYY-MM-DD). Features use data strictly before this date.
            ohlcv_data: Dict of {symbol: OHLCV DataFrame} for index/sector data.
            macro_data: DataFrame with macro indicators indexed by date.
            sentiment_scores: Optional dict of {sector: sentiment_score}.
            portfolio_drawdown: Current drawdown from portfolio peak.
            prev_regime: Previous week's regime label (0=Bull, 1=Bear, 2=Sideways).

        Returns:
            Tensor of shape (18,) — the macro state vector.
        """
        date_dt = pd.Timestamp(date)

        # Compute macro features from raw data (VIX, FII, PCR, etc.)
        macro_feats = self.macro.compute(macro_data, as_of_date=date)

        # Compute sector-level breadth and momentum
        sector_feats = self.sector.compute(ohlcv_data, as_of_date=date)

        # Assemble into ordered vector matching config
        state_dict: Dict[str, float] = {}
        for feat_name in self.macro_feature_names:
            if feat_name in macro_feats:
                state_dict[feat_name] = float(macro_feats[feat_name])
            elif feat_name in sector_feats:
                state_dict[feat_name] = float(sector_feats[feat_name])
            elif feat_name == "portfolio_drawdown":
                state_dict[feat_name] = portfolio_drawdown
            elif feat_name == "regime_label_prev":
                state_dict[feat_name] = float(prev_regime)
            else:
                state_dict[feat_name] = 0.0
                logger.warning("Macro feature '%s' not found, defaulting to 0.0", feat_name)

        state_vector = np.array(
            [state_dict[name] for name in self.macro_feature_names], dtype=np.float32
        )

        # Replace NaN/Inf with 0
        state_vector = np.nan_to_num(state_vector, nan=0.0, posinf=1.0, neginf=-1.0)

        return torch.from_numpy(state_vector)

    # ── Micro State Assembly ─────────────────────────────────────────

    def build_micro_state(
        self,
        date: str,
        stock_universe: List[str],
        ohlcv_data: Dict[str, pd.DataFrame],
        fundamental_data: Dict[str, Dict[str, float]],
        sentiment_scores: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        """Build per-stock feature matrix for the Micro agent.

        Args:
            date: Current date (YYYY-MM-DD).
            stock_universe: List of stock symbols in the filtered universe.
            ohlcv_data: Dict of {symbol: OHLCV DataFrame}.
            fundamental_data: Dict of {symbol: {metric: value}}.
            sentiment_scores: Optional dict of {symbol: sentiment_score}.

        Returns:
            Tensor of shape (num_stocks, 22) — per-stock feature matrix.
        """
        feature_rows: List[np.ndarray] = []

        for symbol in stock_universe:
            row = self._build_single_stock_features(
                symbol, date, ohlcv_data, fundamental_data, sentiment_scores
            )
            feature_rows.append(row)

        if len(feature_rows) == 0:
            logger.warning("Empty stock universe on %s", date)
            return torch.zeros((0, len(self.stock_feature_names)), dtype=torch.float32)

        feature_matrix = np.stack(feature_rows, axis=0).astype(np.float32)
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1.0, neginf=-1.0)

        return torch.from_numpy(feature_matrix)

    def _build_single_stock_features(
        self,
        symbol: str,
        date: str,
        ohlcv_data: Dict[str, pd.DataFrame],
        fundamental_data: Dict[str, Dict[str, float]],
        sentiment_scores: Optional[Dict[str, float]],
    ) -> np.ndarray:
        """Build 22D feature vector for a single stock."""
        features: Dict[str, float] = {}
        date_dt = pd.Timestamp(date)

        # Technical features from OHLCV
        if symbol in ohlcv_data and len(ohlcv_data[symbol]) > 0:
            df = ohlcv_data[symbol]
            df = df[df.index <= date_dt]  # Strict timestamp gating
            tech = self.technical.compute(df)
            features.update(tech)
        else:
            logger.debug("No OHLCV data for %s on %s", symbol, date)

        # Fundamental features
        sector = self.stock_to_sector.get(symbol, "Unknown")
        if symbol in fundamental_data:
            fund = self.fundamental.compute_single(
                fundamental_data[symbol], sector, fundamental_data
            )
            features.update(fund)

        # Sentiment score (if available)
        if sentiment_scores and symbol in sentiment_scores:
            features["sentiment_score"] = sentiment_scores[symbol]

        # Build ordered vector
        row = np.zeros(len(self.stock_feature_names), dtype=np.float32)
        for i, feat_name in enumerate(self.stock_feature_names):
            if feat_name in features:
                row[i] = features[feat_name]

        return row

    # ── Sector Feature Matrix for GNN ────────────────────────────────

    def build_sector_feature_matrix(
        self,
        date: str,
        ohlcv_data: Dict[str, pd.DataFrame],
        macro_data: pd.DataFrame,
        sentiment_scores: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        """Build sector-level feature matrix for the GNN encoder.

        Returns:
            Tensor of shape (num_sectors, sector_feature_dim).
        """
        sector_features: List[np.ndarray] = []

        for sector_name in self.sector_names:
            sector_stocks = self.sector_mapping[sector_name]

            # Aggregate sector-level technical stats
            sector_feat = self.sector.compute_single_sector(
                sector_name, sector_stocks, ohlcv_data, as_of_date=date
            )

            # Add sentiment if available
            if sentiment_scores and sector_name in sentiment_scores:
                sector_feat["sentiment"] = sentiment_scores[sector_name]

            feat_vec = np.array(list(sector_feat.values()), dtype=np.float32)
            sector_features.append(feat_vec)

        feature_matrix = np.stack(sector_features, axis=0)
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1.0, neginf=-1.0)

        return torch.from_numpy(feature_matrix)

    # ── Goal Embedding Construction ──────────────────────────────────

    @staticmethod
    def build_goal_embedding(
        sector_weights: torch.Tensor,
        regime_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate macro output into goal embedding for Micro agent.

        Args:
            sector_weights: Tensor of shape (11,) — sector allocation weights.
            regime_probs: Tensor of shape (3,) — regime class probabilities.

        Returns:
            Tensor of shape (14,) — raw goal; the Micro agent's GoalEncoder
            projects this to 64D.
        """
        return torch.cat([sector_weights, regime_probs], dim=-1)

    # ── Utilities ────────────────────────────────────────────────────

    def get_stock_universe(self, sectors: Optional[List[str]] = None) -> List[str]:
        """Return flat list of all stocks, optionally filtered by sectors."""
        if sectors is None:
            sectors = self.sector_names
        stocks: List[str] = []
        for sector in sectors:
            if sector in self.sector_mapping:
                stocks.extend(self.sector_mapping[sector])
        return stocks

    def get_sector_for_stock(self, symbol: str) -> str:
        """Return sector name for a given stock symbol."""
        return self.stock_to_sector.get(symbol, "Unknown")

    def normalize_features(
        self, tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        """Z-score normalisation with numerical safety."""
        std_safe = std.clamp(min=1e-8)
        return (tensor - mean) / std_safe
