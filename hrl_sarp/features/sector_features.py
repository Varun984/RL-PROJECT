"""
File: sector_features.py
Module: features
Description: Computes sector-level features: 13-week relative strength per sector,
             sector rotation score, and breadth indicator. These feed the sector GNN
             and the Macro agent's sector allocation decision.
Design Decisions:
    - Relative strength (RS) computed over 13 weeks (1 quarter) — standard in
      sector rotation strategies (O'Neil, Faber).
    - Sector rotation score captures momentum shifts between sectors.
    - Breadth indicator measures percentage of advancing stocks per sector.
References:
    - Faber (2007): "A Quantitative Approach to Tactical Asset Allocation"
    - O'Neil (2009): "How to Make Money in Stocks" — sector RS ranking
Author: HRL-SARP Framework
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class SectorFeatures:
    """
    Computes sector-level features for the Macro agent and GNN encoder.

    Features per sector:
        1. relative_strength_13w: 13-week sector return relative to Nifty 50
        2. rotation_score: Rate-of-change of relative strength (momentum of momentum)
        3. breadth_advance_pct: % of sector stocks with positive 20D return
        4. sector_volatility: 20D rolling annualised volatility of sector index
        5. sector_return_1w: Trailing 1-week sector return
        6. sector_return_4w: Trailing 4-week sector return

    These features form the node-level feature matrix for the sector GNN,
    where each of the 11 NSE sectors is a graph node.
    """

    # ── NSE 11 sectors in canonical order ────────────────────────────
    SECTORS = ["IT", "Financials", "Auto", "Pharma", "FMCG",
               "Energy", "Metals", "Realty", "Media", "Telecom", "Infra"]

    def __init__(self, config_path: str = "config/data_config.yaml") -> None:
        with open(config_path, "r", encoding="utf-8") as f:self.config: Dict[str, Any] = yaml.safe_load(f)
        self.sector_mapping: Dict[str, List[str]] = self.config.get("sectors", {}).get("mapping", {})
        logger.info("SectorFeatures initialised | sectors=%d", len(self.sector_mapping))

    # ══════════════════════════════════════════════════════════════════
    # RELATIVE STRENGTH (13-week)
    # ══════════════════════════════════════════════════════════════════
    @staticmethod
    def compute_relative_strength(
        sector_close: pd.Series, benchmark_close: pd.Series, period: int = 65
    ) -> pd.Series:
        """
        Compute sector relative strength vs benchmark (Nifty 50).

        RS = (Sector Return over period) - (Benchmark Return over period)

        A positive RS indicates the sector is outperforming the market.
        13-week (65 trading days) lookback captures the medium-term momentum
        cycle in Indian sector rotation.

        Typical sector rotation cycle in India:
            1. Recovery: Financials, Auto lead
            2. Expansion: IT, FMCG, Consumer
            3. Late cycle: Energy, Metals, Infra (commodity reflation)
            4. Contraction: Pharma, FMCG (defensive rotation)

        Args:
            sector_close: Sector index close price series.
            benchmark_close: Nifty 50 close price series.
            period: Lookback period in trading days (65 ≈ 13 weeks).
        Returns:
            pd.Series of relative strength values.
        """
        sector_return = sector_close.pct_change(periods=period)
        benchmark_return = benchmark_close.pct_change(periods=period)
        rs = sector_return - benchmark_return
        return rs

    # ══════════════════════════════════════════════════════════════════
    # SECTOR ROTATION SCORE
    # ══════════════════════════════════════════════════════════════════
    @staticmethod
    def compute_rotation_score(
        relative_strength: pd.Series, period: int = 20
    ) -> pd.Series:
        """
        Compute sector rotation score: rate-of-change of relative strength.

        Rotation Score = RS_today - RS_{period_days_ago}

        This captures the "momentum of momentum":
            - Positive rotation score: Sector gaining leadership (accelerating momentum)
            - Negative rotation score: Sector losing leadership (decelerating)

        Useful for early sector rotation signals before price trends establish.

        Args:
            relative_strength: Relative strength time series.
            period: Rate-of-change lookback.
        Returns:
            pd.Series of rotation scores.
        """
        return relative_strength.diff(periods=period)

    # ══════════════════════════════════════════════════════════════════
    # BREADTH INDICATOR
    # ══════════════════════════════════════════════════════════════════
    @staticmethod
    def compute_breadth(
        stock_returns: Dict[str, pd.Series], period: int = 20
    ) -> pd.Series:
        """
        Compute sector breadth: % of stocks with positive N-day return.

        Breadth > 70%: Broad-based sector rally (high conviction signal).
        Breadth < 30%: Broad-based sector weakness.
        Divergence (price up + breadth down): Narrow rally, fragile trend.

        In Indian markets, narrow rallies led by 2-3 heavyweight stocks
        (e.g., Reliance in Energy, HDFC Bank in Financials) are common
        and unreliable for sector allocation.

        Args:
            stock_returns: Dict mapping symbol → return series for stocks in sector.
            period: Return lookback period.
        Returns:
            pd.Series of breadth values ∈ [0, 1].
        """
        if not stock_returns:
            return pd.Series(dtype=float)

        # Get common index
        first_series = list(stock_returns.values())[0]
        index = first_series.index

        advancing = pd.DataFrame(index=index)
        for symbol, returns in stock_returns.items():
            period_return = returns.rolling(window=period).sum()
            advancing[symbol] = (period_return > 0).astype(float)

        # Breadth = fraction of advancing stocks
        breadth = advancing.mean(axis=1)
        return breadth

    # ══════════════════════════════════════════════════════════════════
    # SECTOR VOLATILITY
    # ══════════════════════════════════════════════════════════════════
    @staticmethod
    def compute_sector_volatility(
        sector_close: pd.Series, period: int = 20
    ) -> pd.Series:
        """
        Compute annualised rolling volatility for a sector index.

        High sector volatility signals uncertainty and may warrant lower allocation.
        Low volatility sectors are preferred in defensive/sideways regimes.

        Args:
            sector_close: Sector index close prices.
            period: Volatility lookback.
        Returns:
            pd.Series of annualised volatility values.
        """
        log_returns = np.log(sector_close / sector_close.shift(1))
        return log_returns.rolling(window=period).std() * np.sqrt(252)

    # ══════════════════════════════════════════════════════════════════
    # SECTOR CORRELATION MATRIX
    # ══════════════════════════════════════════════════════════════════
    @staticmethod
    def compute_correlation_matrix(
        sector_returns: pd.DataFrame, window: int = 60
    ) -> pd.DataFrame:
        """
        Compute rolling pairwise correlation between sector returns.

        This matrix forms the adjacency matrix for the sector GNN:
            - Nodes: 11 NSE sectors
            - Edges: Rolling 60D return correlation
            - Edge weight: |correlation| (strength of relationship)

        High inter-sector correlation reduces diversification benefit.
        The GNN encoder captures these dynamic relationships.

        Args:
            sector_returns: DataFrame with sector names as columns, daily returns.
            window: Rolling correlation window (60 trading days ≈ 3 months).
        Returns:
            pd.DataFrame: Correlation matrix (11 × 11).
        """
        return sector_returns.rolling(window=window, min_periods=20).corr()

    # ══════════════════════════════════════════════════════════════════
    # MASTER COMPUTATION
    # ══════════════════════════════════════════════════════════════════
    def compute_all(
        self,
        sector_index_data: Dict[str, pd.DataFrame],
        stock_data: Dict[str, pd.DataFrame],
        benchmark_close: pd.Series,
    ) -> pd.DataFrame:
        """
        Compute all sector features for all 11 NSE sectors.

        Args:
            sector_index_data: Dict mapping sector_name → OHLCV DataFrame.
            stock_data: Dict mapping symbol → OHLCV DataFrame (for breadth computation).
            benchmark_close: Nifty 50 close price series (for relative strength).
        Returns:
            pd.DataFrame with multi-level columns: (sector, feature_name).
        """
        all_features: Dict[str, pd.DataFrame] = {}

        for sector in self.SECTORS:
            if sector not in sector_index_data:
                logger.warning("No index data for sector: %s", sector)
                continue

            sector_df = sector_index_data[sector]
            if "Close" not in sector_df.columns:
                continue
            sector_close = sector_df["Close"]

            features = pd.DataFrame(index=sector_close.index)

            # ── Relative strength ────────────────────────────────────
            rs = self.compute_relative_strength(sector_close, benchmark_close)
            features["relative_strength_13w"] = rs

            # ── Rotation score ───────────────────────────────────────
            features["rotation_score"] = self.compute_rotation_score(rs)

            # ── Sector returns ───────────────────────────────────────
            features["sector_return_1w"] = sector_close.pct_change(5)
            features["sector_return_4w"] = sector_close.pct_change(20)

            # ── Sector volatility ────────────────────────────────────
            features["sector_volatility"] = self.compute_sector_volatility(sector_close)

            # ── Breadth indicator ────────────────────────────────────
            sector_symbols = self.sector_mapping.get(sector, [])
            sector_stock_returns: Dict[str, pd.Series] = {}
            for sym in sector_symbols:
                if sym in stock_data and "Close" in stock_data[sym].columns:
                    sector_stock_returns[sym] = np.log(stock_data[sym]["Close"] / stock_data[sym]["Close"].shift(1))

            if sector_stock_returns:
                features["breadth_advance_pct"] = self.compute_breadth(sector_stock_returns)

            all_features[sector] = features

        # ── Combine into multi-column DataFrame ──────────────────────
        if not all_features:
            return pd.DataFrame()

        combined = pd.concat(all_features, axis=1)
        combined = combined.ffill().fillna(0.0)

        logger.info("Sector features computed | sectors=%d | features_per_sector=%d",
                     len(all_features), len(list(all_features.values())[0].columns))
        return combined

    # ══════════════════════════════════════════════════════════════════
    # SECTOR FEATURE MATRIX FOR GNN
    # ══════════════════════════════════════════════════════════════════
    def get_sector_feature_matrix(
        self,
        sector_features: pd.DataFrame,
        date: str,
    ) -> np.ndarray:
        """
        Extract sector feature matrix for a specific date (for GNN input).

        Returns a (num_sectors, num_features) numpy array suitable for
        PyTorch Geometric graph construction.

        Args:
            sector_features: Multi-column DataFrame from compute_all().
            date: Date to extract features for.
        Returns:
            np.ndarray of shape (11, num_features_per_sector).
        """
        num_features = len(sector_features.columns) // len(self.SECTORS) if len(sector_features.columns) > 0 else 0
        matrix = np.zeros((len(self.SECTORS), max(num_features, 1)))

        for i, sector in enumerate(self.SECTORS):
            try:
                sector_cols = [col for col in sector_features.columns if col[0] == sector]
                if sector_cols:
                    row = sector_features.loc[date, sector_cols]
                    matrix[i, :len(row)] = row.values
            except (KeyError, Exception):
                continue

        return matrix

    def get_adjacency_matrix(
        self,
        sector_index_data: Dict[str, pd.DataFrame],
        date: str,
        window: int = 60,
    ) -> np.ndarray:
        """
        Compute correlation-based adjacency matrix for sector GNN.

        Returns (11, 11) adjacency matrix where edge weight = |correlation|
        between sector return time series over past `window` days.

        Args:
            sector_index_data: Dict mapping sector → OHLCV DataFrame.
            date: Date for which to compute adjacency.
            window: Rolling correlation window.
        Returns:
            np.ndarray of shape (11, 11) — adjacency matrix.
        """
        # Build sector returns DataFrame
        returns_df = pd.DataFrame()
        for sector in self.SECTORS:
            if sector in sector_index_data:
                close = sector_index_data[sector]["Close"]
                returns_df[sector] = np.log(close / close.shift(1))

        if returns_df.empty:
            return np.eye(len(self.SECTORS))

        # Get returns up to the given date
        returns_up_to = returns_df.loc[:date].tail(window)

        if len(returns_up_to) < 10:
            return np.eye(len(self.SECTORS))

        corr = returns_up_to.corr()

        # Build adjacency matrix in canonical sector order
        adj = np.eye(len(self.SECTORS))
        for i, s1 in enumerate(self.SECTORS):
            for j, s2 in enumerate(self.SECTORS):
                if s1 in corr.columns and s2 in corr.columns:
                    adj[i, j] = abs(corr.loc[s1, s2])

        return adj
