"""
File: fundamental_features.py
Module: features
Description: Normalised fundamental features: P/E vs sector median z-score, P/B z-score,
             ROE/ROCE, EV/EBITDA, earnings surprise %. All metrics are sector-relative
             to enable cross-sector comparisons within the Micro agent's stock universe.
Design Decisions:
    - Sector-relative z-scores used because absolute P/E is meaningless cross-sector
      (IT P/E of 25 is cheap; Metal P/E of 25 is expensive).
    - Z-score = (stock_metric - sector_median) / sector_std normalises across sectors.
    - Missing values imputed with sector median (conservative neutral assumption).
References:
    - Fama & French (1993): Value factor (HML) uses P/B sorting
    - Asness et al. (2013): "Value and Momentum Everywhere" — cross-sectional normalisation
Author: HRL-SARP Framework
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class FundamentalFeatures:
    """
    Computes sector-relative fundamental features for stock valuation.

    Each metric is normalised to a z-score relative to its sector:
        z = (x_stock - median_sector) / std_sector

    This allows the Micro agent to compare stocks across sectors fairly:
        - z < -1.0: Potentially undervalued relative to sector peers
        - z ≈ 0: Fairly valued
        - z > 1.0: Potentially overvalued

    Features computed:
        1. pe_zscore: P/E z-score vs sector median
        2. pb_zscore: P/B z-score vs sector median
        3. roe: Return on Equity (raw %, normalised to [0, 1])
        4. roce: Return on Capital Employed (raw %, normalised)
        5. debt_to_equity: D/E ratio (raw, capped at 5 for normalisation)
        6. ev_ebitda_zscore: EV/EBITDA z-score vs sector
        7. earnings_surprise: Latest quarter surprise %
        8. promoter_pledge_pct: Promoter pledge % (governance metric)
    """

    def __init__(self, config_path: str = "config/data_config.yaml") -> None:
        with open(config_path, "r") as f:
            self.config: Dict[str, Any] = yaml.safe_load(f)
        self.sector_mapping: Dict[str, List[str]] = self.config.get("sectors", {}).get("mapping", {})
        logger.info("FundamentalFeatures initialised | sectors=%d", len(self.sector_mapping))

    # ══════════════════════════════════════════════════════════════════
    # SECTOR-RELATIVE Z-SCORE COMPUTATION
    # ══════════════════════════════════════════════════════════════════
    @staticmethod
    def compute_zscore(values: pd.Series) -> pd.Series:
        """
        Compute cross-sectional z-score for a metric.

        Z = (x - median) / std

        Using median instead of mean for robustness against outliers
        (common in Indian markets — e.g., loss-making companies with negative P/E).

        Args:
            values: Series of metric values for stocks in the same sector.
        Returns:
            pd.Series of z-scores, clipped to [-3, 3].
        """
        median = values.median()
        std = values.std()
        if std == 0 or pd.isna(std):
            return pd.Series(0.0, index=values.index)
        zscore = (values - median) / std
        return zscore.clip(lower=-3.0, upper=3.0)

    # ══════════════════════════════════════════════════════════════════
    # PER-SECTOR NORMALISATION
    # ══════════════════════════════════════════════════════════════════
    def normalise_within_sector(
        self, fundamentals_df: pd.DataFrame, metric: str
    ) -> pd.Series:
        """
        Compute sector-relative z-scores for a fundamental metric.

        For each stock, the metric is z-scored relative to its sector peers.
        This handles the cross-sector valuation disparity in Indian markets:
            - IT sector: median P/E ~25 (premium for visibility and cash flow)
            - Metals: median P/E ~8 (cyclical, commodity-linked)
            - FMCG: median P/E ~50 (defensive, stable growth)

        Args:
            fundamentals_df: DataFrame with symbol as index, metric columns.
            metric: Column name to normalise (e.g., "pe_ratio").
        Returns:
            pd.Series of z-scores indexed by symbol.
        """
        if metric not in fundamentals_df.columns:
            logger.warning("Metric '%s' not found in fundamentals DataFrame.", metric)
            return pd.Series(0.0, index=fundamentals_df.index)

        result = pd.Series(0.0, index=fundamentals_df.index, dtype=float)

        for sector, symbols in self.sector_mapping.items():
            # Find stocks in this sector that exist in our data
            sector_mask = fundamentals_df.index.isin(symbols)
            sector_values = fundamentals_df.loc[sector_mask, metric].dropna()

            if len(sector_values) < 2:
                # Not enough data for z-score; assign 0 (neutral)
                continue

            zscores = self.compute_zscore(sector_values)
            result.update(zscores)

        return result

    # ══════════════════════════════════════════════════════════════════
    # INDIVIDUAL FEATURE COMPUTATIONS
    # ══════════════════════════════════════════════════════════════════
    def compute_pe_zscore(self, fundamentals_df: pd.DataFrame) -> pd.Series:
        """
        P/E z-score relative to sector median.

        Negative z-score → stock is cheaper than sector peers → value signal.
        Used in the value_discovery_reward: R_value = return * 𝟙(PE_z < -0.5).

        Returns:
            pd.Series of P/E z-scores.
        """
        return self.normalise_within_sector(fundamentals_df, "pe_ratio")

    def compute_pb_zscore(self, fundamentals_df: pd.DataFrame) -> pd.Series:
        """
        P/B z-score relative to sector median.

        P/B is particularly relevant for financials and asset-heavy sectors (Metals, Realty).
        P/B < 1 with positive ROE is a classic deep-value signal.
        """
        return self.normalise_within_sector(fundamentals_df, "pb_ratio")

    def compute_ev_ebitda_zscore(self, fundamentals_df: pd.DataFrame) -> pd.Series:
        """
        EV/EBITDA z-score relative to sector.

        Preferred valuation metric for capital-intensive sectors because it
        normalises for capital structure differences (debt vs equity financing).
        """
        return self.normalise_within_sector(fundamentals_df, "ev_ebitda")

    @staticmethod
    def normalise_roe(roe: pd.Series) -> pd.Series:
        """
        Normalise ROE to [0, 1] range.

        ROE > 15% is considered good in Indian context.
        ROE > 25% is excellent (e.g., TCS, HDFC Bank).
        Negative ROE is clipped to 0.

        Normalisation: ROE_norm = clip(ROE, 0, 40) / 40
        """
        return roe.clip(lower=0, upper=40) / 40.0

    @staticmethod
    def normalise_roce(roce: pd.Series) -> pd.Series:
        """
        Normalise ROCE to [0, 1] range.
        ROCE > 20% is strong. Scale: clip(ROCE, 0, 50) / 50.
        """
        return roce.clip(lower=0, upper=50) / 50.0

    @staticmethod
    def normalise_de_ratio(de: pd.Series) -> pd.Series:
        """
        Normalise Debt-to-Equity ratio.

        D/E ratio inverted and normalised so that:
            - D/E = 0 (no debt) → feature = 1.0 (best)
            - D/E = 1 → feature = 0.5
            - D/E ≥ 3 → feature ≈ 0 (heavily leveraged, higher risk)

        Formula: feature = 1 / (1 + D/E), capped at D/E = 5.
        Higher is better for the agent (less leverage = less risk).
        """
        de_capped = de.clip(lower=0, upper=5)
        return 1.0 / (1.0 + de_capped)

    @staticmethod
    def normalise_pledge_pct(pledge: pd.Series) -> pd.Series:
        """
        Normalise promoter pledge percentage to risk score.

        Pledge > 20% is a governance red flag (risk of forced selling).
        Pledge > 50% is critical risk.

        Returns inverted score: 1.0 = no pledge; 0.0 = 100% pledged.
        """
        return 1.0 - pledge.clip(lower=0, upper=100) / 100.0

    @staticmethod
    def normalise_earnings_surprise(surprise: pd.Series) -> pd.Series:
        """
        Normalise earnings surprise to [-1, 1] range.

        Surprise capped at ±50% for outlier resistance.
        Positive surprise → positive signal for value discovery bonus.
        """
        return surprise.clip(lower=-50, upper=50) / 50.0

    # ══════════════════════════════════════════════════════════════════
    # MASTER COMPUTATION
    # ══════════════════════════════════════════════════════════════════
    def compute_all(self, fundamentals_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all fundamental features for the stock universe.

        Args:
            fundamentals_df: DataFrame with symbol as index, raw fundamental columns.
        Returns:
            pd.DataFrame with normalised features, symbol as index.
        """
        features = pd.DataFrame(index=fundamentals_df.index)

        # ── Sector-relative z-scores ─────────────────────────────────
        features["pe_zscore"] = self.compute_pe_zscore(fundamentals_df)
        features["pb_zscore"] = self.compute_pb_zscore(fundamentals_df)
        features["ev_ebitda_zscore"] = self.compute_ev_ebitda_zscore(fundamentals_df)

        # ── Normalised raw metrics ───────────────────────────────────
        if "roe" in fundamentals_df.columns:
            features["roe"] = self.normalise_roe(fundamentals_df["roe"])
        if "roce" in fundamentals_df.columns:
            features["roce"] = self.normalise_roce(fundamentals_df["roce"])
        if "debt_to_equity" in fundamentals_df.columns:
            features["debt_to_equity"] = self.normalise_de_ratio(fundamentals_df["debt_to_equity"])
        if "promoter_pledge_pct" in fundamentals_df.columns:
            features["promoter_pledge_pct"] = self.normalise_pledge_pct(fundamentals_df["promoter_pledge_pct"])
        if "earnings_growth_yoy" in fundamentals_df.columns:
            features["earnings_surprise"] = self.normalise_earnings_surprise(fundamentals_df["earnings_growth_yoy"])

        # ── Fill missing values with neutral (0) ─────────────────────
        features = features.fillna(0.0)

        logger.info("Computed %d fundamental features for %d stocks", len(features.columns), len(features))
        return features
