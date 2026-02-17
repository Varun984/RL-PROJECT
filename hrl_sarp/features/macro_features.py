"""
File: macro_features.py
Module: features
Description: Computes macro-level features for the Macro agent's 18D state vector:
             India VIX normalisation, FII flow normalisation, PCR index, USD/INR momentum,
             crude regime encoding, and yield curve features.
Design Decisions:
    - All features normalised to approximately [-1, 1] or [0, 1] for neural network stability.
    - Rolling z-scores used for mean-reverting signals (VIX, flows).
    - India-specific: FII/DII flows are normalised by Nifty market cap for stationarity.
References:
    - Macro factor models: Chen, Roll, Ross (1986) — macro factors in asset pricing.
    - VIX regime detection: Papenbrock & Schwendner (2015) — HMM on VIX.
Author: HRL-SARP Framework
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class MacroFeatures:
    """
    Computes and normalises macroeconomic features for the Macro agent.

    The 18D macro state vector consists of:
        1. india_vix: Normalised VIX level
        2. india_vix_change_1w: Weekly VIX change (regime shift indicator)
        3. fii_flow_norm: Z-scored FII net flow
        4. dii_flow_norm: Z-scored DII net flow
        5. pcr_index: Put-Call Ratio (normalised)
        6. usd_inr_momentum: 20-day USD/INR momentum
        7. crude_regime: Crude oil regime encoding (0–3)
        8. rbi_rate: Current repo rate (normalised)
        9. credit_spread: Corporate bond spread (normalised)
        10. yield_curve_slope: 10Y-2Y yield spread
        11. nifty_return_4w: Trailing 4-week Nifty return
        12. nifty_volatility_4w: Trailing 4-week Nifty volatility
        13. breadth_advance_decline: Advance-decline ratio
        14. fno_oi_change: F&O OI weekly change
        15. event_risk_flag: Binary upcoming-event flag
        16. days_to_next_event: Days until next event (normalised)
        17. regime_label_prev: Previous regime classification
        18. portfolio_drawdown: Current drawdown from peak

    All features designed to be stationary for stable RL training.
    """

    def __init__(self, config_path: str = "config/data_config.yaml") -> None:
        with open(config_path, "r") as f:
            self.config: Dict[str, Any] = yaml.safe_load(f)
        logger.info("MacroFeatures initialised.")

    # ══════════════════════════════════════════════════════════════════
    # INDIA VIX FEATURES
    # ══════════════════════════════════════════════════════════════════
    @staticmethod
    def normalise_vix(vix: pd.Series, lookback: int = 252) -> pd.DataFrame:
        """
        Normalise India VIX and compute VIX change features.

        VIX is normalised via rolling z-score because absolute VIX levels
        exhibit regime-dependent means (VIX of 15 is different in 2020 vs 2024).

        VIX change (week-over-week) captures sudden fear spikes:
            - VIX spike > 2σ: panic selling, potential bottom for contrarian entry
            - VIX compression: complacency, potential top

        Args:
            vix: Raw India VIX time series.
            lookback: Rolling window for z-score normalisation.
        Returns:
            pd.DataFrame with [india_vix, india_vix_change_1w].
        """
        rolling_mean = vix.rolling(window=lookback, min_periods=20).mean()
        rolling_std = vix.rolling(window=lookback, min_periods=20).std()

        vix_zscore = (vix - rolling_mean) / rolling_std.replace(0, np.nan)
        vix_weekly_change = vix.pct_change(periods=5)  # 5 trading days ≈ 1 week

        return pd.DataFrame({
            "india_vix": vix_zscore.clip(-3, 3),
            "india_vix_change_1w": vix_weekly_change.clip(-1, 1),
        })

    # ══════════════════════════════════════════════════════════════════
    # FII/DII FLOW NORMALISATION
    # ══════════════════════════════════════════════════════════════════
    @staticmethod
    def normalise_flows(fii_flow: pd.Series, dii_flow: Optional[pd.Series] = None,
                        lookback: int = 60) -> pd.DataFrame:
        """
        Normalise FII and DII flows via rolling z-score.

        Raw FII/DII flows are in ₹ Crores and non-stationary (flows scale
        with market cap growth). Z-scoring over 60-day rolling window makes
        them stationary and comparable across time periods.

        Interpretation:
            - FII z > 1: Strong foreign buying (bullish signal)
            - FII z < -1: Foreign selling (bearish, watch for rupee depreciation)
            - FII selling + DII buying: Domestic institutions absorbing FII supply

        Args:
            fii_flow: FII net flow series (₹ Crores).
            dii_flow: DII net flow series (if available).
            lookback: Rolling z-score window.
        Returns:
            pd.DataFrame with [fii_flow_norm, dii_flow_norm].
        """
        result = pd.DataFrame(index=fii_flow.index)

        # FII normalisation
        fii_mean = fii_flow.rolling(lookback, min_periods=10).mean()
        fii_std = fii_flow.rolling(lookback, min_periods=10).std()
        result["fii_flow_norm"] = ((fii_flow - fii_mean) / fii_std.replace(0, np.nan)).clip(-3, 3)

        # DII normalisation
        if dii_flow is not None:
            dii_mean = dii_flow.rolling(lookback, min_periods=10).mean()
            dii_std = dii_flow.rolling(lookback, min_periods=10).std()
            result["dii_flow_norm"] = ((dii_flow - dii_mean) / dii_std.replace(0, np.nan)).clip(-3, 3)
        else:
            result["dii_flow_norm"] = 0.0

        return result

    # ══════════════════════════════════════════════════════════════════
    # PUT-CALL RATIO
    # ══════════════════════════════════════════════════════════════════
    @staticmethod
    def normalise_pcr(pcr: pd.Series) -> pd.Series:
        """
        Normalise Put-Call Ratio to [-1, 1] range.

        PCR = Put OI / Call OI on Nifty options.
        Normalisation: (PCR - 1.0) — centres around 1.0 (neutral).

        Indian market PCR interpretation:
            - PCR > 1.2 → normalised > 0.2: Bullish (heavy put writing = support)
            - PCR 0.8–1.2 → normalised -0.2 to 0.2: Neutral
            - PCR < 0.8 → normalised < -0.2: Bearish (excess call writing = resistance)

        Args:
            pcr: Raw PCR time series.
        Returns:
            pd.Series: Normalised PCR ∈ [-1, 1].
        """
        normalised = (pcr - 1.0).clip(-1, 1)
        return normalised

    # ══════════════════════════════════════════════════════════════════
    # USD/INR MOMENTUM
    # ══════════════════════════════════════════════════════════════════
    @staticmethod
    def compute_usd_inr_momentum(usd_inr: pd.Series, period: int = 20) -> pd.Series:
        """
        Compute USD/INR rate-of-change (momentum).

        Rupee appreciation (negative momentum) is:
            - Bearish for IT sector (USD revenue translated to fewer rupees)
            - Bullish for importers (cheaper raw materials)
        Rupee depreciation (positive momentum) is the opposite.

        Args:
            usd_inr: USD/INR exchange rate series.
            period: Momentum lookback period.
        Returns:
            pd.Series: Momentum normalised to approximately [-1, 1].
        """
        momentum = usd_inr.pct_change(periods=period)
        # Scale: typical monthly INR movement is 1-3%
        return (momentum * 10).clip(-1, 1)

    # ══════════════════════════════════════════════════════════════════
    # CRUDE OIL REGIME
    # ══════════════════════════════════════════════════════════════════
    @staticmethod
    def encode_crude_regime(crude_price: pd.Series) -> pd.Series:
        """
        Encode crude oil price into discrete regime categories.

        India imports ~85% of crude oil needs, making oil prices a critical
        macro variable. Higher crude → higher inflation → tighter RBI policy
        → negative for equities (especially OMCs, airlines, paints).

        Regime encoding:
            0: Low (<$60/bbl) — fiscal tailwind
            1: Medium ($60-80) — neutral
            2: High ($80-100) — fiscal headwind
            3: Extreme (>$100) — stagflation risk

        Normalised to [0, 1] → regime / 3.

        Args:
            crude_price: Brent crude price series.
        Returns:
            pd.Series: Regime encoding ∈ {0, 0.33, 0.67, 1.0}.
        """
        regime = pd.cut(
            crude_price,
            bins=[0, 60, 80, 100, float("inf")],
            labels=[0, 1, 2, 3],
        ).astype(float)
        return regime / 3.0  # Normalise to [0, 1]

    # ══════════════════════════════════════════════════════════════════
    # NIFTY RETURN AND VOLATILITY
    # ══════════════════════════════════════════════════════════════════
    @staticmethod
    def compute_nifty_features(nifty_close: pd.Series) -> pd.DataFrame:
        """
        Compute trailing return and volatility for Nifty 50.

        4-week (20 trading day) window captures the medium-term trend:
            - Positive 4w return + low vol: Strong bull trend
            - Negative 4w return + high vol: Bear market / correction
            - Low return + low vol: Range-bound / sideways

        Args:
            nifty_close: Nifty 50 close price series.
        Returns:
            pd.DataFrame with [nifty_return_4w, nifty_volatility_4w].
        """
        log_returns = np.log(nifty_close / nifty_close.shift(1))

        return pd.DataFrame({
            "nifty_return_4w": log_returns.rolling(20).sum().clip(-0.3, 0.3),
            "nifty_volatility_4w": (log_returns.rolling(20).std() * np.sqrt(252)).clip(0, 1),
        })

    # ══════════════════════════════════════════════════════════════════
    # YIELD CURVE
    # ══════════════════════════════════════════════════════════════════
    @staticmethod
    def compute_yield_curve_slope(long_yield: pd.Series, short_yield: Optional[pd.Series] = None) -> pd.Series:
        """
        Compute yield curve slope (10Y - 2Y spread).

        Inverted yield curve (negative slope) historically precedes recessions.
        In India, an inverted RBI rate vs 10Y G-sec signals tight monetary conditions.

        If short-term yield is unavailable, we use repo rate as proxy.

        Args:
            long_yield: 10-year yield series.
            short_yield: 2-year yield or repo rate series.
        Returns:
            pd.Series: Yield spread in percentage points, clipped to [-2, 4].
        """
        if short_yield is not None:
            slope = long_yield - short_yield
        else:
            slope = long_yield - 6.25  # Use default repo rate as proxy
        return slope.clip(-2, 4) / 4.0  # Normalise to [-0.5, 1.0]

    # ══════════════════════════════════════════════════════════════════
    # RBI RATE NORMALISATION
    # ══════════════════════════════════════════════════════════════════
    @staticmethod
    def normalise_rbi_rate(rate: float) -> float:
        """
        Normalise RBI repo rate to [0, 1] range.

        Historical range for Indian repo rate: 4.0% (2020 COVID low) to 8.0% (2013 peak).
        Normalisation: (rate - 4) / 4 maps 4% → 0 and 8% → 1.

        Args:
            rate: Current repo rate.
        Returns:
            Normalised rate ∈ [0, 1].
        """
        return max(0.0, min(1.0, (rate - 4.0) / 4.0))

    # ══════════════════════════════════════════════════════════════════
    # MASTER COMPUTATION
    # ══════════════════════════════════════════════════════════════════
    def compute_all(self, macro_data: pd.DataFrame, nifty_close: pd.Series,
                    pcr_series: Optional[pd.Series] = None,
                    rbi_rate: float = 6.25) -> pd.DataFrame:
        """
        Compute all macro features and assemble the 18D state vector.

        Args:
            macro_data: Aggregated macro DataFrame from MacroFetcher.
            nifty_close: Nifty 50 close price series.
            pcr_series: Put-Call Ratio series (if available).
            rbi_rate: Current RBI repo rate.
        Returns:
            pd.DataFrame with all normalised macro features.
        """
        features = pd.DataFrame(index=macro_data.index)

        # ── VIX features ─────────────────────────────────────────────
        if "india_vix" in macro_data.columns:
            vix_feats = self.normalise_vix(macro_data["india_vix"])
            features = features.join(vix_feats)

        # ── FII/DII flows ────────────────────────────────────────────
        if "fii_net_buy" in macro_data.columns:
            flow_feats = self.normalise_flows(
                macro_data["fii_net_buy"],
                macro_data.get("dii_net_buy"),
            )
            features = features.join(flow_feats)

        # ── PCR ──────────────────────────────────────────────────────
        if pcr_series is not None:
            features["pcr_index"] = self.normalise_pcr(pcr_series)
        else:
            features["pcr_index"] = 0.0

        # ── USD/INR momentum ─────────────────────────────────────────
        if "usd_inr" in macro_data.columns:
            features["usd_inr_momentum"] = self.compute_usd_inr_momentum(macro_data["usd_inr"])

        # ── Crude regime ─────────────────────────────────────────────
        if "crude_oil" in macro_data.columns:
            features["crude_regime"] = self.encode_crude_regime(macro_data["crude_oil"])

        # ── RBI rate ─────────────────────────────────────────────────
        features["rbi_rate"] = self.normalise_rbi_rate(rbi_rate)

        # ── Yield curve ──────────────────────────────────────────────
        if "us_10y_yield" in macro_data.columns:
            features["yield_curve_slope"] = self.compute_yield_curve_slope(macro_data["us_10y_yield"])

        # ── Nifty features ───────────────────────────────────────────
        nifty_feats = self.compute_nifty_features(nifty_close)
        features = features.join(nifty_feats, how="left")

        # Fill NaN and log
        features = features.ffill().fillna(0.0)
        logger.info("Macro features computed: %d features, %d rows", len(features.columns), len(features))
        return features
