"""
File: technical_features.py
Module: features
Description: Computes vectorised technical indicators using pandas-ta: RSI, MACD,
             Bollinger Bands, volume ratio, delivery volume %. All computations
             are vectorised for batch processing across multiple stocks.
Design Decisions:
    - pandas-ta chosen for its comprehensive indicator library and vectorised ops.
    - All features normalised to [0, 1] or z-scored for neural network consumption.
    - Delivery volume % is India-specific (NSE provides this data; indicates genuine buying).
References:
    - pandas-ta: https://github.com/twopirllc/pandas-ta
    - RSI: Wilder (1978), "New Concepts in Technical Trading Systems"
    - MACD: Appel (1979)
    - Bollinger Bands: Bollinger (2001), "Bollinger on Bollinger Bands"
Author: HRL-SARP Framework
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Try to import pandas_ta; fallback to manual computation ──────────
try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False
    logger.warning("pandas-ta not installed; using manual indicator computation.")


class TechnicalFeatures:
    """
    Computes vectorised technical indicators for stock price data.

    All indicators are computed from OHLCV data and normalised for neural
    network consumption (the Micro agent's 22D stock feature vector).

    Indicators:
        - RSI (14-period): Momentum oscillator, 0–100 scale
        - MACD (12, 26, 9): Trend strength and direction
        - Bollinger Bands (20, 2): Volatility and mean reversion
        - Volume Ratio: Current volume vs 20-day average
        - Delivery %: India-specific, genuine buying indicator
        - Returns: 1D, 5D, 20D log returns
        - Volatility: 20-day rolling standard deviation of returns

    Design: All methods are static/classmethod for stateless, testable computation.
    """

    # ══════════════════════════════════════════════════════════════════
    # RSI — Relative Strength Index
    # ══════════════════════════════════════════════════════════════════
    @staticmethod
    def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """
        Compute RSI (Relative Strength Index).

        RSI = 100 - 100 / (1 + RS), where RS = avg_gain / avg_loss.
        Wilder's smoothed moving average is used (exponential with α = 1/period).

        Interpretation in Indian equity context:
            - RSI > 70: Overbought (potential mean reversion down)
            - RSI < 30: Oversold (potential mean reversion up)
            - RSI 40–60: Neutral / trending

        Args:
            close: Close price series.
            period: RSI lookback period (default 14).
        Returns:
            pd.Series of RSI values (0–100 scale).
        """
        if HAS_PANDAS_TA:
            return ta.rsi(close, length=period)

        # Manual computation using Wilder's smoothing
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        # Wilder's smoothed average (exponential, α = 1/period)
        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    # ══════════════════════════════════════════════════════════════════
    # MACD — Moving Average Convergence Divergence
    # ══════════════════════════════════════════════════════════════════
    @staticmethod
    def compute_macd(
        close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> pd.DataFrame:
        """
        Compute MACD line, signal line, and histogram.

        MACD = EMA(fast) - EMA(slow)
        Signal = EMA(MACD, signal_period)
        Histogram = MACD - Signal

        The MACD-Signal crossover is used as a trend direction feature.
        We normalise the histogram by the close price for cross-stock comparability.

        Args:
            close: Close price series.
            fast: Fast EMA period (default 12).
            slow: Slow EMA period (default 26).
            signal: Signal EMA period (default 9).
        Returns:
            pd.DataFrame with columns [macd, macd_signal, macd_hist].
        """
        if HAS_PANDAS_TA:
            result = ta.macd(close, fast=fast, slow=slow, signal=signal)
            if result is not None and not result.empty:
                result.columns = ["macd", "macd_hist", "macd_signal"]
                return result

        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return pd.DataFrame({
            "macd": macd_line, "macd_signal": signal_line, "macd_hist": histogram,
        })

    # ══════════════════════════════════════════════════════════════════
    # BOLLINGER BANDS
    # ══════════════════════════════════════════════════════════════════
    @staticmethod
    def compute_bollinger_bands(
        close: pd.Series, period: int = 20, std_dev: float = 2.0
    ) -> pd.DataFrame:
        """
        Compute Bollinger Bands and the %B position indicator.

        Upper Band = SMA(period) + std_dev * StdDev(period)
        Lower Band = SMA(period) - std_dev * StdDev(period)
        %B = (Close - Lower) / (Upper - Lower)

        %B ∈ [0, 1] indicates where price sits within the bands:
            - %B > 1.0: Breakout above upper band (strong momentum)
            - %B ≈ 0.5: At the moving average (neutral)
            - %B < 0.0: Breakout below lower band (extreme weakness)

        We use %B as the normalised feature (scale-invariant across stocks).

        Args:
            close: Close price series.
            period: SMA lookback period.
            std_dev: Number of standard deviations for band width.
        Returns:
            pd.DataFrame with columns [bb_upper, bb_middle, bb_lower, bb_position].
        """
        sma = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()

        upper = sma + std_dev * std
        lower = sma - std_dev * std

        # %B: Normalised position within bands (0 = lower, 1 = upper)
        band_width = upper - lower
        bb_position = (close - lower) / band_width.replace(0, np.nan)

        return pd.DataFrame({
            "bb_upper": upper, "bb_middle": sma, "bb_lower": lower,
            "bb_position": bb_position.clip(lower=-0.5, upper=1.5),
        })

    # ══════════════════════════════════════════════════════════════════
    # VOLUME RATIO
    # ══════════════════════════════════════════════════════════════════
    @staticmethod
    def compute_volume_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
        """
        Compute volume ratio: current volume / average volume.

        Volume Ratio > 2.0: Unusual activity (breakout or breakdown in progress).
        Volume Ratio < 0.5: Low interest (consolidation phase).

        In Indian markets, high volume + high delivery % confirms institutional buying.

        Args:
            volume: Volume series.
            period: Average volume lookback period.
        Returns:
            pd.Series of volume ratios.
        """
        avg_volume = volume.rolling(window=period).mean()
        ratio = volume / avg_volume.replace(0, np.nan)
        return ratio.clip(upper=5.0)  # Cap at 5x to limit outlier impact

    # ══════════════════════════════════════════════════════════════════
    # RETURNS
    # ══════════════════════════════════════════════════════════════════
    @staticmethod
    def compute_returns(close: pd.Series) -> pd.DataFrame:
        """
        Compute multi-horizon log returns.

        Log returns are preferred because:
            1. They are time-additive (sum of daily = period return)
            2. They are approximately normally distributed
            3. They handle zero-crossing gracefully

        Returns:
            pd.DataFrame with columns [return_1d, return_5d, return_20d].
        """
        return pd.DataFrame({
            "return_1d": np.log(close / close.shift(1)),
            "return_5d": np.log(close / close.shift(5)),
            "return_20d": np.log(close / close.shift(20)),
        })

    # ══════════════════════════════════════════════════════════════════
    # VOLATILITY
    # ══════════════════════════════════════════════════════════════════
    @staticmethod
    def compute_volatility(close: pd.Series, period: int = 20) -> pd.Series:
        """
        Compute rolling volatility (annualised standard deviation of daily returns).

        Annualisation: σ_annual = σ_daily * √252 (252 trading days in India).
        Used as a risk feature in position sizing and regime detection.

        Args:
            close: Close price series.
            period: Rolling window for volatility computation.
        Returns:
            pd.Series of annualised volatility.
        """
        daily_returns = np.log(close / close.shift(1))
        rolling_std = daily_returns.rolling(window=period).std()
        # Annualise: multiply by sqrt(252) for Indian trading days
        return rolling_std * np.sqrt(252)

    # ══════════════════════════════════════════════════════════════════
    # MASTER COMPUTATION
    # ══════════════════════════════════════════════════════════════════
    @classmethod
    def compute_all(cls, ohlcv: pd.DataFrame,
                    delivery_pct: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Compute all technical features from OHLCV data.

        This is the main entry point. Returns a DataFrame with all technical
        features aligned to the input's DatetimeIndex.

        Args:
            ohlcv: DataFrame with columns [Open, High, Low, Close, Volume].
            delivery_pct: Optional delivery volume percentage series (India-specific).
        Returns:
            pd.DataFrame with all technical feature columns.
        """
        close = ohlcv["Close"]
        volume = ohlcv["Volume"]

        features = pd.DataFrame(index=ohlcv.index)

        # ── Returns ──────────────────────────────────────────────────
        returns_df = cls.compute_returns(close)
        features = features.join(returns_df)

        # ── Volatility ───────────────────────────────────────────────
        features["volatility_20d"] = cls.compute_volatility(close, period=20)

        # ── RSI ──────────────────────────────────────────────────────
        features["rsi_14"] = cls.compute_rsi(close, period=14)
        # Normalise RSI to [-1, 1] range for neural network
        features["rsi_14_norm"] = (features["rsi_14"] - 50.0) / 50.0

        # ── MACD ─────────────────────────────────────────────────────
        macd_df = cls.compute_macd(close)
        # Normalise MACD signal by close price for cross-stock comparability
        features["macd_signal"] = macd_df["macd_hist"] / close.replace(0, np.nan)

        # ── Bollinger Bands ──────────────────────────────────────────
        bb_df = cls.compute_bollinger_bands(close)
        features["bb_position"] = bb_df["bb_position"]

        # ── Volume Ratio ─────────────────────────────────────────────
        features["volume_ratio_20d"] = cls.compute_volume_ratio(volume)

        # ── Delivery Volume % (India-specific) ───────────────────────
        if delivery_pct is not None:
            features["delivery_pct"] = delivery_pct
        else:
            # If not available, use NaN (will be imputed downstream)
            features["delivery_pct"] = np.nan

        logger.debug("Computed %d technical features with %d rows", len(features.columns), len(features))
        return features
