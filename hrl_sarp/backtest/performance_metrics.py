"""
File: performance_metrics.py
Module: backtest
Description: Comprehensive performance metrics calculator for portfolio backtesting.
    Computes Sharpe, Sortino, Calmar, Max Drawdown, VaR/CVaR, Information Ratio,
    Treynor ratio, and India-specific metrics (rolling Nifty alpha, FII flow correlation).
Design Decisions: All metrics computed from return series for consistency.
    Annualisation uses 252 trading days (NSE calendar). Risk-free rate defaults
    to India 10Y G-Sec yield.
References: Bacon (2008) "Practical Portfolio Performance", SEBI benchmarking
Author: HRL-SARP Framework
"""

import logging
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """Compute comprehensive portfolio performance metrics."""

    def __init__(
        self,
        risk_free_rate: float = 0.07,
        trading_days: int = 252,
    ) -> None:
        self.rf = risk_free_rate
        self.daily_rf = (1 + risk_free_rate) ** (1 / trading_days) - 1
        self.trading_days = trading_days

    def compute_all(
        self,
        portfolio_returns: np.ndarray,
        benchmark_returns: Optional[np.ndarray] = None,
        portfolio_values: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Compute all performance metrics.

        Args:
            portfolio_returns: (T,) daily return series.
            benchmark_returns: (T,) benchmark return series.
            portfolio_values: (T+1,) portfolio value series.

        Returns:
            Dict of metric_name → value.
        """
        metrics = {}

        # Return metrics
        metrics["total_return"] = self.total_return(portfolio_returns)
        metrics["cagr"] = self.cagr(portfolio_returns)
        metrics["annualised_return"] = self.annualised_return(portfolio_returns)
        metrics["annualised_volatility"] = self.annualised_volatility(portfolio_returns)

        # Risk-adjusted returns
        metrics["sharpe_ratio"] = self.sharpe_ratio(portfolio_returns)
        metrics["sortino_ratio"] = self.sortino_ratio(portfolio_returns)
        metrics["calmar_ratio"] = self.calmar_ratio(portfolio_returns)

        # Drawdown
        dd_stats = self.drawdown_stats(portfolio_returns)
        metrics.update(dd_stats)

        # Risk metrics
        metrics["var_95"] = self.value_at_risk(portfolio_returns, 0.95)
        metrics["cvar_95"] = self.conditional_var(portfolio_returns, 0.95)
        metrics["var_99"] = self.value_at_risk(portfolio_returns, 0.99)
        metrics["cvar_99"] = self.conditional_var(portfolio_returns, 0.99)

        # Higher moments
        metrics["skewness"] = self.skewness(portfolio_returns)
        metrics["kurtosis"] = self.kurtosis(portfolio_returns)

        # Win/loss
        metrics["win_rate"] = self.win_rate(portfolio_returns)
        metrics["profit_factor"] = self.profit_factor(portfolio_returns)

        # Benchmark-relative
        if benchmark_returns is not None:
            metrics["alpha"] = self.alpha(portfolio_returns, benchmark_returns)
            metrics["beta"] = self.beta(portfolio_returns, benchmark_returns)
            metrics["information_ratio"] = self.information_ratio(
                portfolio_returns, benchmark_returns
            )
            metrics["treynor_ratio"] = self.treynor_ratio(
                portfolio_returns, benchmark_returns
            )
            metrics["tracking_error"] = self.tracking_error(
                portfolio_returns, benchmark_returns
            )
            metrics["max_relative_drawdown"] = self.max_relative_drawdown(
                portfolio_returns, benchmark_returns
            )

        return metrics

    # ── Return Metrics ───────────────────────────────────────────────

    def total_return(self, returns: np.ndarray) -> float:
        return float(np.prod(1 + returns) - 1)

    def cagr(self, returns: np.ndarray) -> float:
        total = np.prod(1 + returns)
        n_years = len(returns) / self.trading_days
        if n_years <= 0 or total <= 0:
            return 0.0
        return float(total ** (1 / n_years) - 1)

    def annualised_return(self, returns: np.ndarray) -> float:
        return float(np.mean(returns) * self.trading_days)

    def annualised_volatility(self, returns: np.ndarray) -> float:
        return float(np.std(returns, ddof=1) * np.sqrt(self.trading_days))

    # ── Risk-Adjusted Returns ────────────────────────────────────────

    def sharpe_ratio(self, returns: np.ndarray) -> float:
        excess = returns - self.daily_rf
        std = np.std(excess, ddof=1)
        if std < 1e-10:
            return 0.0
        return float(np.mean(excess) / std * np.sqrt(self.trading_days))

    def sortino_ratio(self, returns: np.ndarray) -> float:
        excess = returns - self.daily_rf
        downside = excess[excess < 0]
        if len(downside) == 0:
            return float("inf")
        downside_std = np.std(downside, ddof=1)
        if downside_std < 1e-10:
            return 0.0
        return float(np.mean(excess) / downside_std * np.sqrt(self.trading_days))

    def calmar_ratio(self, returns: np.ndarray) -> float:
        ann_ret = self.cagr(returns)
        dd = self.drawdown_stats(returns)
        max_dd = dd["max_drawdown"]
        if max_dd < 1e-10:
            return 0.0
        return float(ann_ret / max_dd)

    # ── Drawdown ─────────────────────────────────────────────────────

    def drawdown_stats(self, returns: np.ndarray) -> Dict[str, float]:
        cumulative = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative) / peak

        if len(drawdown) == 0:
            return {"max_drawdown": 0.0, "avg_drawdown": 0.0, "max_dd_duration": 0}

        max_dd = float(np.max(drawdown))
        avg_dd = float(np.mean(drawdown))

        # Max drawdown duration
        in_dd = drawdown > 0
        max_duration = 0
        current_duration = 0
        for d in in_dd:
            if d:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0

        return {
            "max_drawdown": max_dd,
            "avg_drawdown": avg_dd,
            "max_dd_duration": max_duration,
        }

    # ── Risk Metrics ─────────────────────────────────────────────────

    def value_at_risk(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        alpha = 1.0 - confidence
        return float(-np.percentile(returns, alpha * 100))

    def conditional_var(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        var = self.value_at_risk(returns, confidence)
        tail = returns[returns <= -var]
        if len(tail) == 0:
            return var
        return float(-np.mean(tail))

    # ── Higher Moments ───────────────────────────────────────────────

    def skewness(self, returns: np.ndarray) -> float:
        n = len(returns)
        if n < 3:
            return 0.0
        mean = np.mean(returns)
        std = np.std(returns, ddof=1)
        if std < 1e-10:
            return 0.0
        return float((n / ((n - 1) * (n - 2))) * np.sum(((returns - mean) / std) ** 3))

    def kurtosis(self, returns: np.ndarray) -> float:
        n = len(returns)
        if n < 4:
            return 0.0
        mean = np.mean(returns)
        std = np.std(returns, ddof=1)
        if std < 1e-10:
            return 0.0
        k = float(np.mean(((returns - mean) / std) ** 4))
        return k - 3.0  # Excess kurtosis

    # ── Win/Loss ─────────────────────────────────────────────────────

    def win_rate(self, returns: np.ndarray) -> float:
        if len(returns) == 0:
            return 0.0
        return float((returns > 0).sum() / len(returns))

    def profit_factor(self, returns: np.ndarray) -> float:
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        if losses < 1e-10:
            return float("inf")
        return float(gains / losses)

    # ── Benchmark-Relative ───────────────────────────────────────────

    def alpha(
        self, returns: np.ndarray, benchmark_returns: np.ndarray
    ) -> float:
        b = self.beta(returns, benchmark_returns)
        port_ann = self.annualised_return(returns)
        bench_ann = self.annualised_return(benchmark_returns)
        # Jensen's alpha: R_p - [R_f + β(R_m - R_f)]
        return float(port_ann - (self.rf + b * (bench_ann - self.rf)))

    def beta(
        self, returns: np.ndarray, benchmark_returns: np.ndarray
    ) -> float:
        if len(returns) != len(benchmark_returns):
            min_len = min(len(returns), len(benchmark_returns))
            returns = returns[:min_len]
            benchmark_returns = benchmark_returns[:min_len]
        cov = np.cov(returns, benchmark_returns)[0, 1]
        var = np.var(benchmark_returns, ddof=1)
        if var < 1e-10:
            return 0.0
        return float(cov / var)

    def information_ratio(
        self, returns: np.ndarray, benchmark_returns: np.ndarray
    ) -> float:
        te = self.tracking_error(returns, benchmark_returns)
        if te < 1e-10:
            return 0.0
        active = returns[:len(benchmark_returns)] - benchmark_returns[:len(returns)]
        return float(np.mean(active) / te * np.sqrt(self.trading_days))

    def treynor_ratio(
        self, returns: np.ndarray, benchmark_returns: np.ndarray
    ) -> float:
        b = self.beta(returns, benchmark_returns)
        if abs(b) < 1e-10:
            return 0.0
        excess_return = self.annualised_return(returns) - self.rf
        return float(excess_return / b)

    def tracking_error(
        self, returns: np.ndarray, benchmark_returns: np.ndarray
    ) -> float:
        min_len = min(len(returns), len(benchmark_returns))
        active = returns[:min_len] - benchmark_returns[:min_len]
        return float(np.std(active, ddof=1))

    def max_relative_drawdown(
        self, returns: np.ndarray, benchmark_returns: np.ndarray
    ) -> float:
        min_len = min(len(returns), len(benchmark_returns))
        active = returns[:min_len] - benchmark_returns[:min_len]
        return self.drawdown_stats(active)["max_drawdown"]
