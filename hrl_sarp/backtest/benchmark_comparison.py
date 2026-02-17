"""
File: benchmark_comparison.py
Module: backtest
Description: Compare HRL-SARP portfolio against standard benchmarks and baseline
    strategies. Includes Nifty 50, equal-weight, momentum, minimum variance,
    and risk parity strategies.
Design Decisions: Each benchmark is implemented as a simple, self-contained strategy
    so results are directly comparable. Statistical significance tests (paired t-test,
    bootstrap) verify whether HRL outperformance is robust.
References: DeMiguel et al., "1/N" (RFS 2009), Asness "Value and Momentum" (JFE 2013)
Author: HRL-SARP Framework
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from backtest.performance_metrics import PerformanceMetrics

logger = logging.getLogger(__name__)


class BenchmarkComparison:
    """Compare HRL-SARP against benchmark strategies."""

    def __init__(
        self,
        risk_free_rate: float = 0.07,
        trading_days: int = 252,
    ) -> None:
        self.metrics_calc = PerformanceMetrics(
            risk_free_rate=risk_free_rate,
            trading_days=trading_days,
        )

    # ── Benchmark Strategies ─────────────────────────────────────────

    def equal_weight_returns(
        self,
        asset_returns: np.ndarray,
    ) -> np.ndarray:
        """1/N equal-weight portfolio returns.

        Args:
            asset_returns: (T, N) return matrix.
        Returns:
            (T,) portfolio return series.
        """
        n = asset_returns.shape[1]
        weights = np.ones(n) / n
        return asset_returns @ weights

    def momentum_returns(
        self,
        asset_returns: np.ndarray,
        lookback: int = 60,
        top_k: int = 5,
        rebalance_freq: int = 21,
    ) -> np.ndarray:
        """Cross-sectional momentum strategy.

        Select top-k performers over lookback period, rebalance monthly.
        """
        T, N = asset_returns.shape
        portfolio_returns = np.zeros(T)
        weights = np.ones(N) / N

        for t in range(T):
            portfolio_returns[t] = float(np.dot(weights, asset_returns[t]))

            if t >= lookback and t % rebalance_freq == 0:
                # Rank by cumulative return over lookback
                cum_ret = np.prod(1 + asset_returns[t - lookback:t], axis=0) - 1
                top_idx = np.argsort(-cum_ret)[:top_k]
                weights = np.zeros(N)
                weights[top_idx] = 1.0 / top_k

        return portfolio_returns

    def minimum_variance_returns(
        self,
        asset_returns: np.ndarray,
        lookback: int = 126,
        rebalance_freq: int = 21,
    ) -> np.ndarray:
        """Global minimum variance portfolio.

        Uses sample covariance matrix with shrinkage to Diagonal (Ledoit-Wolf).
        """
        T, N = asset_returns.shape
        portfolio_returns = np.zeros(T)
        weights = np.ones(N) / N

        for t in range(T):
            portfolio_returns[t] = float(np.dot(weights, asset_returns[t]))

            if t >= lookback and t % rebalance_freq == 0:
                hist = asset_returns[t - lookback:t]
                cov = np.cov(hist.T)

                # Ledoit-Wolf shrinkage
                trace = np.trace(cov)
                shrinkage = 0.3
                cov_shrunk = (1 - shrinkage) * cov + shrinkage * (trace / N) * np.eye(N)

                try:
                    inv_cov = np.linalg.inv(cov_shrunk)
                    ones = np.ones(N)
                    weights = inv_cov @ ones
                    weights /= weights.sum()
                    weights = np.clip(weights, 0, 0.35)
                    weights /= weights.sum()
                except np.linalg.LinAlgError:
                    weights = np.ones(N) / N

        return portfolio_returns

    def risk_parity_returns(
        self,
        asset_returns: np.ndarray,
        lookback: int = 63,
        rebalance_freq: int = 21,
    ) -> np.ndarray:
        """Risk parity (inverse volatility) portfolio."""
        T, N = asset_returns.shape
        portfolio_returns = np.zeros(T)
        weights = np.ones(N) / N

        for t in range(T):
            portfolio_returns[t] = float(np.dot(weights, asset_returns[t]))

            if t >= lookback and t % rebalance_freq == 0:
                hist = asset_returns[t - lookback:t]
                vols = np.std(hist, axis=0, ddof=1)
                vols = np.clip(vols, 1e-8, None)
                inv_vol = 1.0 / vols
                weights = inv_vol / inv_vol.sum()

        return portfolio_returns

    def buy_and_hold_nifty(
        self,
        nifty_returns: np.ndarray,
    ) -> np.ndarray:
        """Buy-and-hold Nifty 50 benchmark."""
        return nifty_returns.copy()

    # ── Full Comparison ──────────────────────────────────────────────

    def compare_all(
        self,
        hrl_returns: np.ndarray,
        asset_returns: np.ndarray,
        nifty_returns: np.ndarray,
    ) -> Dict[str, Dict[str, Any]]:
        """Compare HRL-SARP against all benchmark strategies.

        Args:
            hrl_returns: (T,) HRL-SARP daily returns.
            asset_returns: (T, N) sector/stock return matrix.
            nifty_returns: (T,) Nifty 50 daily returns.

        Returns:
            Dict[strategy_name → metrics_dict]
        """
        strategies = {
            "HRL-SARP": hrl_returns,
            "Nifty50_BuyHold": self.buy_and_hold_nifty(nifty_returns),
            "Equal_Weight": self.equal_weight_returns(asset_returns),
            "Momentum_60_5": self.momentum_returns(asset_returns),
            "Min_Variance": self.minimum_variance_returns(asset_returns),
            "Risk_Parity": self.risk_parity_returns(asset_returns),
        }

        results = {}
        for name, returns in strategies.items():
            metrics = self.metrics_calc.compute_all(returns, benchmark_returns=nifty_returns)

            # Add statistical test vs HRL
            if name != "HRL-SARP":
                stat_test = self.statistical_test(hrl_returns, returns)
                metrics["vs_hrl"] = stat_test

            results[name] = metrics

        return results

    # ── Statistical Significance ─────────────────────────────────────

    def statistical_test(
        self,
        hrl_returns: np.ndarray,
        benchmark_returns: np.ndarray,
    ) -> Dict[str, float]:
        """Test statistical significance of HRL outperformance.

        Uses paired t-test and bootstrap confidence interval on Sharpe difference.
        """
        min_len = min(len(hrl_returns), len(benchmark_returns))
        hrl = hrl_returns[:min_len]
        bench = benchmark_returns[:min_len]
        diff = hrl - bench

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(hrl, bench)

        # Bootstrap Sharpe difference
        n_bootstrap = 1000
        sharpe_diffs = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(min_len, size=min_len, replace=True)
            hrl_sharpe = np.mean(hrl[idx]) / (np.std(hrl[idx]) + 1e-10)
            bench_sharpe = np.mean(bench[idx]) / (np.std(bench[idx]) + 1e-10)
            sharpe_diffs.append(hrl_sharpe - bench_sharpe)

        sharpe_diffs = np.array(sharpe_diffs)

        return {
            "mean_excess_return": float(np.mean(diff)),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant_5pct": p_value < 0.05,
            "significant_1pct": p_value < 0.01,
            "sharpe_diff_mean": float(np.mean(sharpe_diffs)),
            "sharpe_diff_ci_lower": float(np.percentile(sharpe_diffs, 2.5)),
            "sharpe_diff_ci_upper": float(np.percentile(sharpe_diffs, 97.5)),
        }

    # ── Summary Table ────────────────────────────────────────────────

    def summary_table(
        self,
        results: Dict[str, Dict[str, Any]],
    ) -> str:
        """Generate formatted comparison table."""
        header = (
            f"{'Strategy':<20} {'CAGR':>8} {'Sharpe':>8} {'Sortino':>8} "
            f"{'MaxDD':>8} {'Calmar':>8} {'Alpha':>8}"
        )
        separator = "-" * len(header)

        lines = [header, separator]
        for name, metrics in results.items():
            line = (
                f"{name:<20} "
                f"{metrics.get('cagr', 0) * 100:>7.2f}% "
                f"{metrics.get('sharpe_ratio', 0):>8.3f} "
                f"{metrics.get('sortino_ratio', 0):>8.3f} "
                f"{metrics.get('max_drawdown', 0) * 100:>7.2f}% "
                f"{metrics.get('calmar_ratio', 0):>8.3f} "
                f"{metrics.get('alpha', 0) * 100:>7.2f}%"
            )
            lines.append(line)

        return "\n".join(lines)
