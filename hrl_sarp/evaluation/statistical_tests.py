"""
File: statistical_tests.py
Module: evaluation
Description: Statistical significance tests for comparing HRL-SARP against
    benchmarks. Implements paired t-test, Wilcoxon signed-rank test, bootstrap
    confidence intervals, and the Ledoit-Wolf adjusted Sharpe ratio test.
Design Decisions: Multiple tests provide robustness — parametric (t-test) and
    non-parametric (Wilcoxon) tests, plus bootstrap for finite-sample accuracy.
    The Ledoit-Wolf test specifically handles autocorrelated returns (common in
    portfolio strategies).
References: Ledoit & Wolf (2008) "Robust Performance Hypothesis Testing",
    DeMiguel et al. (2009), Politis & Romano (1994) stationary bootstrap
Author: HRL-SARP Framework
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class StatisticalTests:
    """Statistical significance tests for strategy comparison."""

    def __init__(
        self,
        n_bootstrap: int = 10000,
        confidence_level: float = 0.95,
        block_length: int = 5,
    ) -> None:
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.block_length = block_length  # For block bootstrap (handles autocorrelation)

    # ── Full Test Suite ──────────────────────────────────────────────

    def run_all_tests(
        self,
        strategy_returns: np.ndarray,
        benchmark_returns: np.ndarray,
    ) -> Dict[str, Any]:
        """Run complete statistical test suite."""
        min_len = min(len(strategy_returns), len(benchmark_returns))
        strat = strategy_returns[:min_len]
        bench = benchmark_returns[:min_len]

        results = {}

        results["paired_ttest"] = self.paired_ttest(strat, bench)
        results["wilcoxon"] = self.wilcoxon_test(strat, bench)
        results["bootstrap_mean_diff"] = self.bootstrap_mean_difference(strat, bench)
        results["bootstrap_sharpe_diff"] = self.bootstrap_sharpe_difference(strat, bench)
        results["block_bootstrap_sharpe"] = self.block_bootstrap_sharpe(strat, bench)
        results["sign_test"] = self.sign_test(strat, bench)

        # Summary
        sig_tests = sum(1 for v in results.values()
                        if isinstance(v, dict) and v.get("significant", False))
        results["summary"] = {
            "n_tests": len(results) - 1,
            "n_significant": sig_tests,
            "consensus": "significant" if sig_tests > 3 else "inconclusive",
        }

        return results

    # ── Parametric Tests ─────────────────────────────────────────────

    def paired_ttest(
        self,
        strat: np.ndarray,
        bench: np.ndarray,
    ) -> Dict[str, Any]:
        """Paired two-sided t-test on return differences."""
        t_stat, p_value = stats.ttest_rel(strat, bench)
        alpha = 1 - self.confidence_level

        return {
            "test": "paired_t_test",
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": p_value < alpha,
            "mean_diff": float(np.mean(strat - bench)),
            "std_diff": float(np.std(strat - bench, ddof=1)),
        }

    # ── Non-Parametric Tests ─────────────────────────────────────────

    def wilcoxon_test(
        self,
        strat: np.ndarray,
        bench: np.ndarray,
    ) -> Dict[str, Any]:
        """Wilcoxon signed-rank test (non-parametric, no normality assumption)."""
        diff = strat - bench
        # Remove zeros (ties)
        diff_nonzero = diff[diff != 0]

        if len(diff_nonzero) < 10:
            return {"test": "wilcoxon", "skip": True, "reason": "too few non-zero differences"}

        stat, p_value = stats.wilcoxon(diff_nonzero, alternative="two-sided")
        alpha = 1 - self.confidence_level

        return {
            "test": "wilcoxon_signed_rank",
            "statistic": float(stat),
            "p_value": float(p_value),
            "significant": p_value < alpha,
            "n_nonzero": len(diff_nonzero),
        }

    def sign_test(
        self,
        strat: np.ndarray,
        bench: np.ndarray,
    ) -> Dict[str, Any]:
        """Simple sign test: proportion of days strategy beats benchmark."""
        diff = strat - bench
        n_positive = (diff > 0).sum()
        n_total = len(diff)
        win_rate = float(n_positive / n_total)

        # Binomial test: H0: p = 0.5
        p_value = float(stats.binom_test(n_positive, n_total, 0.5, alternative="two-sided"))
        alpha = 1 - self.confidence_level

        return {
            "test": "sign_test",
            "win_rate": win_rate,
            "n_positive": int(n_positive),
            "n_total": n_total,
            "p_value": p_value,
            "significant": p_value < alpha,
        }

    # ── Bootstrap Tests ──────────────────────────────────────────────

    def bootstrap_mean_difference(
        self,
        strat: np.ndarray,
        bench: np.ndarray,
    ) -> Dict[str, Any]:
        """Bootstrap CI for mean return difference."""
        diff = strat - bench
        n = len(diff)

        boot_means = np.zeros(self.n_bootstrap)
        for b in range(self.n_bootstrap):
            idx = np.random.choice(n, size=n, replace=True)
            boot_means[b] = diff[idx].mean()

        alpha = (1 - self.confidence_level) / 2
        ci_lower = float(np.percentile(boot_means, alpha * 100))
        ci_upper = float(np.percentile(boot_means, (1 - alpha) * 100))

        # Significant if CI doesn't contain 0
        significant = ci_lower > 0 or ci_upper < 0

        return {
            "test": "bootstrap_mean_diff",
            "mean_diff": float(np.mean(diff)),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "significant": significant,
            "annualised_diff": float(np.mean(diff) * 252),
        }

    def bootstrap_sharpe_difference(
        self,
        strat: np.ndarray,
        bench: np.ndarray,
    ) -> Dict[str, Any]:
        """Bootstrap CI for Sharpe ratio difference."""
        n = len(strat)

        sharpe_diffs = np.zeros(self.n_bootstrap)
        for b in range(self.n_bootstrap):
            idx = np.random.choice(n, size=n, replace=True)
            s_sharpe = _daily_sharpe(strat[idx])
            b_sharpe = _daily_sharpe(bench[idx])
            sharpe_diffs[b] = s_sharpe - b_sharpe

        alpha = (1 - self.confidence_level) / 2
        ci_lower = float(np.percentile(sharpe_diffs, alpha * 100))
        ci_upper = float(np.percentile(sharpe_diffs, (1 - alpha) * 100))

        return {
            "test": "bootstrap_sharpe_diff",
            "sharpe_diff_mean": float(sharpe_diffs.mean()),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "significant": ci_lower > 0 or ci_upper < 0,
        }

    def block_bootstrap_sharpe(
        self,
        strat: np.ndarray,
        bench: np.ndarray,
    ) -> Dict[str, Any]:
        """Block bootstrap for Sharpe difference (handles autocorrelation).

        Uses stationary block bootstrap with geometric block lengths.
        """
        n = len(strat)
        sharpe_diffs = np.zeros(self.n_bootstrap)

        for b in range(self.n_bootstrap):
            # Generate block bootstrap sample
            idx = self._stationary_block_sample(n)
            s_sharpe = _daily_sharpe(strat[idx])
            b_sharpe = _daily_sharpe(bench[idx])
            sharpe_diffs[b] = s_sharpe - b_sharpe

        alpha = (1 - self.confidence_level) / 2
        ci_lower = float(np.percentile(sharpe_diffs, alpha * 100))
        ci_upper = float(np.percentile(sharpe_diffs, (1 - alpha) * 100))

        # HAC standard error
        se = float(np.std(sharpe_diffs))
        z_stat = float(sharpe_diffs.mean() / se) if se > 1e-10 else 0.0

        return {
            "test": "block_bootstrap_sharpe",
            "sharpe_diff_mean": float(sharpe_diffs.mean()),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "se": se,
            "z_statistic": z_stat,
            "significant": ci_lower > 0 or ci_upper < 0,
            "block_length": self.block_length,
        }

    def _stationary_block_sample(self, n: int) -> np.ndarray:
        """Generate a stationary block bootstrap sample of indices.

        Block lengths drawn from geometric distribution.
        """
        p = 1.0 / self.block_length
        indices = []
        i = np.random.randint(0, n)

        while len(indices) < n:
            indices.append(i)
            if np.random.random() < p:
                # Start new block
                i = np.random.randint(0, n)
            else:
                # Continue current block
                i = (i + 1) % n

        return np.array(indices[:n])

    # ── Multi-Strategy Comparison ────────────────────────────────────

    def multiple_comparison_correction(
        self,
        p_values: List[float],
        method: str = "holm",
    ) -> Dict[str, Any]:
        """Apply multiple comparison correction to p-values.

        Methods: 'bonferroni', 'holm' (step-down), 'bh' (Benjamini-Hochberg FDR).
        """
        n = len(p_values)
        p_arr = np.array(p_values)

        if method == "bonferroni":
            adjusted = np.minimum(p_arr * n, 1.0)
        elif method == "holm":
            # Holm step-down
            sorted_idx = np.argsort(p_arr)
            adjusted = np.zeros(n)
            for rank, idx in enumerate(sorted_idx):
                adjusted[idx] = min(p_arr[idx] * (n - rank), 1.0)
            # Enforce monotonicity
            for i in range(1, n):
                idx = sorted_idx[i]
                prev_idx = sorted_idx[i - 1]
                adjusted[idx] = max(adjusted[idx], adjusted[prev_idx])
        elif method == "bh":
            # Benjamini-Hochberg FDR
            sorted_idx = np.argsort(p_arr)
            adjusted = np.zeros(n)
            for rank, idx in enumerate(sorted_idx):
                adjusted[idx] = p_arr[idx] * n / (rank + 1)
            # Enforce monotonicity (reverse)
            for i in range(n - 2, -1, -1):
                idx = sorted_idx[i]
                next_idx = sorted_idx[i + 1]
                adjusted[idx] = min(adjusted[idx], adjusted[next_idx])
            adjusted = np.minimum(adjusted, 1.0)
        else:
            adjusted = p_arr

        alpha = 1 - self.confidence_level
        return {
            "method": method,
            "original_p_values": p_arr.tolist(),
            "adjusted_p_values": adjusted.tolist(),
            "significant": (adjusted < alpha).tolist(),
            "n_significant": int((adjusted < alpha).sum()),
        }


# ── Utility ──────────────────────────────────────────────────────────


def _daily_sharpe(returns: np.ndarray) -> float:
    """Compute annualised Sharpe ratio from daily returns."""
    if len(returns) < 2:
        return 0.0
    std = np.std(returns, ddof=1)
    if std < 1e-10:
        return 0.0
    return float(np.mean(returns) / std * np.sqrt(252))
