"""
File: evaluator.py
Module: evaluation
Description: Comprehensive evaluation pipeline that orchestrates backtesting,
    benchmark comparison, stress testing, and explainability analysis into a
    single end-to-end evaluation run. Produces structured results for the
    report generator.
Design Decisions: Acts as the top-level orchestrator. Runs evaluation in stages:
    (1) Backtest → (2) Benchmark comparison → (3) Stress tests → (4) Regime-
    conditional analysis → (5) Ablation studies. Each stage is independently
    callable and cacheable.
References: CSCW paper evaluation protocol, ML reproducibility checklist
Author: HRL-SARP Framework
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np

from backtest.backtester import Backtester
from backtest.benchmark_comparison import BenchmarkComparison
from backtest.performance_metrics import PerformanceMetrics
from risk.stress_testing import StressTester

logger = logging.getLogger(__name__)


class Evaluator:
    """End-to-end evaluation orchestrator for the HRL-SARP system."""

    def __init__(
        self,
        risk_free_rate: float = 0.07,
        initial_capital: float = 10_000_000.0,
        output_dir: str = "results",
    ) -> None:
        self.risk_free_rate = risk_free_rate
        self.initial_capital = initial_capital
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.backtester = Backtester(
            risk_free_rate=risk_free_rate,
            initial_capital=initial_capital,
        )
        self.benchmark = BenchmarkComparison(risk_free_rate=risk_free_rate)
        self.stress_tester = StressTester()
        self.metrics_calc = PerformanceMetrics(risk_free_rate=risk_free_rate)

        self.results: Dict[str, Any] = {}
        logger.info("Evaluator initialised | output_dir=%s", output_dir)

    # ── Full Evaluation Pipeline ─────────────────────────────────────

    def run_full_evaluation(
        self,
        env,
        macro_agent=None,
        micro_agent=None,
        asset_returns: Optional[np.ndarray] = None,
        nifty_returns: Optional[np.ndarray] = None,
        regime_labels: Optional[np.ndarray] = None,
        n_backtest_episodes: int = 5,
    ) -> Dict[str, Any]:
        """Run the complete evaluation pipeline.

        Stages:
            1. Walk-forward backtest
            2. Benchmark comparison
            3. Stress testing
            4. Regime-conditional analysis
            5. Robustness checks

        Returns:
            Comprehensive evaluation results dict.
        """
        logger.info("=" * 60)
        logger.info("STARTING FULL EVALUATION PIPELINE")
        logger.info("=" * 60)

        # Stage 1: Backtest
        logger.info("Stage 1/5: Walk-forward backtest")
        backtest_results = self.backtester.run(
            env, macro_agent, micro_agent,
            n_episodes=n_backtest_episodes,
        )
        self.results["backtest"] = backtest_results

        # Stage 2: Benchmark comparison
        hrl_returns = np.array(self.backtester.portfolio_returns)
        if asset_returns is not None and nifty_returns is not None:
            logger.info("Stage 2/5: Benchmark comparison")
            benchmark_results = self.benchmark.compare_all(
                hrl_returns, asset_returns, nifty_returns,
            )
            self.results["benchmark"] = benchmark_results
            logger.info("\n%s", self.benchmark.summary_table(benchmark_results))

        # Stage 3: Stress testing
        logger.info("Stage 3/5: Stress testing")
        sector_weights = self._get_latest_sector_weights()
        stress_results = self.stress_tester.run_all(
            sector_weights, self.initial_capital,
        )
        stress_report = self.stress_tester.generate_report(stress_results)
        self.results["stress_test"] = stress_report

        # Stage 4: Regime-conditional analysis
        if regime_labels is not None:
            logger.info("Stage 4/5: Regime-conditional analysis")
            regime_results = self._regime_conditional_analysis(
                hrl_returns, regime_labels,
            )
            self.results["regime_analysis"] = regime_results

        # Stage 5: Robustness checks
        logger.info("Stage 5/5: Robustness analysis")
        robustness = self._robustness_analysis(hrl_returns)
        self.results["robustness"] = robustness

        # Save results
        self._save_results()

        logger.info("=" * 60)
        logger.info("EVALUATION COMPLETE")
        logger.info("=" * 60)

        return self.results

    # ── Stage 4: Regime-Conditional Analysis ─────────────────────────

    def _regime_conditional_analysis(
        self,
        returns: np.ndarray,
        regime_labels: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """Analyse performance conditional on market regime."""
        min_len = min(len(returns), len(regime_labels))
        returns = returns[:min_len]
        regime_labels = regime_labels[:min_len]

        regime_names = {0: "Bull", 1: "Bear", 2: "Sideways"}
        results = {}

        for regime_id, regime_name in regime_names.items():
            mask = regime_labels == regime_id
            if mask.sum() < 10:
                continue

            regime_returns = returns[mask]
            metrics = self.metrics_calc.compute_all(regime_returns)
            metrics["n_days"] = int(mask.sum())
            metrics["pct_of_total"] = float(mask.mean())
            results[regime_name] = metrics

        return results

    # ── Stage 5: Robustness Analysis ─────────────────────────────────

    def _robustness_analysis(
        self,
        returns: np.ndarray,
    ) -> Dict[str, Any]:
        """Run robustness checks on HRL performance."""
        results = {}

        # 1. Rolling Sharpe (6-month window)
        window = 126
        rolling_sharpes = []
        for i in range(window, len(returns)):
            window_returns = returns[i - window:i]
            s = self.metrics_calc.sharpe_ratio(window_returns)
            rolling_sharpes.append(s)

        if rolling_sharpes:
            results["rolling_sharpe"] = {
                "mean": float(np.mean(rolling_sharpes)),
                "std": float(np.std(rolling_sharpes)),
                "min": float(np.min(rolling_sharpes)),
                "max": float(np.max(rolling_sharpes)),
                "pct_positive": float(np.mean(np.array(rolling_sharpes) > 0)),
            }

        # 2. Year-by-year performance
        days_per_year = 252
        yearly_results = {}
        for year_idx in range(len(returns) // days_per_year):
            start = year_idx * days_per_year
            end = start + days_per_year
            year_returns = returns[start:end]
            yearly_results[f"year_{year_idx + 1}"] = self.metrics_calc.compute_all(
                year_returns
            )
        results["yearly"] = yearly_results

        # 3. Worst periods analysis
        worst_month_return = float("inf")
        best_month_return = float("-inf")
        for start in range(0, len(returns), 21):
            end = min(start + 21, len(returns))
            month_ret = float(np.prod(1 + returns[start:end]) - 1)
            worst_month_return = min(worst_month_return, month_ret)
            best_month_return = max(best_month_return, month_ret)

        results["extremes"] = {
            "worst_month": worst_month_return if worst_month_return != float("inf") else 0.0,
            "best_month": best_month_return if best_month_return != float("-inf") else 0.0,
        }

        # 4. Tail analysis
        p5 = float(np.percentile(returns, 5))
        p95 = float(np.percentile(returns, 95))
        results["tail_analysis"] = {
            "left_tail_5pct": p5,
            "right_tail_95pct": p95,
            "tail_ratio": abs(p95 / p5) if abs(p5) > 1e-10 else float("inf"),
        }

        return results

    # ── Helpers ──────────────────────────────────────────────────────

    def _get_latest_sector_weights(self) -> np.ndarray:
        """Get latest sector weights from backtest history."""
        sw = self.backtester.sector_weights_history
        if sw:
            return sw[-1]
        return np.ones(11) / 11  # Equal weight fallback

    def _save_results(self) -> None:
        """Save results to JSON."""
        path = os.path.join(self.output_dir, "evaluation_results.json")

        # Convert numpy types for JSON serialisation
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            return str(obj)

        with open(path, "w") as f:
            json.dump(self.results, f, indent=2, default=convert)

        logger.info("Evaluation results saved to %s", path)
