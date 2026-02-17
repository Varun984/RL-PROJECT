"""
File: report_generator.py
Module: evaluation
Description: Generates publication-ready evaluation reports in Markdown and LaTeX.
    Combines backtest results, benchmark comparisons, stress test outcomes,
    and statistical significance tests into formatted tables and narratives.
Design Decisions: Markdown output for quick inspection, LaTeX for paper-ready tables.
    Auto-generates executive summary, key findings, and risk warnings.
References: Academic paper evaluation sections, factor model reporting standards
Author: HRL-SARP Framework
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate formatted evaluation reports."""

    def __init__(
        self,
        output_dir: str = "results/reports",
    ) -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ── Full Report Generation ───────────────────────────────────────

    def generate(
        self,
        eval_results: Dict[str, Any],
        title: str = "HRL-SARP Evaluation Report",
    ) -> str:
        """Generate complete Markdown evaluation report.

        Args:
            eval_results: Output from Evaluator.run_full_evaluation().
            title: Report title.

        Returns:
            Path to generated report file.
        """
        sections = []

        # Header
        sections.append(f"# {title}")
        sections.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

        # Executive Summary
        sections.append(self._executive_summary(eval_results))

        # Performance Metrics
        if "backtest" in eval_results:
            sections.append(self._backtest_section(eval_results["backtest"]))

        # Benchmark Comparison
        if "benchmark" in eval_results:
            sections.append(self._benchmark_section(eval_results["benchmark"]))

        # Stress Testing
        if "stress_test" in eval_results:
            sections.append(self._stress_test_section(eval_results["stress_test"]))

        # Regime Analysis
        if "regime_analysis" in eval_results:
            sections.append(self._regime_section(eval_results["regime_analysis"]))

        # Robustness
        if "robustness" in eval_results:
            sections.append(self._robustness_section(eval_results["robustness"]))

        # Risk Warnings
        sections.append(self._risk_warnings(eval_results))

        # Combine
        report = "\n\n".join(sections)

        # Save
        md_path = os.path.join(self.output_dir, "evaluation_report.md")
        with open(md_path, "w") as f:
            f.write(report)

        logger.info("Report generated: %s", md_path)
        return md_path

    # ── Executive Summary ────────────────────────────────────────────

    def _executive_summary(self, results: Dict[str, Any]) -> str:
        lines = ["## Executive Summary\n"]

        bt = results.get("backtest", {}).get("full_period", {})
        if bt:
            cagr = bt.get("cagr", 0) * 100
            sharpe = bt.get("sharpe_ratio", 0)
            max_dd = bt.get("max_drawdown", 0) * 100
            sortino = bt.get("sortino_ratio", 0)

            lines.append(f"The HRL-SARP framework achieved a **CAGR of {cagr:.2f}%** "
                         f"with a **Sharpe ratio of {sharpe:.3f}** and "
                         f"**maximum drawdown of {max_dd:.2f}%**.\n")

            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append(f"| CAGR | {cagr:.2f}% |")
            lines.append(f"| Sharpe Ratio | {sharpe:.3f} |")
            lines.append(f"| Sortino Ratio | {sortino:.3f} |")
            lines.append(f"| Max Drawdown | {max_dd:.2f}% |")
            lines.append(f"| CVaR (95%) | {bt.get('cvar_95', 0) * 100:.3f}% |")
            lines.append(f"| Win Rate | {bt.get('win_rate', 0) * 100:.1f}% |")

        stress = results.get("stress_test", {})
        if stress:
            passed = stress.get("scenarios_passed", 0)
            total = stress.get("n_scenarios", 0)
            lines.append(f"\nStress testing: **{passed}/{total}** scenarios passed.")

        return "\n".join(lines)

    # ── Backtest Section ─────────────────────────────────────────────

    def _backtest_section(self, bt_results: Dict[str, Any]) -> str:
        lines = ["## Performance Metrics\n"]

        fp = bt_results.get("full_period", {})
        if not fp:
            return "\n".join(lines)

        lines.append("### Return Profile\n")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")

        metric_labels = [
            ("total_return", "Total Return", True),
            ("cagr", "CAGR", True),
            ("annualised_return", "Annualised Return", True),
            ("annualised_volatility", "Annualised Volatility", True),
        ]
        for key, label, is_pct in metric_labels:
            val = fp.get(key, 0)
            if is_pct:
                lines.append(f"| {label} | {val * 100:.2f}% |")
            else:
                lines.append(f"| {label} | {val:.4f} |")

        lines.append("\n### Risk-Adjusted Returns\n")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        risk_metrics = [
            ("sharpe_ratio", "Sharpe Ratio"),
            ("sortino_ratio", "Sortino Ratio"),
            ("calmar_ratio", "Calmar Ratio"),
        ]
        for key, label in risk_metrics:
            lines.append(f"| {label} | {fp.get(key, 0):.3f} |")

        lines.append("\n### Drawdown Analysis\n")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Max Drawdown | {fp.get('max_drawdown', 0) * 100:.2f}% |")
        lines.append(f"| Avg Drawdown | {fp.get('avg_drawdown', 0) * 100:.2f}% |")
        lines.append(f"| Max DD Duration | {fp.get('max_dd_duration', 0)} days |")

        lines.append("\n### Tail Risk\n")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| VaR (95%) | {fp.get('var_95', 0) * 100:.3f}% |")
        lines.append(f"| CVaR (95%) | {fp.get('cvar_95', 0) * 100:.3f}% |")
        lines.append(f"| VaR (99%) | {fp.get('var_99', 0) * 100:.3f}% |")
        lines.append(f"| CVaR (99%) | {fp.get('cvar_99', 0) * 100:.3f}% |")
        lines.append(f"| Skewness | {fp.get('skewness', 0):.3f} |")
        lines.append(f"| Excess Kurtosis | {fp.get('kurtosis', 0):.3f} |")

        return "\n".join(lines)

    # ── Benchmark Section ────────────────────────────────────────────

    def _benchmark_section(self, bm_results: Dict[str, Any]) -> str:
        lines = ["## Benchmark Comparison\n"]

        # Summary table
        header = "| Strategy | CAGR | Sharpe | Sortino | Max DD | Alpha |"
        separator = "|----------|------|--------|---------|--------|-------|"
        lines.append(header)
        lines.append(separator)

        for name, metrics in bm_results.items():
            if isinstance(metrics, dict):
                lines.append(
                    f"| {name} "
                    f"| {metrics.get('cagr', 0) * 100:.2f}% "
                    f"| {metrics.get('sharpe_ratio', 0):.3f} "
                    f"| {metrics.get('sortino_ratio', 0):.3f} "
                    f"| {metrics.get('max_drawdown', 0) * 100:.2f}% "
                    f"| {metrics.get('alpha', 0) * 100:.2f}% |"
                )

        # Statistical significance
        lines.append("\n### Statistical Significance\n")
        for name, metrics in bm_results.items():
            if isinstance(metrics, dict) and "vs_hrl" in metrics:
                vs = metrics["vs_hrl"]
                sig = "✅ Significant" if vs.get("significant_5pct", False) else "❌ Not significant"
                lines.append(
                    f"- **HRL vs {name}**: p={vs.get('p_value', 1):.4f} ({sig}), "
                    f"Sharpe diff CI: [{vs.get('sharpe_diff_ci_lower', 0):.3f}, "
                    f"{vs.get('sharpe_diff_ci_upper', 0):.3f}]"
                )

        return "\n".join(lines)

    # ── Stress Test Section ──────────────────────────────────────────

    def _stress_test_section(self, stress_results: Dict[str, Any]) -> str:
        lines = ["## Stress Testing\n"]

        lines.append(
            f"**{stress_results.get('scenarios_passed', 0)}/"
            f"{stress_results.get('n_scenarios', 0)}** scenarios passed "
            f"the drawdown limit.\n"
        )

        if "scenario_details" in stress_results:
            lines.append("| Scenario | Portfolio Impact | Max DD | Passes |")
            lines.append("|----------|----------------|--------|--------|")

            for name, detail in stress_results["scenario_details"].items():
                pnl_pct = detail.get("portfolio_return", 0) * 100
                max_dd = detail.get("max_drawdown", 0) * 100
                passes = "✅" if detail.get("passes_drawdown_limit", False) else "❌"
                lines.append(
                    f"| {detail.get('description', name)[:40]} "
                    f"| {pnl_pct:.2f}% | {max_dd:.2f}% | {passes} |"
                )

        worst = stress_results.get("worst_scenario", "N/A")
        worst_pnl = stress_results.get("worst_case_pnl", 0)
        lines.append(f"\n**Worst-case scenario**: {worst} (₹{worst_pnl:,.0f} P&L)")

        return "\n".join(lines)

    # ── Regime Section ───────────────────────────────────────────────

    def _regime_section(self, regime_results: Dict[str, Any]) -> str:
        lines = ["## Regime-Conditional Performance\n"]

        lines.append("| Regime | Days | CAGR | Sharpe | Max DD |")
        lines.append("|--------|------|------|--------|--------|")

        for regime, metrics in regime_results.items():
            if isinstance(metrics, dict):
                lines.append(
                    f"| {regime} "
                    f"| {metrics.get('n_days', 0)} "
                    f"| {metrics.get('cagr', 0) * 100:.2f}% "
                    f"| {metrics.get('sharpe_ratio', 0):.3f} "
                    f"| {metrics.get('max_drawdown', 0) * 100:.2f}% |"
                )

        return "\n".join(lines)

    # ── Robustness Section ───────────────────────────────────────────

    def _robustness_section(self, robustness: Dict[str, Any]) -> str:
        lines = ["## Robustness Analysis\n"]

        # Rolling Sharpe
        rs = robustness.get("rolling_sharpe", {})
        if rs:
            lines.append("### Rolling 6-Month Sharpe Ratio\n")
            lines.append("| Statistic | Value |")
            lines.append("|-----------|-------|")
            lines.append(f"| Mean | {rs.get('mean', 0):.3f} |")
            lines.append(f"| Std | {rs.get('std', 0):.3f} |")
            lines.append(f"| Min | {rs.get('min', 0):.3f} |")
            lines.append(f"| Max | {rs.get('max', 0):.3f} |")
            lines.append(f"| % Positive | {rs.get('pct_positive', 0) * 100:.1f}% |")

        # Extremes
        extremes = robustness.get("extremes", {})
        if extremes:
            lines.append(f"\n- **Best month**: {extremes.get('best_month', 0) * 100:.2f}%")
            lines.append(f"- **Worst month**: {extremes.get('worst_month', 0) * 100:.2f}%")

        # Tail analysis
        tail = robustness.get("tail_analysis", {})
        if tail:
            lines.append(f"- **Tail ratio**: {tail.get('tail_ratio', 0):.2f} "
                          "(>1 indicates positive skew)")

        return "\n".join(lines)

    # ── Risk Warnings ────────────────────────────────────────────────

    def _risk_warnings(self, results: Dict[str, Any]) -> str:
        lines = ["## ⚠️ Risk Warnings\n"]
        warnings = []

        bt = results.get("backtest", {}).get("full_period", {})
        if bt.get("max_drawdown", 0) > 0.20:
            warnings.append(
                f"- **High drawdown risk**: Maximum drawdown of "
                f"{bt['max_drawdown'] * 100:.1f}% exceeds 20%."
            )

        if bt.get("kurtosis", 0) > 3.0:
            warnings.append(
                f"- **Fat tails detected**: Excess kurtosis of "
                f"{bt['kurtosis']:.2f} indicates higher tail risk than normal."
            )

        if bt.get("skewness", 0) < -0.5:
            warnings.append(
                f"- **Negative skew**: Skewness of {bt['skewness']:.2f} "
                f"suggests more frequent extreme losses."
            )

        stress = results.get("stress_test", {})
        if stress.get("scenarios_failed", 0) > 0:
            warnings.append(
                f"- **Stress test failures**: {stress['scenarios_failed']} "
                f"scenario(s) breached drawdown limits."
            )

        robustness = results.get("robustness", {})
        rs = robustness.get("rolling_sharpe", {})
        if rs.get("pct_positive", 1) < 0.7:
            warnings.append(
                f"- **Inconsistent performance**: Rolling Sharpe positive only "
                f"{rs['pct_positive'] * 100:.0f}% of the time."
            )

        if not warnings:
            warnings.append("- No significant risk warnings identified.")

        lines.extend(warnings)

        lines.append(
            "\n---\n*This report is for research purposes only. Past performance "
            "does not guarantee future results. The framework has not been "
            "validated for live trading.*"
        )

        return "\n".join(lines)

    # ── LaTeX Table Export ───────────────────────────────────────────

    def generate_latex_table(
        self,
        bm_results: Dict[str, Any],
        caption: str = "Performance comparison of HRL-SARP vs benchmark strategies",
    ) -> str:
        """Generate LaTeX-formatted comparison table for academic papers."""
        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{{caption}}}",
            "\\label{tab:performance}",
            "\\begin{tabular}{lrrrrrr}",
            "\\toprule",
            "Strategy & CAGR & Sharpe & Sortino & Max DD & Alpha & IR \\\\",
            "\\midrule",
        ]

        for name, metrics in bm_results.items():
            if isinstance(metrics, dict):
                row = (
                    f"{name} & "
                    f"{metrics.get('cagr', 0) * 100:.2f}\\% & "
                    f"{metrics.get('sharpe_ratio', 0):.3f} & "
                    f"{metrics.get('sortino_ratio', 0):.3f} & "
                    f"{metrics.get('max_drawdown', 0) * 100:.2f}\\% & "
                    f"{metrics.get('alpha', 0) * 100:.2f}\\% & "
                    f"{metrics.get('information_ratio', 0):.3f} \\\\"
                )
                lines.append(row)

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ])

        latex = "\n".join(lines)

        path = os.path.join(self.output_dir, "performance_table.tex")
        with open(path, "w") as f:
            f.write(latex)
        logger.info("LaTeX table saved: %s", path)

        return latex
