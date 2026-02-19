"""
File: stress_testing.py
Module: risk
Description: Scenario-based stress testing framework for portfolio resilience.
    Tests agent performance under historically calibrated stress events
    (COVID-19, demonetisation, taper tantrum, IL&FS crisis) and synthetic
    extremes (circuit breakers, liquidity shocks, rate spikes).
Design Decisions: Each scenario defines return shocks per sector, enabling
    rapid portfolio P&L computation without full re-simulation. Results
    include P&L, drawdown, VaR/CVaR impact, and recovery analysis.
References: risk_config.yaml stress_scenarios, Basel III stress testing
Author: HRL-SARP Framework
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# STRESS SCENARIO DEFINITION
# ══════════════════════════════════════════════════════════════════════


class StressScenario:
    """A single stress scenario with sector-level return shocks."""

    def __init__(
        self,
        name: str,
        description: str,
        sector_shocks: Dict[str, float],
        duration_days: int = 5,
        recovery_days: int = 20,
        probability: float = 0.05,
    ) -> None:
        self.name = name
        self.description = description
        self.sector_shocks = sector_shocks
        self.duration_days = duration_days
        self.recovery_days = recovery_days
        self.probability = probability


class StressTester:
    """Stress testing framework for portfolio resilience analysis."""

    # NSE sector order (matching data_config.yaml)
    SECTOR_NAMES = [
        "IT", "Financials", "Pharma", "FMCG", "Auto",
        "Energy", "Metals", "Realty", "Telecom", "Media", "Infra",
    ]

    def __init__(
        self,
        config_path: str = "config/risk_config.yaml",
    ) -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        self.scenarios = self._build_scenarios()
        logger.info("StressTester initialised | %d scenarios", len(self.scenarios))

    def _build_scenarios(self) -> List[StressScenario]:
        """Build predefined India-specific stress scenarios."""
        scenarios = []

        # 1. COVID-19 Crash (March 2020)
        scenarios.append(StressScenario(
            name="covid_crash",
            description="COVID-19 market crash — Nifty fell 38% in 33 trading days",
            sector_shocks={
                "IT": -0.18, "Financials": -0.35, "Pharma": -0.10,
                "FMCG": -0.15, "Auto": -0.30, "Energy": -0.40,
                "Metals": -0.35, "Realty": -0.45, "Telecom": -0.12,
                "Media": -0.40, "Infra": -0.38,
            },
            duration_days=25,
            recovery_days=120,
            probability=0.01,
        ))

        # 2. Demonetisation (November 2016)
        scenarios.append(StressScenario(
            name="demonetisation",
            description="Currency demonetisation — cash-dependent sectors hit hard",
            sector_shocks={
                "IT": 0.02, "Financials": -0.15, "Pharma": -0.05,
                "FMCG": -0.12, "Auto": -0.18, "Energy": -0.08,
                "Metals": -0.10, "Realty": -0.25, "Telecom": -0.03,
                "Media": -0.08, "Infra": -0.14,
            },
            duration_days=15,
            recovery_days=60,
            probability=0.02,
        ))

        # 3. Taper Tantrum (2013)
        scenarios.append(StressScenario(
            name="taper_tantrum",
            description="US Fed taper tantrum — FII outflows and INR depreciation",
            sector_shocks={
                "IT": 0.05, "Financials": -0.20, "Pharma": 0.03,
                "FMCG": -0.08, "Auto": -0.15, "Energy": -0.12,
                "Metals": -0.18, "Realty": -0.22, "Telecom": -0.10,
                "Media": -0.12, "Infra": -0.16,
            },
            duration_days=30,
            recovery_days=90,
            probability=0.03,
        ))

        # 4. IL&FS / NBFC Crisis (2018)
        scenarios.append(StressScenario(
            name="ilfs_crisis",
            description="IL&FS default — NBFC/financial contagion",
            sector_shocks={
                "IT": -0.02, "Financials": -0.25, "Pharma": -0.05,
                "FMCG": -0.03, "Auto": -0.12, "Energy": -0.06,
                "Metals": -0.08, "Realty": -0.30, "Telecom": -0.04,
                "Media": -0.10, "Infra": -0.20,
            },
            duration_days=20,
            recovery_days=90,
            probability=0.03,
        ))

        # 5. Global Financial Crisis (2008)
        scenarios.append(StressScenario(
            name="gfc_2008",
            description="Global financial crisis — broad-based meltdown",
            sector_shocks={
                "IT": -0.30, "Financials": -0.45, "Pharma": -0.15,
                "FMCG": -0.10, "Auto": -0.35, "Energy": -0.40,
                "Metals": -0.50, "Realty": -0.60, "Telecom": -0.20,
                "Media": -0.35, "Infra": -0.45,
            },
            duration_days=60,
            recovery_days=365,
            probability=0.005,
        ))

        # 6. Rate Hike Cycle
        scenarios.append(StressScenario(
            name="rate_hike_cycle",
            description="RBI aggressive rate hike cycle — 200bps in 6 months",
            sector_shocks={
                "IT": 0.02, "Financials": -0.12, "Pharma": 0.0,
                "FMCG": -0.05, "Auto": -0.15, "Energy": -0.03,
                "Metals": -0.08, "Realty": -0.20, "Telecom": -0.05,
                "Media": -0.06, "Infra": -0.10,
            },
            duration_days=120,
            recovery_days=180,
            probability=0.05,
        ))

        # 7. Geopolitical Shock (India-Pakistan tension)
        scenarios.append(StressScenario(
            name="geopolitical_shock",
            description="Acute geopolitical tension — defence up, broad market down",
            sector_shocks={
                "IT": -0.08, "Financials": -0.15, "Pharma": -0.03,
                "FMCG": -0.05, "Auto": -0.12, "Energy": -0.10,
                "Metals": -0.08, "Realty": -0.15, "Telecom": -0.05,
                "Media": -0.10, "Infra": -0.08,
            },
            duration_days=5,
            recovery_days=15,
            probability=0.04,
        ))

        return scenarios

    # ── Stress Test Execution ────────────────────────────────────────

    def run_all(
        self,
        portfolio_weights: np.ndarray,
        portfolio_value: float,
    ) -> Dict[str, Dict[str, Any]]:
        """Run all stress scenarios on the current portfolio.

        Args:
            portfolio_weights: (11,) sector-level weights.
            portfolio_value: Current portfolio value in INR.

        Returns:
            Dict[scenario_name → result_dict].
        """
        results = {}
        for scenario in self.scenarios:
            results[scenario.name] = self.run_single(
                scenario, portfolio_weights, portfolio_value
            )
        return results

    def run_single(
        self,
        scenario: StressScenario,
        portfolio_weights: np.ndarray,
        portfolio_value: float,
    ) -> Dict[str, Any]:
        """Run a single stress scenario."""
        shock_vector = self._scenario_to_vector(scenario)
        n_sectors = len(portfolio_weights)
        shock_vector = shock_vector[:n_sectors]

        # Portfolio P&L
        portfolio_return = float(np.dot(portfolio_weights, shock_vector))
        pnl = portfolio_return * portfolio_value
        stressed_value = portfolio_value * (1.0 + portfolio_return)

        # Worst-case drawdown (assuming shocks happen linearly over duration)
        daily_shock = shock_vector / max(scenario.duration_days, 1)
        cumulative_returns = np.zeros(scenario.duration_days)
        for d in range(scenario.duration_days):
            day_return = float(np.dot(portfolio_weights, daily_shock))
            if d > 0:
                cumulative_returns[d] = (1 + cumulative_returns[d - 1]) * (1 + day_return) - 1
            else:
                cumulative_returns[d] = day_return

        max_drawdown = float(-np.min(cumulative_returns)) if len(cumulative_returns) > 0 else 0.0

        # Sector-level breakdown
        sector_pnl = {
            self.SECTOR_NAMES[i]: float(portfolio_weights[i] * shock_vector[i] * portfolio_value)
            for i in range(min(n_sectors, len(self.SECTOR_NAMES)))
        }

        # Top 3 worst-hit sectors
        sorted_sectors = sorted(sector_pnl.items(), key=lambda x: x[1])
        worst_sectors = sorted_sectors[:3]

        result = {
            "scenario": scenario.name,
            "description": scenario.description,
            "portfolio_return": portfolio_return,
            "pnl": pnl,
            "stressed_value": stressed_value,
            "max_drawdown": max_drawdown,
            "duration_days": scenario.duration_days,
            "recovery_days": scenario.recovery_days,
            "probability": scenario.probability,
            "expected_loss": pnl * scenario.probability,
            "sector_pnl": sector_pnl,
            "worst_sectors": worst_sectors,
            "passes_drawdown_limit": max_drawdown < self.cfg["portfolio"]["max_drawdown_pct"],
        }

        return result

    def _scenario_to_vector(self, scenario: StressScenario) -> np.ndarray:
        """Convert scenario sector shocks dict to numpy vector."""
        vector = np.zeros(len(self.SECTOR_NAMES))
        for i, name in enumerate(self.SECTOR_NAMES):
            vector[i] = scenario.sector_shocks.get(name, 0.0)
        return vector

    # ── Summary Report ───────────────────────────────────────────────

    def generate_report(
        self,
        results: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate aggregate stress test report."""
        all_pnl = [r["pnl"] for r in results.values()]
        all_dd = [r["max_drawdown"] for r in results.values()]
        scenarios_passed = sum(1 for r in results.values() if r["passes_drawdown_limit"])
        expected_total_loss = sum(r["expected_loss"] for r in results.values())

        report = {
            "n_scenarios": len(results),
            "scenarios_passed": scenarios_passed,
            "scenarios_failed": len(results) - scenarios_passed,
            "worst_case_pnl": float(min(all_pnl)),
            "worst_case_dd": float(max(all_dd)),
            "expected_annual_stress_loss": expected_total_loss,
            "scenario_details": results,
        }

        worst_name = min(results, key=lambda k: results[k]["pnl"])
        report["worst_scenario"] = worst_name

        logger.info(
            "Stress test: %d/%d passed | worst=%s (%.2f%%)",
            scenarios_passed, len(results),
            worst_name, results[worst_name]["portfolio_return"] * 100,
        )

        return report
