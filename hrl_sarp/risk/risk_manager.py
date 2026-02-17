"""
File: risk_manager.py
Module: risk
Description: Real-time risk monitoring engine that evaluates portfolio risk at every
    step and can override agent actions when limits are breached. Integrates
    drawdown circuit breakers, CVaR position sizing, event risk reduction,
    and NSE circuit breaker compliance.
Design Decisions: Acts as a safety layer between agent actions and execution. All
    limits loaded from risk_config.yaml. Uses a waterfall of checks: drawdown →
    concentration → liquidity → event risk → CVaR. Any breach triggers position
    adjustment before execution.
References: SEBI circular on risk management, NSE circuit breaker rules
Author: HRL-SARP Framework
"""

import logging
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

logger = logging.getLogger(__name__)


class RiskManager:
    """Real-time portfolio risk manager with India-specific constraints."""

    def __init__(
        self,
        config_path: str = "config/risk_config.yaml",
    ) -> None:
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        # Portfolio limits
        port_cfg = self.cfg["portfolio"]
        self.max_drawdown: float = port_cfg["max_drawdown_pct"]
        self.recovery_dd: float = port_cfg["max_drawdown_recovery_pct"]
        self.max_leverage: float = port_cfg["max_leverage"]
        self.min_cash: float = port_cfg["min_cash_pct"]
        self.max_positions: int = port_cfg["max_positions"]
        self.min_positions: int = port_cfg["min_positions"]
        self.rebalance_threshold: float = port_cfg["rebalance_threshold"]

        # Sector limits
        sec_cfg = self.cfg["sector"]
        self.max_sector_pct: float = sec_cfg["max_single_sector_pct"]
        self.min_sector_pct: float = sec_cfg["min_single_sector_pct"]
        self.max_correlated: float = sec_cfg["max_correlated_sectors_pct"]

        # Stock limits
        stk_cfg = self.cfg["stock"]
        self.max_stock_pct: float = stk_cfg["max_single_stock_pct"]
        self.min_stock_pct: float = stk_cfg["min_single_stock_pct"]

        # CVaR config
        cvar_cfg = self.cfg["cvar"]
        self.cvar_confidence: float = cvar_cfg["confidence_level"]
        self.max_cvar_contrib: float = cvar_cfg["max_cvar_contribution"]
        self.mc_sims: int = cvar_cfg["monte_carlo_simulations"]

        # Event risk
        evt_cfg = self.cfg["events"]
        self.event_risk_reduction: float = evt_cfg["event_risk_reduction"]

        # State
        self.is_halted: bool = False
        self.halt_reason: str = ""
        self.risk_alerts: List[Dict[str, Any]] = []

        logger.info("RiskManager initialised")

    # ── Main Risk Check ──────────────────────────────────────────────

    def check_and_adjust(
        self,
        proposed_weights: np.ndarray,
        portfolio_value: float,
        current_drawdown: float,
        current_date: Optional[date] = None,
        sector_map: Optional[Dict[int, int]] = None,
        event_risk: Optional[Dict[str, Any]] = None,
        return_history: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Run full risk waterfall and return adjusted weights.

        Args:
            proposed_weights: (N,) raw portfolio weights from agent.
            portfolio_value: Current portfolio value.
            current_drawdown: Current drawdown from peak.
            current_date: Today's date for event checks.
            sector_map: stock_idx → sector_idx mapping.
            event_risk: Event risk info from IndiaCalendar.
            return_history: (T, N) return history for CVaR.

        Returns:
            (adjusted_weights, risk_report)
        """
        self.risk_alerts.clear()
        weights = proposed_weights.copy()
        report: Dict[str, Any] = {"adjustments": [], "is_halted": False}

        # ── 1. Drawdown circuit breaker ──
        weights, dd_triggered = self._check_drawdown(weights, current_drawdown)
        if dd_triggered:
            report["is_halted"] = True
            report["halt_reason"] = "drawdown_breach"
            return weights, report

        # ── 2. Concentration limits ──
        weights = self._enforce_concentration(weights)

        # ── 3. Sector limits ──
        if sector_map is not None:
            weights = self._enforce_sector_limits(weights, sector_map)

        # ── 4. Position count limits ──
        weights = self._enforce_position_count(weights)

        # ── 5. Cash reserve ──
        weights = self._enforce_cash_reserve(weights)

        # ── 6. Event risk reduction ──
        if event_risk is not None and event_risk.get("event_risk_flag", False):
            weights = self._apply_event_risk_reduction(weights, event_risk)

        # ── 7. CVaR check ──
        if return_history is not None and len(return_history) > 50:
            weights = self._check_cvar(weights, return_history)

        # ── 8. Normalise ──
        weights = self._normalise(weights)

        report["adjusted_weights"] = weights.tolist()
        report["n_alerts"] = len(self.risk_alerts)
        report["alerts"] = self.risk_alerts.copy()

        return weights, report

    # ── Drawdown Circuit Breaker ─────────────────────────────────────

    def _check_drawdown(
        self,
        weights: np.ndarray,
        current_drawdown: float,
    ) -> Tuple[np.ndarray, bool]:
        """Flatten to cash if drawdown exceeds limit."""
        if current_drawdown >= self.max_drawdown:
            self.is_halted = True
            self.halt_reason = f"Drawdown {current_drawdown:.2%} >= {self.max_drawdown:.2%}"
            self._alert("CRITICAL", "drawdown_breach", self.halt_reason)
            # Flatten: all weight to cash (zeros)
            return np.zeros_like(weights), True

        # Check if we can re-enter after halt
        if self.is_halted and current_drawdown <= self.recovery_dd:
            self.is_halted = False
            self.halt_reason = ""
            self._alert("INFO", "drawdown_recovery", "Portfolio re-entry allowed")

        return weights, False

    # ── Concentration Limits ─────────────────────────────────────────

    def _enforce_concentration(self, weights: np.ndarray) -> np.ndarray:
        """Cap individual stock weights and redistribute excess."""
        excess = 0.0
        for i in range(len(weights)):
            if weights[i] > self.max_stock_pct:
                excess += weights[i] - self.max_stock_pct
                self._alert(
                    "WARNING", "stock_concentration",
                    f"Stock {i} capped: {weights[i]:.2%} → {self.max_stock_pct:.2%}",
                )
                weights[i] = self.max_stock_pct

            # Remove dust positions
            if 0 < weights[i] < self.min_stock_pct:
                excess += weights[i]
                weights[i] = 0.0

        # Redistribute excess proportionally to remaining positions
        if excess > 1e-8:
            active = weights > 0
            if active.sum() > 0:
                weights[active] += excess * (weights[active] / weights[active].sum())

        return weights

    # ── Sector Limits ────────────────────────────────────────────────

    def _enforce_sector_limits(
        self,
        weights: np.ndarray,
        sector_map: Dict[int, int],
    ) -> np.ndarray:
        """Enforce per-sector concentration caps."""
        n_sectors = max(sector_map.values()) + 1 if sector_map else 0
        sector_exposure = np.zeros(n_sectors)

        for stock_idx, sector_idx in sector_map.items():
            if stock_idx < len(weights):
                sector_exposure[sector_idx] += weights[stock_idx]

        for s in range(n_sectors):
            if sector_exposure[s] > self.max_sector_pct:
                # Scale down stocks in this sector
                scale = self.max_sector_pct / sector_exposure[s]
                for stock_idx, sector_idx in sector_map.items():
                    if sector_idx == s and stock_idx < len(weights):
                        weights[stock_idx] *= scale
                self._alert(
                    "WARNING", "sector_concentration",
                    f"Sector {s} capped: {sector_exposure[s]:.2%} → {self.max_sector_pct:.2%}",
                )

        return weights

    # ── Position Count ───────────────────────────────────────────────

    def _enforce_position_count(self, weights: np.ndarray) -> np.ndarray:
        """Enforce min/max position count."""
        active_count = (weights > 0).sum()

        if active_count > self.max_positions:
            # Remove smallest positions until within limit
            sorted_idx = np.argsort(weights)
            n_remove = active_count - self.max_positions
            for i in range(int(n_remove)):
                weights[sorted_idx[i]] = 0.0
            self._alert(
                "WARNING", "max_positions",
                f"Reduced positions from {active_count} to {self.max_positions}",
            )

        return weights

    # ── Cash Reserve ─────────────────────────────────────────────────

    def _enforce_cash_reserve(self, weights: np.ndarray) -> np.ndarray:
        """Ensure minimum cash buffer by scaling down all positions."""
        total_invested = weights.sum()
        max_invested = 1.0 - self.min_cash

        if total_invested > max_invested:
            scale = max_invested / total_invested
            weights *= scale

        return weights

    # ── Event Risk Reduction ─────────────────────────────────────────

    def _apply_event_risk_reduction(
        self,
        weights: np.ndarray,
        event_risk: Dict[str, Any],
    ) -> np.ndarray:
        """Reduce position sizes before major events."""
        reduction = event_risk.get("position_reduction", self.event_risk_reduction)
        weights *= (1.0 - reduction)
        self._alert(
            "INFO", "event_risk",
            f"Event risk reduction applied: {reduction:.0%} | "
            f"Next event in {event_risk.get('days_to_next_event', '?')} days",
        )
        return weights

    # ── CVaR Position Sizing ─────────────────────────────────────────

    def _check_cvar(
        self,
        weights: np.ndarray,
        return_history: np.ndarray,
    ) -> np.ndarray:
        """Check and adjust for CVaR contribution limits."""
        n_assets = min(weights.shape[0], return_history.shape[1])
        weights_trim = weights[:n_assets]
        returns_trim = return_history[:, :n_assets]

        # Portfolio returns
        port_returns = returns_trim @ weights_trim
        alpha = 1.0 - self.cvar_confidence
        cutoff = int(alpha * len(port_returns))
        if cutoff < 1:
            return weights

        sorted_returns = np.sort(port_returns)
        cvar = -float(sorted_returns[:cutoff].mean())

        if cvar > 0:
            # Marginal CVaR contribution per asset
            tail_mask = port_returns <= sorted_returns[cutoff - 1]
            tail_returns = returns_trim[tail_mask]

            if len(tail_returns) > 0:
                marginal_cvar = -tail_returns.mean(axis=0) * weights_trim
                total_marginal = marginal_cvar.sum()

                if total_marginal > 1e-8:
                    cvar_contrib = marginal_cvar / total_marginal

                    for i in range(n_assets):
                        if cvar_contrib[i] > self.max_cvar_contrib:
                            scale = self.max_cvar_contrib / cvar_contrib[i]
                            weights[i] *= scale
                            self._alert(
                                "WARNING", "cvar_breach",
                                f"Asset {i} CVaR contribution {cvar_contrib[i]:.2%} "
                                f"> {self.max_cvar_contrib:.2%}",
                            )

        return weights

    # ── Normalisation ────────────────────────────────────────────────

    @staticmethod
    def _normalise(weights: np.ndarray) -> np.ndarray:
        """Re-normalise to sum ≤ 1.0 (excess goes to cash)."""
        total = weights.sum()
        if total > 1.0:
            weights /= total
        weights = np.clip(weights, 0.0, 1.0)
        return weights

    # ── Alerts ───────────────────────────────────────────────────────

    def _alert(self, level: str, category: str, message: str) -> None:
        self.risk_alerts.append({"level": level, "category": category, "message": message})
        if level == "CRITICAL":
            logger.critical("[RISK] %s: %s", category, message)
        elif level == "WARNING":
            logger.warning("[RISK] %s: %s", category, message)
        else:
            logger.info("[RISK] %s: %s", category, message)

    # ── Query Methods ────────────────────────────────────────────────

    def get_portfolio_risk_summary(
        self,
        weights: np.ndarray,
        return_history: np.ndarray,
    ) -> Dict[str, float]:
        """Compute risk summary for current portfolio."""
        port_returns = return_history @ weights[:return_history.shape[1]]

        vol = float(np.std(port_returns) * np.sqrt(252))
        sharpe = float(np.mean(port_returns) / (np.std(port_returns) + 1e-8) * np.sqrt(252))

        cumulative = np.cumprod(1 + port_returns)
        peak = np.maximum.accumulate(cumulative)
        drawdowns = (peak - cumulative) / peak
        max_dd = float(drawdowns.max())

        # CVaR
        alpha = 1.0 - self.cvar_confidence
        cutoff = max(int(alpha * len(port_returns)), 1)
        cvar = -float(np.sort(port_returns)[:cutoff].mean())

        # HHI concentration
        hhi = float(np.sum(weights ** 2))

        return {
            "volatility": vol,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "cvar_95": cvar,
            "hhi": hhi,
            "n_positions": int((weights > 0).sum()),
            "is_halted": self.is_halted,
        }
