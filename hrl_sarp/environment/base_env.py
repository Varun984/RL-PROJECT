"""
File: base_env.py
Module: environment
Description: Abstract base Gymnasium environment providing shared utilities for
    portfolio tracking, transaction cost computation, drawdown monitoring, and
    position bookkeeping. Both MacroEnv and MicroEnv inherit from this class.
Design Decisions: Centralising cost/drawdown logic avoids duplication and ensures
    consistent accounting across the hierarchy.
References: Gymnasium API (Farama Foundation), OpenAI Gym interface
Author: HRL-SARP Framework
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import yaml

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# BASE PORTFOLIO ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════


class BasePortfolioEnv(gym.Env, ABC):
    """Abstract base class for all HRL-SARP environments."""

    metadata = {"render_modes": ["human", "log"]}

    def __init__(
        self,
        risk_config_path: str = "config/risk_config.yaml",
        initial_capital: float = 1_00_00_000.0,  # ₹1 Crore
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode

        # Load risk configuration
        with open(risk_config_path, "r", encoding="utf-8") as f:self.risk_cfg = yaml.safe_load(f)

        # Transaction cost parameters
        tc = self.risk_cfg["transaction_costs"]
        self.stt_delivery_buy: float = tc["stt"]["delivery_buy"]
        self.stt_delivery_sell: float = tc["stt"]["delivery_sell"]
        self.brokerage_delivery: float = tc["brokerage"]["delivery"]
        self.exchange_charge: float = tc["exchange"]["nse_turnover_charge"]
        self.sebi_fee: float = tc["sebi_fee"]
        self.gst_rate: float = tc["gst_rate"]
        self.stamp_duty_buy: float = tc["stamp_duty"]["buy"]
        self.slippage_bps: float = tc["slippage"]["base_bps"]
        self.slippage_volume_factor: float = tc["slippage"]["volume_impact_factor"]

        # Portfolio state
        self.initial_capital = initial_capital
        self.portfolio_value: float = initial_capital
        self.cash: float = initial_capital
        self.peak_value: float = initial_capital
        self.current_drawdown: float = 0.0
        self.weights: np.ndarray = np.array([])
        self.prev_weights: np.ndarray = np.array([])

        # Tracking
        self.portfolio_history: List[float] = []
        self.return_history: List[float] = []
        self.cost_history: List[float] = []
        self.step_count: int = 0
        self.episode_count: int = 0

    # ── Transaction Cost Model ───────────────────────────────────────

    def compute_transaction_cost(
        self,
        trade_value: float,
        is_buy: bool,
        is_delivery: bool = True,
    ) -> float:
        """Compute India-specific transaction costs for a single trade.

        Includes STT, brokerage, exchange charge, SEBI fee, GST, stamp duty,
        and slippage estimate.
        """
        if abs(trade_value) < 1.0:
            return 0.0

        abs_value = abs(trade_value)

        # STT
        if is_delivery:
            stt = abs_value * (self.stt_delivery_buy if is_buy else self.stt_delivery_sell)
        else:
            stt = abs_value * self.risk_cfg["transaction_costs"]["stt"]["intraday_sell"] if not is_buy else 0.0

        # Brokerage
        brokerage = abs_value * self.brokerage_delivery

        # Exchange + SEBI charges
        exchange = abs_value * self.exchange_charge
        sebi = abs_value * self.sebi_fee

        # GST on brokerage + exchange charges
        gst = (brokerage + exchange) * self.gst_rate

        # Stamp duty (buy side only)
        stamp = abs_value * self.stamp_duty_buy if is_buy else 0.0

        # Slippage estimate
        slippage = abs_value * (self.slippage_bps / 10_000.0)

        total = stt + brokerage + exchange + sebi + gst + stamp + slippage
        return total

    def compute_rebalance_cost(
        self,
        old_weights: np.ndarray,
        new_weights: np.ndarray,
        portfolio_value: float,
    ) -> float:
        """Compute total cost for rebalancing from old_weights to new_weights."""
        weight_diff = new_weights - old_weights
        total_cost = 0.0

        for i, diff in enumerate(weight_diff):
            trade_value = abs(diff) * portfolio_value
            if trade_value < 1.0:
                continue
            is_buy = diff > 0
            total_cost += self.compute_transaction_cost(trade_value, is_buy)

        return total_cost

    # ── Drawdown Tracking ────────────────────────────────────────────

    def update_drawdown(self) -> float:
        """Update peak tracking and return current drawdown fraction."""
        if self.portfolio_value > self.peak_value:
            self.peak_value = self.portfolio_value
        self.current_drawdown = (self.peak_value - self.portfolio_value) / max(
            self.peak_value, 1e-8
        )
        return self.current_drawdown

    def check_circuit_breaker(self) -> bool:
        """Return True if max drawdown gate triggered."""
        max_dd = self.risk_cfg["portfolio"]["max_drawdown_pct"]
        return self.current_drawdown >= max_dd

    # ── Portfolio Accounting ────────────────────────────────────────

    def compute_portfolio_return(self, asset_returns: np.ndarray) -> float:
        """Weighted portfolio return for one step."""
        if len(self.weights) == 0 or len(asset_returns) == 0:
            return 0.0
        return float(np.dot(self.weights, asset_returns))

    def update_portfolio_value(
        self,
        asset_returns: np.ndarray,
        new_weights: np.ndarray,
    ) -> Tuple[float, float]:
        """Apply asset returns, rebalance to new weights, deduct costs.

        Returns:
            (net_return, total_cost)
        """
        # Step 1: mark-to-market with existing weights
        gross_return = self.compute_portfolio_return(asset_returns)
        self.portfolio_value *= (1.0 + gross_return)

        # Step 2: rebalance cost
        old_w = self.weights if len(self.weights) == len(new_weights) else np.zeros_like(new_weights)
        cost = self.compute_rebalance_cost(old_w, new_weights, self.portfolio_value)
        self.portfolio_value -= cost
        self.cash = self.portfolio_value * (1.0 - np.sum(new_weights))

        # Step 3: apply new weights
        self.prev_weights = old_w.copy()
        self.weights = new_weights.copy()

        # Step 4: track
        self.portfolio_history.append(self.portfolio_value)
        self.return_history.append(gross_return)
        self.cost_history.append(cost)

        self.update_drawdown()
        return gross_return, cost

    # ── Weight Normalisation Utilities ───────────────────────────────

    @staticmethod
    def softmax_weights(logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax to produce portfolio weights."""
        shifted = logits - np.max(logits)
        exp_vals = np.exp(shifted)
        return exp_vals / (np.sum(exp_vals) + 1e-8)

    @staticmethod
    def clip_weights(
        weights: np.ndarray,
        max_weight: float,
        min_weight: float = 0.0,
    ) -> np.ndarray:
        """Clip individual weights and re-normalise to sum to ≤ 1."""
        clipped = np.clip(weights, min_weight, max_weight)
        total = np.sum(clipped)
        if total > 1.0:
            clipped = clipped / total
        return clipped

    # ── Metrics Helpers ──────────────────────────────────────────────

    def get_sharpe(self, window: int = 52, annualise_factor: float = np.sqrt(52)) -> float:
        """Compute Sharpe ratio over recent returns (default: weekly, annualised)."""
        if len(self.return_history) < 2:
            return 0.0
        returns = np.array(self.return_history[-window:])
        mean_r = np.mean(returns)
        std_r = np.std(returns)
        if std_r < 1e-8:
            return 0.0
        return float(mean_r / std_r * annualise_factor)

    def get_calmar(self, window: int = 252) -> float:
        """Calmar ratio = annualised return / max drawdown."""
        if len(self.return_history) < 2:
            return 0.0
        cum_return = np.prod(1.0 + np.array(self.return_history[-window:])) - 1.0
        # Use min 1% drawdown to prevent reward explosion when DD is near zero
        max_dd = max(self.current_drawdown, 0.01)
        return float(cum_return / max_dd)

    def get_sortino(self, window: int = 52, annualise_factor: float = np.sqrt(52)) -> float:
        """Sortino ratio using downside deviation only."""
        if len(self.return_history) < 2:
            return 0.0
        returns = np.array(self.return_history[-window:])
        mean_r = np.mean(returns)
        downside = returns[returns < 0]
        if len(downside) < 1:
            return float(mean_r * annualise_factor) if mean_r > 0 else 0.0
        downside_std = np.std(downside)
        if downside_std < 1e-8:
            return 0.0
        return float(mean_r / downside_std * annualise_factor)

    def get_cvar(self, confidence: float = 0.95) -> float:
        """Conditional Value-at-Risk at given confidence level."""
        if len(self.return_history) < 5:
            return 0.0
        returns = np.sort(self.return_history)
        cutoff = int(len(returns) * (1.0 - confidence))
        cutoff = max(cutoff, 1)
        return float(-np.mean(returns[:cutoff]))

    # ── Abstract Interface ───────────────────────────────────────────

    @abstractmethod
    def _get_observation(self) -> np.ndarray:
        """Return the current observation vector."""
        ...

    @abstractmethod
    def _compute_reward(self, info: Dict[str, Any]) -> float:
        """Compute reward for the current step."""
        ...

    def reset_portfolio(self) -> None:
        """Reset all portfolio state to initial values."""
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        self.peak_value = self.initial_capital
        self.current_drawdown = 0.0
        self.weights = np.array([])
        self.prev_weights = np.array([])
        self.portfolio_history = [self.initial_capital]
        self.return_history = []
        self.cost_history = []
        self.step_count = 0

    def render(self) -> None:
        """Render current portfolio state."""
        if self.render_mode == "human":
            print(
                f"Step {self.step_count} | Value: ₹{self.portfolio_value:,.0f} | "
                f"DD: {self.current_drawdown:.2%} | Sharpe: {self.get_sharpe():.2f}"
            )
        elif self.render_mode == "log":
            logger.info(
                "Step %d | Value: %.0f | DD: %.4f | Sharpe: %.2f",
                self.step_count,
                self.portfolio_value,
                self.current_drawdown,
                self.get_sharpe(),
            )
