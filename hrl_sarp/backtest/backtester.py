"""
File: backtester.py
Module: backtest
Description: Walk-forward backtesting engine that replays historical data through
    trained HRL agents with realistic transaction costs, slippage, and risk
    management. Supports expanding-window and rolling-window protocols.
Design Decisions: Backtester operates on the HierarchicalEnv to test the full
    Macro→Micro pipeline. Transaction costs from base_env ensure consistency
    between training and evaluation. Walk-forward prevents look-ahead bias.
References: Pardo (2008) "Design, Testing, and Optimization of Trading Systems"
Author: HRL-SARP Framework
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backtest.performance_metrics import PerformanceMetrics

logger = logging.getLogger(__name__)


class Backtester:
    """Walk-forward backtesting engine for the HRL-SARP system."""

    def __init__(
        self,
        risk_free_rate: float = 0.07,
        initial_capital: float = 10_000_000.0,
        trading_days: int = 252,
    ) -> None:
        self.risk_free_rate = risk_free_rate
        self.initial_capital = initial_capital
        self.trading_days = trading_days
        self.metrics_calculator = PerformanceMetrics(
            risk_free_rate=risk_free_rate,
            trading_days=trading_days,
        )

        # Results storage
        self.portfolio_values: List[float] = []
        self.portfolio_returns: List[float] = []
        self.actions_history: List[Dict[str, Any]] = []
        self.regime_history: List[int] = []
        self.costs_history: List[float] = []
        self.sector_weights_history: List[np.ndarray] = []

        logger.info(
            "Backtester initialised | capital=%.0f | rf=%.2f%%",
            initial_capital, risk_free_rate * 100,
        )

    # ── Run Backtest ─────────────────────────────────────────────────

    def run(
        self,
        env,
        macro_agent=None,
        micro_agent=None,
        n_episodes: int = 1,
        deterministic: bool = True,
    ) -> Dict[str, Any]:
        """Run backtest through the environment.

        Args:
            env: HierarchicalEnv or any Gymnasium env.
            macro_agent: Trained MacroAgent (optional).
            micro_agent: Trained MicroAgent (optional).
            n_episodes: Number of backtest episodes (walk-forward windows).
            deterministic: Use deterministic policy (no exploration noise).

        Returns:
            Comprehensive backtest results dict.
        """
        self._reset()
        all_episode_results = []

        for ep in range(n_episodes):
            ep_result = self._run_single_episode(
                env, macro_agent, micro_agent, deterministic
            )
            all_episode_results.append(ep_result)
            logger.info(
                "Backtest episode %d/%d | return=%.2f%% | sharpe=%.3f",
                ep + 1, n_episodes,
                ep_result["total_return"] * 100,
                ep_result["sharpe_ratio"],
            )

        # Aggregate results
        results = self._aggregate_results(all_episode_results)
        return results

    def _run_single_episode(
        self,
        env,
        macro_agent,
        micro_agent,
        deterministic: bool,
    ) -> Dict[str, Any]:
        """Run a single backtest episode."""
        obs, info = env.reset()
        done = False

        ep_values = [self.initial_capital]
        ep_returns = []
        ep_actions = []
        ep_costs = []
        ep_regimes = []
        ep_sector_weights = []

        current_value = self.initial_capital

        while not done:
            # Get action from agents
            action = self._get_action(
                obs, info, macro_agent, micro_agent, deterministic
            )

            # Step environment
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated

            # Track portfolio value
            step_return = step_info.get("portfolio_return", reward)
            step_cost = step_info.get("transaction_cost", 0.0)

            current_value *= (1.0 + step_return)
            ep_values.append(current_value)
            ep_returns.append(step_return)
            ep_costs.append(step_cost)
            ep_regimes.append(step_info.get("regime", -1))

            # Track weights
            weights = step_info.get("sector_weights", np.zeros(11))
            ep_sector_weights.append(weights)

            # Store action details
            ep_actions.append({
                "step": len(ep_returns),
                "action": action.tolist() if hasattr(action, "tolist") else action,
                "reward": reward,
                "value": current_value,
            })

            obs = next_obs

        # Compute metrics
        returns_arr = np.array(ep_returns)
        metrics = self.metrics_calculator.compute_all(returns_arr)

        # Add value series
        metrics["portfolio_values"] = ep_values
        metrics["n_steps"] = len(ep_returns)
        metrics["total_costs"] = sum(ep_costs)
        metrics["cost_drag"] = sum(ep_costs) / max(sum(abs(r) for r in ep_returns), 1e-8)

        # Store for aggregate computation
        self.portfolio_values.extend(ep_values)
        self.portfolio_returns.extend(ep_returns)
        self.costs_history.extend(ep_costs)
        self.sector_weights_history.extend(ep_sector_weights)

        return metrics

    def _get_action(
        self,
        obs: np.ndarray,
        info: Dict,
        macro_agent,
        micro_agent,
        deterministic: bool,
    ) -> np.ndarray:
        """Get action from appropriate agent or fallback."""
        is_macro = info.get("is_macro_step", True)

        if is_macro and macro_agent is not None:
            macro_state = obs[:macro_agent.macro_state_dim]
            sector_emb = obs[macro_agent.macro_state_dim:].reshape(
                macro_agent.num_sectors, macro_agent.sector_emb_dim
            )
            action, _, _ = macro_agent.select_action(
                macro_state, sector_emb, deterministic
            )
            return action

        elif not is_macro and micro_agent is not None:
            max_s = micro_agent.max_stocks
            feat_dim = micro_agent.stock_feature_dim
            stock_features = obs[:max_s * feat_dim].reshape(max_s, feat_dim)
            goal = info.get("goal", np.zeros(micro_agent.goal_input_dim))
            mask = (np.abs(stock_features).sum(axis=-1) > 1e-6).astype(np.float32)
            action = micro_agent.select_action(
                stock_features, goal, mask, deterministic
            )
            return action

        else:
            # Fallback: random action
            return np.zeros(11)

    # ── Results Aggregation ──────────────────────────────────────────

    def _aggregate_results(
        self,
        episode_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Aggregate results across episodes."""
        all_returns = np.array(self.portfolio_returns)

        # Full-period metrics
        full_metrics = self.metrics_calculator.compute_all(all_returns)

        # Per-episode summary
        episode_summary = {
            "n_episodes": len(episode_results),
            "returns": [r["total_return"] for r in episode_results],
            "sharpes": [r["sharpe_ratio"] for r in episode_results],
            "max_drawdowns": [r["max_drawdown"] for r in episode_results],
            "calmars": [r["calmar_ratio"] for r in episode_results],
        }

        # Monthly returns (if enough data)
        monthly_returns = self._compute_monthly_returns(all_returns)

        results = {
            "full_period": full_metrics,
            "episodes": episode_summary,
            "monthly_returns": monthly_returns,
            "portfolio_values": self.portfolio_values,
            "total_cost_drag": sum(self.costs_history),
            "avg_turnover": self._compute_avg_turnover(),
        }

        logger.info(
            "Backtest complete | CAGR=%.2f%% | Sharpe=%.3f | MaxDD=%.2f%%",
            full_metrics.get("cagr", 0.0) * 100,
            full_metrics.get("sharpe_ratio", 0.0),
            full_metrics.get("max_drawdown", 0.0) * 100,
        )

        return results

    def _compute_monthly_returns(self, daily_returns: np.ndarray) -> List[float]:
        """Approximate monthly returns (21 trading days per month)."""
        days_per_month = 21
        monthly = []
        for start in range(0, len(daily_returns), days_per_month):
            end = min(start + days_per_month, len(daily_returns))
            month_ret = float(np.prod(1 + daily_returns[start:end]) - 1)
            monthly.append(month_ret)
        return monthly

    def _compute_avg_turnover(self) -> float:
        """Compute average portfolio turnover from weight history."""
        if len(self.sector_weights_history) < 2:
            return 0.0
        turnovers = []
        for i in range(1, len(self.sector_weights_history)):
            prev = self.sector_weights_history[i - 1]
            curr = self.sector_weights_history[i]
            min_len = min(len(prev), len(curr))
            turnover = float(np.sum(np.abs(curr[:min_len] - prev[:min_len])) / 2)
            turnovers.append(turnover)
        return float(np.mean(turnovers))

    def _reset(self) -> None:
        """Reset backtester state."""
        self.portfolio_values.clear()
        self.portfolio_returns.clear()
        self.actions_history.clear()
        self.regime_history.clear()
        self.costs_history.clear()
        self.sector_weights_history.clear()
