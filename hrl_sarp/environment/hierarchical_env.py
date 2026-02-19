"""
File: hierarchical_env.py
Module: environment
Description: HierarchicalEnv wraps MacroEnv and MicroEnv into a single coordinated
    environment. Handles goal passing from Macro → Micro, temporal alignment
    (weekly macro steps ↔ daily micro steps), and shared portfolio state.
Design Decisions: The Macro agent steps once per week, producing a goal embedding.
    The Micro agent steps every day within that week, conditioned on the goal.
    Portfolio value is shared — Micro's trades affect the real portfolio that
    Macro observes. This two-timescale loop is the core of HRL-SARP.
References: Option-critic architecture (Bacon 2017), Feudal Networks (Vezhnevets 2017)
Author: HRL-SARP Framework
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces

from environment.macro_env import MacroEnv
from environment.micro_env import MicroEnv
from environment.india_calendar import IndiaCalendar

logger = logging.getLogger(__name__)

# Trading days per week (approximately)
TRADING_DAYS_PER_WEEK = 5


# ══════════════════════════════════════════════════════════════════════
# HIERARCHICAL ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════


class HierarchicalEnv(gym.Env):
    """Two-level hierarchical environment coordinating Macro and Micro agents.

    The environment operates on two timescales:
    - Macro agent: steps weekly, outputs sector weights + regime
    - Micro agent: steps daily within each macro period

    External callers interact with this env at the daily level.
    The Macro agent is queried automatically at week boundaries.
    """

    metadata = {"render_modes": ["human", "log"]}

    def __init__(
        self,
        # Macro data
        sector_returns_data: np.ndarray,
        benchmark_returns_data: np.ndarray,
        macro_states_data: np.ndarray,
        sector_gnn_embeddings_data: Optional[np.ndarray] = None,
        regime_labels: Optional[np.ndarray] = None,
        macro_pe_zscores: Optional[np.ndarray] = None,
        # Micro data
        stock_returns_data: np.ndarray = None,
        stock_features_data: np.ndarray = None,
        stock_to_sector_idx: np.ndarray = None,
        stock_pe_zscores: Optional[np.ndarray] = None,
        # Config
        macro_config_path: str = "config/macro_agent_config.yaml",
        micro_config_path: str = "config/micro_agent_config.yaml",
        risk_config_path: str = "config/risk_config.yaml",
        initial_capital: float = 1_00_00_000.0,
        max_stocks: int = 50,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode

        # Create sub-environments
        self.macro_env = MacroEnv(
            sector_returns_data=sector_returns_data,
            benchmark_returns_data=benchmark_returns_data,
            macro_states_data=macro_states_data,
            sector_gnn_embeddings_data=sector_gnn_embeddings_data,
            regime_labels=regime_labels,
            pe_zscores=macro_pe_zscores,
            macro_config_path=macro_config_path,
            risk_config_path=risk_config_path,
            initial_capital=initial_capital,
            render_mode=render_mode,
        )

        self.micro_env = MicroEnv(
            stock_returns_data=stock_returns_data,
            stock_features_data=stock_features_data,
            stock_to_sector_idx=stock_to_sector_idx,
            pe_zscores_data=stock_pe_zscores,
            micro_config_path=micro_config_path,
            risk_config_path=risk_config_path,
            initial_capital=initial_capital,
            max_stocks=max_stocks,
            render_mode=render_mode,
        )

        # The external action/observation space is the Micro agent's
        self.observation_space = self.micro_env.observation_space
        self.action_space = self.micro_env.action_space

        # Temporal alignment
        self.daily_step: int = 0
        self.weekly_step: int = 0
        self.days_in_current_week: int = 0
        self.total_daily_steps: int = stock_returns_data.shape[0] - 1
        self.total_weekly_steps: int = sector_returns_data.shape[0] - 1

        # Current goal from Macro
        self.current_goal: Optional[np.ndarray] = None
        self.current_macro_action: Optional[np.ndarray] = None

        # Calendar for week detection
        self.calendar = IndiaCalendar(risk_config_path)

        # Tracking
        self.macro_decisions: List[Dict[str, Any]] = []
        self.micro_decisions: List[Dict[str, Any]] = []

        logger.info(
            "HierarchicalEnv initialised | daily_steps=%d | weekly_steps=%d | stocks=%d",
            self.total_daily_steps, self.total_weekly_steps, max_stocks,
        )

    # ── Gymnasium Interface ──────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset both environments and trigger initial macro step."""
        self.daily_step = 0
        self.weekly_step = 0
        self.days_in_current_week = 0
        self.macro_decisions = []
        self.micro_decisions = []

        # Reset sub-environments
        macro_obs, macro_info = self.macro_env.reset(seed=seed)
        micro_obs, micro_info = self.micro_env.reset(seed=seed)

        # Initial macro action (provided via options or defaults)
        if options and "initial_macro_action" in options:
            self.current_macro_action = np.array(
                options["initial_macro_action"], dtype=np.float32
            )
        else:
            # Default: equal sector weights + sideways regime
            sector_w = np.zeros(self.macro_env.num_sectors, dtype=np.float32)
            regime_logits = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            self.current_macro_action = np.concatenate([sector_w, regime_logits])

        # Set initial goal for micro
        goal = self._extract_goal_from_macro_action(self.current_macro_action)
        self.current_goal = goal
        self.micro_env.set_goal(goal)

        # Re-get micro obs with goal
        micro_obs = self.micro_env._get_observation()

        info = {
            "daily_step": 0,
            "weekly_step": 0,
            "macro_info": macro_info,
            "micro_info": micro_info,
            "goal": goal.copy(),
        }
        return micro_obs, info

    def step(
        self, micro_action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one daily step with the Micro agent's action.

        At week boundaries, the Macro env also steps (macro action must be
        provided via set_macro_action before the boundary).
        """
        # Execute micro step
        micro_obs, micro_reward, micro_term, micro_trunc, micro_info = (
            self.micro_env.step(micro_action)
        )

        self.daily_step += 1
        self.days_in_current_week += 1

        # Sync portfolio values (Micro drives the actual portfolio)
        self._sync_portfolio_state()

        # Check if week boundary reached
        is_week_end = self.days_in_current_week >= TRADING_DAYS_PER_WEEK
        macro_info = {}

        if is_week_end and self.weekly_step < self.total_weekly_steps:
            macro_info = self._execute_macro_step()
            self.days_in_current_week = 0

        # Termination
        terminated = micro_term or self.daily_step >= self.total_daily_steps
        truncated = micro_trunc

        info = {
            "daily_step": self.daily_step,
            "weekly_step": self.weekly_step,
            "is_week_end": is_week_end,
            "micro_info": micro_info,
            "macro_info": macro_info,
            "goal": self.current_goal.copy() if self.current_goal is not None else None,
            "portfolio_value": self.micro_env.portfolio_value,
        }

        self.micro_decisions.append({
            "step": self.daily_step,
            "reward": micro_reward,
            "value": self.micro_env.portfolio_value,
        })

        return micro_obs, micro_reward, terminated, truncated, info

    # ── Macro Agent Interface ────────────────────────────────────────

    def set_macro_action(self, macro_action: np.ndarray) -> None:
        """Set the next macro action to be used at the week boundary.

        Called by the training loop when the Macro agent produces its action.
        """
        self.current_macro_action = np.array(macro_action, dtype=np.float32)

    def get_macro_observation(self) -> np.ndarray:
        """Get the current macro-level observation for the Macro agent to act on."""
        return self.macro_env._get_observation()

    def _execute_macro_step(self) -> Dict[str, Any]:
        """Step the Macro environment and update goal for Micro."""
        if self.current_macro_action is None:
            logger.warning("No macro action set; using previous weights")
            sw = self.macro_env.sector_weights
            rl = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            self.current_macro_action = np.concatenate([sw, rl])

        _, macro_reward, macro_term, macro_trunc, macro_info = (
            self.macro_env.step(self.current_macro_action)
        )

        self.weekly_step += 1

        # Extract new goal
        goal = self._extract_goal_from_macro_action(self.current_macro_action)
        self.current_goal = goal
        self.micro_env.set_goal(goal)

        self.macro_decisions.append({
            "week": self.weekly_step,
            "reward": macro_reward,
            "sector_weights": macro_info.get("sector_weights"),
            "regime": macro_info.get("predicted_regime"),
        })

        macro_info["macro_reward"] = macro_reward
        return macro_info

    def _extract_goal_from_macro_action(self, macro_action: np.ndarray) -> np.ndarray:
        """Convert raw macro action to goal embedding (14D: 11 sector + 3 regime)."""
        num_sectors = self.macro_env.num_sectors
        sector_signal = macro_action[:num_sectors]
        regime_signal = macro_action[num_sectors:]

        # If actor already provides probability/simplex vectors, keep them.
        # Otherwise, interpret as logits and apply softmax.
        if (
            np.all(np.isfinite(sector_signal))
            and np.all(sector_signal >= 0.0)
            and np.isclose(np.sum(sector_signal), 1.0, atol=1e-3)
        ):
            sector_weights = sector_signal.astype(np.float32, copy=True)
        else:
            sector_weights = _softmax(sector_signal)

        if (
            np.all(np.isfinite(regime_signal))
            and np.all(regime_signal >= 0.0)
            and np.isclose(np.sum(regime_signal), 1.0, atol=1e-3)
        ):
            regime_probs = regime_signal.astype(np.float32, copy=True)
        else:
            regime_probs = _softmax(regime_signal)

        return np.concatenate([sector_weights, regime_probs]).astype(np.float32)

    # ── Portfolio Sync ───────────────────────────────────────────────

    def _sync_portfolio_state(self) -> None:
        """Sync Macro env's portfolio value with Micro env's (Micro is source of truth)."""
        self.macro_env.portfolio_value = self.micro_env.portfolio_value
        self.macro_env.peak_value = max(
            self.macro_env.peak_value, self.micro_env.portfolio_value
        )
        self.macro_env.current_drawdown = self.micro_env.current_drawdown

    # ── Data Collection for HER ──────────────────────────────────────

    def get_achieved_goal(self) -> np.ndarray:
        """Return the goal actually achieved by the Micro agent (for HER)."""
        return self.micro_env.get_achieved_goal()

    def get_desired_goal(self) -> np.ndarray:
        """Return the desired goal set by the Macro agent."""
        return self.current_goal.copy() if self.current_goal is not None else np.zeros(14)

    # ── Episode Summary ──────────────────────────────────────────────

    def get_episode_summary(self) -> Dict[str, Any]:
        """Return summary statistics for the completed episode."""
        return {
            "total_daily_steps": self.daily_step,
            "total_weekly_steps": self.weekly_step,
            "final_value": self.micro_env.portfolio_value,
            "total_return": (
                self.micro_env.portfolio_value / self.micro_env.initial_capital - 1.0
            ),
            "max_drawdown": self.micro_env.current_drawdown,
            "sharpe": self.micro_env.get_sharpe(
                window=min(len(self.micro_env.return_history), 252),
                annualise_factor=np.sqrt(252),
            ),
            "calmar": self.micro_env.get_calmar(),
            "macro_decisions": len(self.macro_decisions),
            "micro_decisions": len(self.micro_decisions),
        }

    def render(self) -> None:
        if self.render_mode == "human":
            print(
                f"Day {self.daily_step} / Week {self.weekly_step} | "
                f"Value: ₹{self.micro_env.portfolio_value:,.0f} | "
                f"DD: {self.micro_env.current_drawdown:.2%}"
            )


# ── Helper ───────────────────────────────────────────────────────────

def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x)
    exp_x = np.exp(shifted)
    return exp_x / (np.sum(exp_x) + 1e-8)
