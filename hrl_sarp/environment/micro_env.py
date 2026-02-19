"""
File: micro_env.py
Module: environment
Description: MicroEnv is a daily-stepping Gymnasium environment for the Micro agent.
    State = per-stock features + goal embedding from Macro. Action = stock-level weights.
    Reward combines Sharpe, goal alignment, drawdown penalty, and value bonus.
Design Decisions: Daily stepping captures intraday opportunities. Goal conditioning via
    the Macro agent's sector weights enables hierarchical coordination. Variable-size
    stock universe handled via padding and masking.
References: TD3 (Fujimoto 2018), HER (Andrychowicz 2017), Goal-conditioned RL (Schaul 2015)
Author: HRL-SARP Framework
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces

from environment.base_env import BasePortfolioEnv
from environment.reward_functions import compute_total_micro_reward, goal_alignment_reward

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# MICRO ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════


class MicroEnv(BasePortfolioEnv):
    """Daily-stepping environment for the Micro (stock selection) agent.

    Observation: per-stock features (N, 22) flattened + goal embedding (14D)
    Action: stock weights (N,) via softmax
    Reward: R_total_micro composite
    """

    def __init__(
        self,
        stock_returns_data: np.ndarray,
        stock_features_data: np.ndarray,
        stock_to_sector_idx: np.ndarray,
        goal_embedding: Optional[np.ndarray] = None,
        pe_zscores_data: Optional[np.ndarray] = None,
        micro_config_path: str = "config/micro_agent_config.yaml",
        risk_config_path: str = "config/risk_config.yaml",
        initial_capital: float = 1_00_00_000.0,
        max_stocks: int = 50,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__(risk_config_path, initial_capital, render_mode)

        with open(micro_config_path, "r", encoding="utf-8") as f:self.micro_cfg = yaml.safe_load(f)

        # Episode data
        self.stock_returns_data = stock_returns_data  # (T, N_stocks)
        self.stock_features_data = stock_features_data  # (T, N_stocks, 22)
        self.stock_to_sector_idx = stock_to_sector_idx  # (N_stocks,) sector index per stock

        self.num_stocks_actual: int = stock_returns_data.shape[1]
        self.max_stocks: int = max_stocks
        self.stock_feature_dim: int = self.micro_cfg["network"]["stock_feature_dim"]
        self.goal_dim: int = self.micro_cfg["network"]["goal_encoder"]["input_dim"]  # 14

        # Goal from Macro agent (sector_weights 11 + regime_one_hot 3 = 14)
        if goal_embedding is not None:
            self.goal_embedding = goal_embedding.astype(np.float32)
        else:
            # Default: equal sector weights + sideways regime
            default_sw = np.ones(11, dtype=np.float32) / 11.0
            default_regime = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            self.goal_embedding = np.concatenate([default_sw, default_regime])

        # PE z-scores for value discovery reward
        self.pe_zscores = pe_zscores_data if pe_zscores_data is not None else np.zeros(
            (stock_returns_data.shape[0], self.num_stocks_actual), dtype=np.float32
        )

        # Observation: flattened stock features (max_stocks * 22) + goal (14)
        obs_dim = self.max_stocks * self.stock_feature_dim + self.goal_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Action: weight per stock (padded to max_stocks)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.max_stocks,), dtype=np.float32
        )

        # Reward config
        self.reward_cfg = self.micro_cfg["reward"]
        self.dd_threshold: float = self.reward_cfg["drawdown_threshold"]

        # Episode state
        self.max_steps: int = stock_returns_data.shape[0] - 1
        self.current_step: int = 0
        self.stock_weights = np.zeros(self.num_stocks_actual, dtype=np.float32)
        self.weekly_returns_buffer: List[float] = []

        # Stock mask (1 for valid stocks, 0 for padding)
        self.stock_mask = np.zeros(self.max_stocks, dtype=np.float32)
        self.stock_mask[:self.num_stocks_actual] = 1.0

        logger.info(
            "MicroEnv initialised | stocks=%d (pad to %d) | obs_dim=%d | max_steps=%d",
            self.num_stocks_actual, self.max_stocks, obs_dim, self.max_steps,
        )

    # ── Gymnasium Interface ──────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.reset_portfolio()
        self.current_step = 0
        self.episode_count += 1
        self.stock_weights = np.zeros(self.num_stocks_actual, dtype=np.float32)
        self.weekly_returns_buffer = []

        # Allow updating goal via options
        if options and "goal_embedding" in options:
            self.goal_embedding = np.array(options["goal_embedding"], dtype=np.float32)

        obs = self._get_observation()
        info = {"step": 0, "portfolio_value": self.initial_capital}
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Extract valid stock actions (ignore padding)
        raw_action = action[:self.num_stocks_actual]

        # Softmax over valid stocks for portfolio weights
        new_weights = self.softmax_weights(raw_action)

        # Apply single-stock concentration cap
        max_stock_pct = self.risk_cfg["stock"]["max_single_stock_pct"]
        new_weights = self.clip_weights(new_weights, max_stock_pct)

        # Get returns for this step
        stock_returns = self.stock_returns_data[self.current_step]

        # Compute portfolio return and update
        gross_return, cost = self.update_portfolio_value(stock_returns, new_weights)
        self.weekly_returns_buffer.append(gross_return)
        self.stock_weights = new_weights.copy()

        # Compute achieved sector allocation (aggregate stock weights by sector)
        achieved_sector_weights = self._compute_achieved_sector_weights(new_weights)
        target_sector_weights = self.goal_embedding[:11]

        # Portfolio-level PE z-score (weighted average)
        pe_z = float(np.dot(new_weights, self.pe_zscores[self.current_step]))

        # Compute reward (weekly granularity: accumulate daily, compute at week end)
        weekly_rets = np.array(self.weekly_returns_buffer, dtype=np.float32)
        reward = compute_total_micro_reward(
            weekly_returns=weekly_rets,
            achieved_sector_weights=achieved_sector_weights,
            target_sector_weights=target_sector_weights,
            current_drawdown=self.current_drawdown,
            calmar_ratio=self.get_calmar(),
            cvar_95=self.get_cvar(),
            stt_cost=cost / max(self.portfolio_value, 1.0),
            pe_zscore=pe_z,
            weekly_return=gross_return,
            w_micro=self.reward_cfg["w_micro"],
            w_portfolio=self.reward_cfg["w_portfolio"],
            w_value=self.reward_cfg["w_value_bonus"],
            sharpe_weight=self.reward_cfg["sharpe_weight"],
            goal_weight=self.reward_cfg["goal_alignment_weight"],
            dd_weight=self.reward_cfg["drawdown_penalty_weight"],
            dd_threshold=self.dd_threshold,
        )

        # Advance
        self.current_step += 1
        self.step_count += 1
        terminated = self.current_step >= self.max_steps
        truncated = self.check_circuit_breaker()

        # Goal alignment score
        goal_cos = goal_alignment_reward(achieved_sector_weights, target_sector_weights)

        info = {
            "step": self.current_step,
            "portfolio_value": self.portfolio_value,
            "drawdown": self.current_drawdown,
            "stock_weights": new_weights.copy(),
            "goal_alignment": goal_cos,
            "achieved_sector_weights": achieved_sector_weights,
            "gross_return": gross_return,
            "cost": cost,
            "sharpe": self.get_sharpe(window=20, annualise_factor=np.sqrt(252)),
        }

        obs = self._get_observation()
        return obs, reward, terminated, truncated, info

    # ── Observation Construction ─────────────────────────────────────

    def _get_observation(self) -> np.ndarray:
        idx = min(self.current_step, len(self.stock_features_data) - 1)

        # Pad stock features to max_stocks
        feats = self.stock_features_data[idx]  # (N_actual, 22)
        padded = np.zeros((self.max_stocks, self.stock_feature_dim), dtype=np.float32)
        padded[:self.num_stocks_actual] = feats

        flat_feats = padded.flatten()  # (max_stocks * 22,)
        obs = np.concatenate([flat_feats, self.goal_embedding]).astype(np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

    def _compute_reward(self, info: Dict[str, Any]) -> float:
        return 0.0

    # ── Sector Weight Aggregation ────────────────────────────────────

    def _compute_achieved_sector_weights(self, stock_weights: np.ndarray) -> np.ndarray:
        """Aggregate stock-level weights into sector-level weights."""
        sector_weights = np.zeros(11, dtype=np.float32)
        for i, w in enumerate(stock_weights):
            sector_idx = int(self.stock_to_sector_idx[i])
            if 0 <= sector_idx < 11:
                sector_weights[sector_idx] += w
        return sector_weights

    # ── Goal Update ──────────────────────────────────────────────────

    def set_goal(self, goal_embedding: np.ndarray) -> None:
        """Update the goal from the Macro agent (called at each macro step)."""
        self.goal_embedding = np.array(goal_embedding, dtype=np.float32)
        self.weekly_returns_buffer = []

    def get_achieved_goal(self) -> np.ndarray:
        """Return the currently achieved goal (for HER relabelling)."""
        achieved = self._compute_achieved_sector_weights(self.stock_weights)
        # Append realised regime approximation (use return sign heuristic)
        if len(self.return_history) > 0:
            recent_ret = np.mean(self.return_history[-5:])
            if recent_ret > 0.01:
                regime = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            elif recent_ret < -0.01:
                regime = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            else:
                regime = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            regime = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return np.concatenate([achieved, regime])
