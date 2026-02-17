"""
File: macro_env.py
Module: environment
Description: MacroEnv is a weekly-stepping Gymnasium environment for the Macro agent.
    State = macro features + sector GNN embeddings. Action = sector allocation weights.
    Reward combines sector alpha, portfolio metrics, regime accuracy, and value bonus.
Design Decisions: Weekly stepping aligns with institutional rebalancing frequency.
    Sector-level abstraction reduces action space vs stock-level, enabling faster PPO convergence.
References: PPO (Schulman 2017), Sector rotation strategies (Conover et al. 2008)
Author: HRL-SARP Framework
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces

from environment.base_env import BasePortfolioEnv
from environment.india_calendar import IndiaCalendar
from environment.reward_functions import compute_total_macro_reward

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# MACRO ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════


class MacroEnv(BasePortfolioEnv):
    """Weekly-stepping environment for the Macro (sector allocation) agent.

    Observation: macro_state (18D) + flattened sector GNN embeddings (11*64 = 704D) = 722D
    Action: sector weights (11D) via softmax + regime prediction (3D) = 14D continuous
    Reward: R_total_macro composite
    """

    def __init__(
        self,
        sector_returns_data: np.ndarray,
        benchmark_returns_data: np.ndarray,
        macro_states_data: np.ndarray,
        sector_gnn_embeddings_data: Optional[np.ndarray] = None,
        regime_labels: Optional[np.ndarray] = None,
        pe_zscores: Optional[np.ndarray] = None,
        macro_config_path: str = "config/macro_agent_config.yaml",
        risk_config_path: str = "config/risk_config.yaml",
        initial_capital: float = 1_00_00_000.0,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__(risk_config_path, initial_capital, render_mode)

        with open(macro_config_path, "r") as f:
            self.macro_cfg = yaml.safe_load(f)

        # Pre-loaded episode data (walk-forward slices)
        self.sector_returns_data = sector_returns_data  # (T, 11)
        self.benchmark_returns_data = benchmark_returns_data  # (T,)
        self.macro_states_data = macro_states_data  # (T, 18)

        # Optional: GNN embeddings, regime labels, PE z-scores
        self.num_sectors: int = self.macro_cfg["network"]["num_sectors"]
        self.sector_emb_dim: int = self.macro_cfg["network"]["sector_embedding_dim"]
        self.macro_state_dim: int = self.macro_cfg["network"]["macro_state_dim"]

        if sector_gnn_embeddings_data is not None:
            self.gnn_embeddings = sector_gnn_embeddings_data  # (T, 11, 64)
        else:
            self.gnn_embeddings = np.zeros(
                (len(sector_returns_data), self.num_sectors, self.sector_emb_dim),
                dtype=np.float32,
            )

        self.regime_labels = regime_labels  # (T,) ground truth
        self.pe_zscores = pe_zscores if pe_zscores is not None else np.zeros(len(sector_returns_data))

        # Total timesteps in this episode
        self.max_steps: int = len(sector_returns_data) - 1
        self.current_step: int = 0

        # Observation = macro_state + flattened GNN embeddings
        obs_dim = self.macro_state_dim + self.num_sectors * self.sector_emb_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Action = sector weights (11) + regime logits (3) = 14D
        action_dim = self.num_sectors + self.macro_cfg["network"]["actor"]["regime_classes"]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )

        # Reward config
        self.reward_cfg = self.macro_cfg["reward"]
        self.regime_cfg = self.macro_cfg["regime"]

        # Calendar for event risk
        self.calendar = IndiaCalendar(risk_config_path)

        # Sector weights tracking
        self.sector_weights = np.ones(self.num_sectors, dtype=np.float32) / self.num_sectors

        logger.info(
            "MacroEnv initialised | obs_dim=%d | action_dim=%d | max_steps=%d",
            obs_dim, action_dim, self.max_steps,
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
        self.sector_weights = np.ones(self.num_sectors, dtype=np.float32) / self.num_sectors

        obs = self._get_observation()
        info = {"step": 0, "portfolio_value": self.initial_capital}
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Parse action: first 11 dims = sector logits, last 3 = regime logits
        sector_logits = action[: self.num_sectors]
        regime_logits = action[self.num_sectors:]

        # Convert to weights via softmax
        new_sector_weights = self.softmax_weights(sector_logits)

        # Sector concentration cap
        max_sector_pct = self.risk_cfg["sector"]["max_single_sector_pct"]
        new_sector_weights = self.clip_weights(new_sector_weights, max_sector_pct)

        # Regime prediction (argmax)
        predicted_regime = int(np.argmax(regime_logits))

        # Get market data for this step
        sector_returns = self.sector_returns_data[self.current_step]
        benchmark_return = float(self.benchmark_returns_data[self.current_step])

        # Update portfolio
        turnover = float(np.sum(np.abs(new_sector_weights - self.sector_weights)))
        gross_return, cost = self.update_portfolio_value(sector_returns, new_sector_weights)
        self.sector_weights = new_sector_weights.copy()

        # Realised regime label
        realised_regime = 2  # default sideways
        if self.regime_labels is not None and self.current_step < len(self.regime_labels):
            realised_regime = int(self.regime_labels[self.current_step])
        else:
            # Heuristic: classify based on benchmark return
            if benchmark_return > self.regime_cfg["bull_threshold"]:
                realised_regime = 0
            elif benchmark_return < self.regime_cfg["bear_threshold"]:
                realised_regime = 1

        # Compute reward
        pe_z = float(self.pe_zscores[self.current_step]) if self.pe_zscores is not None else 0.0

        reward = compute_total_macro_reward(
            sector_returns=sector_returns,
            sector_weights=new_sector_weights,
            benchmark_return=benchmark_return,
            calmar_ratio=self.get_calmar(),
            cvar_95=self.get_cvar(),
            stt_cost=cost / max(self.portfolio_value, 1.0),
            predicted_regime=predicted_regime,
            realised_regime=realised_regime,
            pe_zscore=pe_z,
            weekly_return=gross_return,
            turnover_cost=turnover,
            w_macro=self.reward_cfg["w_macro"],
            w_portfolio=self.reward_cfg["w_portfolio"],
            w_regime=self.reward_cfg["w_regime"],
            w_value=self.reward_cfg["w_value_bonus"],
            herfindahl_coef=self.reward_cfg["herfindahl_penalty_coef"],
            turnover_coef=self.reward_cfg["turnover_penalty_coef"],
            regime_correct=self.regime_cfg["regime_reward_correct"],
            regime_incorrect=self.regime_cfg["regime_reward_incorrect"],
        )

        # Check termination
        self.current_step += 1
        self.step_count += 1
        terminated = self.current_step >= self.max_steps
        truncated = self.check_circuit_breaker()

        # Info dict
        info = {
            "step": self.current_step,
            "portfolio_value": self.portfolio_value,
            "drawdown": self.current_drawdown,
            "sector_weights": new_sector_weights.copy(),
            "predicted_regime": predicted_regime,
            "realised_regime": realised_regime,
            "gross_return": gross_return,
            "cost": cost,
            "turnover": turnover,
            "sharpe": self.get_sharpe(),
        }

        obs = self._get_observation()
        return obs, reward, terminated, truncated, info

    # ── Observation Construction ─────────────────────────────────────

    def _get_observation(self) -> np.ndarray:
        idx = min(self.current_step, len(self.macro_states_data) - 1)
        macro_state = self.macro_states_data[idx]  # (18,)
        gnn_emb = self.gnn_embeddings[idx].flatten()  # (11*64=704,)
        obs = np.concatenate([macro_state, gnn_emb]).astype(np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

    def _compute_reward(self, info: Dict[str, Any]) -> float:
        # Reward computed inline in step() for clarity
        return 0.0

    # ── Goal Output for Micro Agent ──────────────────────────────────

    def get_goal_for_micro(self) -> Dict[str, Any]:
        """Package current macro output as goal embedding for the Micro agent."""
        regime_one_hot = np.zeros(3, dtype=np.float32)
        # Use last predicted regime
        regime_one_hot[int(np.argmax(self.sector_weights[:3]))] = 1.0

        return {
            "sector_weights": self.sector_weights.copy(),
            "regime_one_hot": regime_one_hot,
            "goal_vector": np.concatenate([self.sector_weights, regime_one_hot]),
        }
