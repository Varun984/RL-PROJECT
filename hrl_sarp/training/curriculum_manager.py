"""
File: curriculum_manager.py
Module: training
Description: Curriculum learning manager that schedules training difficulty from
    easy (bull markets) through medium (sideways) to hard (bear/crisis) scenarios.
    Controls the mix of real vs synthetic data across training phases.
Design Decisions: Stage transitions based on performance thresholds (Sharpe, return).
    Uses MarketSimulator for synthetic data injection. Supports both automatic
    progression and manual stage override.
References: Bengio et al., "Curriculum Learning", 2009
Author: HRL-SARP Framework
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# CURRICULUM STAGES
# ══════════════════════════════════════════════════════════════════════


class CurriculumStage:
    """Definition for a single curriculum difficulty stage."""

    def __init__(
        self,
        name: str,
        difficulty: int,
        regime_mix: Dict[int, float],
        synthetic_ratio: float,
        min_episodes: int,
        promotion_sharpe: float,
        promotion_return: float,
    ) -> None:
        self.name = name
        self.difficulty = difficulty
        self.regime_mix = regime_mix          # {0: Bull%, 1: Bear%, 2: Sideways%}
        self.synthetic_ratio = synthetic_ratio  # Fraction of synthetic data
        self.min_episodes = min_episodes      # Min episodes before promotion check
        self.promotion_sharpe = promotion_sharpe
        self.promotion_return = promotion_return


# ══════════════════════════════════════════════════════════════════════
# CURRICULUM MANAGER
# ══════════════════════════════════════════════════════════════════════


class CurriculumManager:
    """Manages progressive difficulty training for HRL agents.

    Five curricula stages:
        Stage 0 (Warmup): Supervised pre-training on easy patterns
        Stage 1 (Easy):   70% bull, 20% sideways, 10% bear
        Stage 2 (Medium): 30% bull, 40% sideways, 30% bear
        Stage 3 (Hard):   10% bull, 20% sideways, 70% bear + stress scenarios
        Stage 4 (Full):   Historical distribution, no synthetic data
    """

    def __init__(
        self,
        config_path: str = "config/macro_agent_config.yaml",
    ) -> None:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        train_cfg = cfg.get("training", {})
        self.curriculum_enabled: bool = train_cfg.get("curriculum_enabled", True)

        # Define stages
        self.stages: List[CurriculumStage] = [
            CurriculumStage(
                name="warmup",
                difficulty=0,
                regime_mix={0: 1.0, 1: 0.0, 2: 0.0},
                synthetic_ratio=0.8,
                min_episodes=50,
                promotion_sharpe=0.0,
                promotion_return=-0.05,
            ),
            CurriculumStage(
                name="easy",
                difficulty=1,
                regime_mix={0: 0.7, 1: 0.1, 2: 0.2},
                synthetic_ratio=0.5,
                min_episodes=200,
                promotion_sharpe=0.5,
                promotion_return=0.02,
            ),
            CurriculumStage(
                name="medium",
                difficulty=2,
                regime_mix={0: 0.3, 1: 0.3, 2: 0.4},
                synthetic_ratio=0.3,
                min_episodes=300,
                promotion_sharpe=0.3,
                promotion_return=0.0,
            ),
            CurriculumStage(
                name="hard",
                difficulty=3,
                regime_mix={0: 0.1, 1: 0.7, 2: 0.2},
                synthetic_ratio=0.2,
                min_episodes=500,
                promotion_sharpe=0.2,
                promotion_return=-0.02,
            ),
            CurriculumStage(
                name="full",
                difficulty=4,
                regime_mix={0: 0.33, 1: 0.33, 2: 0.34},
                synthetic_ratio=0.0,
                min_episodes=1000,
                promotion_sharpe=float("inf"),  # Never auto-promote
                promotion_return=float("inf"),
            ),
        ]

        self.current_stage_idx: int = 0
        self.episodes_in_stage: int = 0
        self.total_episodes: int = 0

        # Performance tracking per stage
        self.stage_returns: List[float] = []
        self.stage_sharpes: List[float] = []

        logger.info("CurriculumManager initialised | enabled=%s", self.curriculum_enabled)

    @property
    def current_stage(self) -> CurriculumStage:
        return self.stages[self.current_stage_idx]

    @property
    def stage_name(self) -> str:
        return self.current_stage.name

    @property
    def is_final_stage(self) -> bool:
        return self.current_stage_idx >= len(self.stages) - 1

    # ── Episode Management ───────────────────────────────────────────

    def on_episode_end(
        self,
        episode_return: float,
        episode_sharpe: float,
    ) -> Dict[str, Any]:
        """Called at the end of each episode to update curriculum state.

        Returns:
            Dict with 'promoted', 'stage', 'difficulty' keys.
        """
        self.episodes_in_stage += 1
        self.total_episodes += 1
        self.stage_returns.append(episode_return)
        self.stage_sharpes.append(episode_sharpe)

        promoted = False
        if self.curriculum_enabled and not self.is_final_stage:
            promoted = self._check_promotion()

        return {
            "promoted": promoted,
            "stage": self.stage_name,
            "difficulty": self.current_stage.difficulty,
            "episodes_in_stage": self.episodes_in_stage,
            "total_episodes": self.total_episodes,
        }

    def _check_promotion(self) -> bool:
        """Check if agent should be promoted to next difficulty stage."""
        stage = self.current_stage

        # Must complete minimum episodes
        if self.episodes_in_stage < stage.min_episodes:
            return False

        # Check rolling performance (last 50 episodes)
        window = min(50, len(self.stage_returns))
        recent_returns = self.stage_returns[-window:]
        recent_sharpes = self.stage_sharpes[-window:]

        avg_return = float(np.mean(recent_returns))
        avg_sharpe = float(np.mean(recent_sharpes))

        if avg_sharpe >= stage.promotion_sharpe and avg_return >= stage.promotion_return:
            self._promote()
            return True

        return False

    def _promote(self) -> None:
        """Move to the next curriculum stage."""
        old_name = self.stage_name
        self.current_stage_idx += 1
        self.episodes_in_stage = 0
        self.stage_returns.clear()
        self.stage_sharpes.clear()

        logger.info(
            "Curriculum promotion: %s → %s (after %d total episodes)",
            old_name, self.stage_name, self.total_episodes,
        )

    # ── Data Sampling ────────────────────────────────────────────────

    def get_regime_for_episode(self) -> int:
        """Sample a regime for the next episode based on current stage mix."""
        mix = self.current_stage.regime_mix
        regimes = list(mix.keys())
        probs = [mix[r] for r in regimes]
        return int(np.random.choice(regimes, p=probs))

    def should_use_synthetic(self) -> bool:
        """Determine if next episode should use synthetic data."""
        return np.random.random() < self.current_stage.synthetic_ratio

    def get_data_split_config(self) -> Dict[str, Any]:
        """Return data configuration for current curriculum stage."""
        return {
            "regime_mix": self.current_stage.regime_mix,
            "synthetic_ratio": self.current_stage.synthetic_ratio,
            "difficulty": self.current_stage.difficulty,
            "stage_name": self.stage_name,
        }

    # ── Manual Control ───────────────────────────────────────────────

    def set_stage(self, stage_idx: int) -> None:
        """Manually set the curriculum stage."""
        if 0 <= stage_idx < len(self.stages):
            self.current_stage_idx = stage_idx
            self.episodes_in_stage = 0
            self.stage_returns.clear()
            self.stage_sharpes.clear()
            logger.info("Curriculum manually set to stage %d: %s", stage_idx, self.stage_name)

    def reset(self) -> None:
        """Reset curriculum to initial state."""
        self.current_stage_idx = 0
        self.episodes_in_stage = 0
        self.total_episodes = 0
        self.stage_returns.clear()
        self.stage_sharpes.clear()

    # ── Reporting ────────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        """Return current curriculum status for logging."""
        return {
            "stage_idx": self.current_stage_idx,
            "stage_name": self.stage_name,
            "difficulty": self.current_stage.difficulty,
            "episodes_in_stage": self.episodes_in_stage,
            "total_episodes": self.total_episodes,
            "synthetic_ratio": self.current_stage.synthetic_ratio,
            "regime_mix": self.current_stage.regime_mix,
        }
