"""
File: train_micro_frozen_macro.py
Module: training
Description: Phase 4 — RL training of the Micro agent (TD3+HER) with the Macro agent's
    weights frozen. The Micro learns to execute stock-level allocations that best
    implement the fixed Macro policy's sector allocation goals.
Design Decisions: TD3 off-policy training allows efficient use of the HER replay buffer.
    Freezing Macro provides stable goals, removing goal non-stationarity.
    HER relabelling creates k=4 additional training samples per transition.
References: TD3 (Fujimoto 2018), HER (Andrychowicz 2017)
Author: HRL-SARP Framework
"""

import logging
import os
from typing import Any, Dict, Optional

import numpy as np
import yaml

from agents.macro_agent import MacroAgent
from agents.micro_agent import MicroAgent
from training.curriculum_manager import CurriculumManager
from training.trainer_utils import (
    EarlyStopping,
    MetricsTracker,
    save_checkpoint,
    set_global_seed,
)

logger = logging.getLogger(__name__)


def train_micro_frozen_macro(
    micro_agent: MicroAgent,
    macro_agent: MacroAgent,
    env,
    val_env=None,
    config_path: str = "config/micro_agent_config.yaml",
    log_dir: str = "logs/phase4_micro",
    seed: int = 42,
) -> Dict[str, Any]:
    """Phase 4: RL training of Micro agent with frozen Macro.

    Training loop:
        1. Macro (frozen) produces sector allocation goal at start of each week
        2. Micro selects daily stock weights to implement goal
        3. Transitions stored in HER buffer with desired + achieved goals
        4. TD3 updates after each step (off-policy)

    Args:
        micro_agent: MicroAgent to train (TD3+HER).
        macro_agent: MacroAgent (frozen, provides goals).
        env: MicroEnv or HierarchicalEnv instance.
        val_env: Optional validation environment.
        config_path: Path to micro config.
        log_dir: Directory for logs.
        seed: Random seed.

    Returns:
        Training summary dict.
    """
    set_global_seed(seed)
    os.makedirs(log_dir, exist_ok=True)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg.get("training", {})
    total_timesteps: int = train_cfg.get("total_timesteps", 1_000_000)
    warmup_steps: int = train_cfg.get("warmup_steps", 10_000)
    update_every: int = train_cfg.get("update_every", 1)
    eval_interval: int = train_cfg.get("eval_interval_episodes", 50)
    save_interval: int = train_cfg.get("save_interval_episodes", 100)

    # Freeze Macro
    macro_agent.freeze()

    tracker = MetricsTracker(log_dir=log_dir)
    early_stop = EarlyStopping(patience=30, mode="max")

    best_eval_return = float("-inf")
    best_path = os.path.join(log_dir, "best_micro.pt")
    episode_count = 0
    global_step = 0

    logger.info(
        "Phase 4: Micro RL training (Macro frozen) | total_steps=%d | warmup=%d",
        total_timesteps, warmup_steps,
    )

    while global_step < total_timesteps:
        obs, info = env.reset()
        done = False
        episode_return = 0.0
        episode_steps = 0

        # Get Macro goal for this episode (from frozen Macro)
        macro_goal = _get_macro_goal(macro_agent, info)

        while not done and global_step < total_timesteps:
            # Parse observation
            stock_features, goal, mask = _parse_micro_obs(
                obs, micro_agent, macro_goal
            )

            # Select action
            if global_step < warmup_steps:
                # Random exploration during warmup
                action = np.random.dirichlet(
                    np.ones(micro_agent.max_stocks) * 0.5
                ).astype(np.float32)
                action *= mask
                valid_sum = action.sum()
                if valid_sum > 1e-8:
                    action /= valid_sum
            else:
                action = micro_agent.select_action(
                    stock_features, macro_goal, mask, deterministic=False
                )

            # Step environment
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated

            # Get achieved goal (actual sector allocation achieved)
            achieved_goal = step_info.get(
                "achieved_sector_weights",
                macro_goal  # Fallback
            )

            # Flatten state for buffer storage
            flat_state = _flatten_micro_state(obs, macro_goal, micro_agent)
            flat_next = _flatten_micro_state(next_obs, macro_goal, micro_agent)

            # Store transition
            micro_agent.store_transition(
                state=flat_state,
                action=action,
                reward=reward,
                next_state=flat_next,
                done=done,
                desired_goal=macro_goal,
                achieved_goal=achieved_goal,
            )

            # TD3 update (off-policy, can update every step)
            if global_step >= warmup_steps and global_step % update_every == 0:
                update_metrics = micro_agent.update()
                if update_metrics:
                    tracker.update(update_metrics, step=global_step)

            obs = next_obs
            episode_return += reward
            episode_steps += 1
            global_step += 1

        # Episode completed
        episode_count += 1
        episode_sharpe = step_info.get("sharpe", 0.0) if step_info else 0.0

        ep_metrics = {
            "episode/return": episode_return,
            "episode/length": episode_steps,
            "episode/sharpe": episode_sharpe,
            "episode/goal_alignment": step_info.get("goal_alignment", 0.0) if step_info else 0.0,
            "episode/exploration_noise": micro_agent._get_exploration_noise(),
        }
        tracker.update(ep_metrics, step=global_step)

        # Periodic evaluation
        if val_env is not None and episode_count % eval_interval == 0:
            eval_metrics = _evaluate_micro(micro_agent, macro_agent, val_env)
            tracker.update(eval_metrics, step=global_step)

            if eval_metrics["eval_return_mean"] > best_eval_return:
                best_eval_return = eval_metrics["eval_return_mean"]
                micro_agent.save(best_path)

            if early_stop.step(eval_metrics.get("eval_sharpe_mean", 0.0)):
                logger.info("Early stopping at episode %d", episode_count)
                break

        # Periodic save
        if episode_count % save_interval == 0:
            ckpt_path = os.path.join(log_dir, f"micro_ep{episode_count}.pt")
            micro_agent.save(ckpt_path)

        if episode_count % 20 == 0:
            logger.info(
                "Episode %d | return=%.4f | sharpe=%.4f | noise=%.4f | steps=%d",
                episode_count, episode_return, episode_sharpe,
                micro_agent._get_exploration_noise(), global_step,
            )

    tracker.save("phase4_micro_metrics.json")
    macro_agent.unfreeze()

    return {
        "episodes_trained": episode_count,
        "total_steps": global_step,
        "best_eval_return": best_eval_return,
        "best_checkpoint": best_path,
        "metrics_summary": tracker.summary(),
    }


# ── Helpers ──────────────────────────────────────────────────────────


def _get_macro_goal(macro_agent: MacroAgent, info: Dict) -> np.ndarray:
    """Get goal from frozen Macro agent based on env info."""
    if "macro_state" in info and "sector_embeddings" in info:
        action, _, _ = macro_agent.select_action(
            info["macro_state"],
            info["sector_embeddings"],
            deterministic=True,
        )
        return macro_agent.get_goal_embedding(action)
    else:
        # Default uniform goal
        n_sectors = macro_agent.num_sectors
        n_regime = macro_agent.regime_classes
        goal = np.zeros(n_sectors + n_regime, dtype=np.float32)
        goal[:n_sectors] = 1.0 / n_sectors
        goal[n_sectors] = 1.0  # Bull regime
        return goal


def _parse_micro_obs(
    obs: np.ndarray,
    micro_agent: MicroAgent,
    goal: np.ndarray,
) -> tuple:
    """Parse flat observation into stock features + goal + mask."""
    max_s = micro_agent.max_stocks
    feat_dim = micro_agent.stock_feature_dim
    stock_flat = obs[:max_s * feat_dim]
    stock_features = stock_flat.reshape(max_s, feat_dim)
    mask = (np.abs(stock_features).sum(axis=-1) > 1e-6).astype(np.float32)
    return stock_features, goal, mask


def _flatten_micro_state(
    obs: np.ndarray,
    goal: np.ndarray,
    micro_agent: MicroAgent,
) -> np.ndarray:
    """Flatten observation + goal into a single vector for buffer storage."""
    max_s = micro_agent.max_stocks
    feat_dim = micro_agent.stock_feature_dim
    stock_flat = obs[:max_s * feat_dim]
    return np.concatenate([stock_flat, goal]).astype(np.float32)


@torch.no_grad()
def _evaluate_micro(
    micro_agent: MicroAgent,
    macro_agent: MacroAgent,
    env,
    n_episodes: int = 5,
) -> Dict[str, float]:
    """Evaluate Micro agent over multiple episodes."""
    returns = []
    sharpes = []
    goal_alignments = []

    for _ in range(n_episodes):
        obs, info = env.reset()
        macro_goal = _get_macro_goal(macro_agent, info)
        done = False
        ep_return = 0.0

        while not done:
            stock_features, goal, mask = _parse_micro_obs(obs, micro_agent, macro_goal)
            action = micro_agent.select_action(stock_features, macro_goal, mask, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            done = terminated or truncated

        returns.append(ep_return)
        sharpes.append(info.get("sharpe", 0.0))
        goal_alignments.append(info.get("goal_alignment", 0.0))

    return {
        "eval_return_mean": float(np.mean(returns)),
        "eval_return_std": float(np.std(returns)),
        "eval_sharpe_mean": float(np.mean(sharpes)),
        "eval_goal_alignment": float(np.mean(goal_alignments)),
    }
