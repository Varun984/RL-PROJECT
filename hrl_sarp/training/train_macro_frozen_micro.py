"""
File: train_macro_frozen_micro.py
Module: training
Description: Phase 3 — RL training of the Macro agent (PPO) with the Micro agent's
    weights frozen. The Macro learns to produce sector allocation goals that lead
    to good portfolio outcomes when executed by the fixed Micro policy.
Design Decisions: Freezing Micro eliminates non-stationarity in the Macro's environment.
    Uses the full HierarchicalEnv wrapper: Macro acts weekly, Micro executes daily.
    PPO rollouts collected over n_steps weeks, then batched update.
References: HRL training stabilisation (Nachum 2018), Option-Critic (Bacon 2017)
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
    evaluate_agent,
    save_checkpoint,
    set_global_seed,
)

logger = logging.getLogger(__name__)


def train_macro_frozen_micro(
    macro_agent: MacroAgent,
    micro_agent: MicroAgent,
    env,
    val_env=None,
    config_path: str = "config/macro_agent_config.yaml",
    log_dir: str = "logs/phase3_macro",
    seed: int = 42,
) -> Dict[str, Any]:
    """Phase 3: RL training of Macro agent with frozen Micro.

    Training loop:
        1. Reset hierarchical environment
        2. Macro produces weekly sector allocation goals
        3. Frozen Micro executes daily within those goals
        4. Macro receives composite reward (sector alpha + portfolio metrics)
        5. PPO update after n_steps rollouts

    Args:
        macro_agent: MacroAgent to train.
        micro_agent: MicroAgent (frozen).
        env: HierarchicalEnv or MacroEnv instance.
        val_env: Optional validation environment.
        config_path: Path to macro config.
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
    total_timesteps: int = train_cfg.get("total_timesteps", 500_000)
    eval_interval: int = train_cfg.get("eval_interval_episodes", 50)
    save_interval: int = train_cfg.get("save_interval_episodes", 100)

    # Freeze Micro
    micro_agent.freeze()

    n_steps = macro_agent.n_steps
    tracker = MetricsTracker(log_dir=log_dir)
    early_stop = EarlyStopping(patience=30, mode="max")
    curriculum = CurriculumManager(config_path=config_path)

    best_eval_return = float("-inf")
    best_path = os.path.join(log_dir, "best_macro.pt")
    episode_count = 0
    global_step = 0

    logger.info(
        "Phase 3: Macro RL training (Micro frozen) | total_steps=%d",
        total_timesteps,
    )

    while global_step < total_timesteps:
        obs, info = env.reset()
        done = False
        episode_return = 0.0
        episode_steps = 0

        while not done and global_step < total_timesteps:
            # Parse observation into macro_state + sector embeddings
            macro_state = obs[:macro_agent.macro_state_dim]
            sector_emb = obs[macro_agent.macro_state_dim:].reshape(
                macro_agent.num_sectors, macro_agent.sector_emb_dim
            )

            # Select action from Macro policy
            action, log_prob, value = macro_agent.select_action(macro_state, sector_emb)

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store in rollout buffer
            macro_agent.store_transition(
                obs=obs,
                action=action,
                reward=reward,
                value=value,
                log_prob=log_prob,
                done=done,
                sector_embedding=sector_emb,
            )

            obs = next_obs
            episode_return += reward
            episode_steps += 1
            global_step += 1

            # PPO update when buffer is full
            if macro_agent.buffer.full:
                # Bootstrap value for incomplete trajectory
                if not done:
                    ms = next_obs[:macro_agent.macro_state_dim]
                    se = next_obs[macro_agent.macro_state_dim:].reshape(
                        macro_agent.num_sectors, macro_agent.sector_emb_dim
                    )
                    _, _, last_value = macro_agent.select_action(ms, se, deterministic=True)
                else:
                    last_value = 0.0

                update_metrics = macro_agent.update(last_value, done)
                tracker.update(update_metrics, step=global_step)

        # Episode completed
        episode_count += 1
        episode_sharpe = info.get("sharpe", 0.0)

        ep_metrics = {
            "episode/return": episode_return,
            "episode/length": episode_steps,
            "episode/sharpe": episode_sharpe,
            "episode/drawdown": info.get("drawdown", 0.0),
        }
        tracker.update(ep_metrics, step=global_step)

        # Curriculum update
        curriculum.on_episode_end(episode_return, episode_sharpe)

        # Periodic evaluation
        if val_env is not None and episode_count % eval_interval == 0:
            eval_metrics = evaluate_agent(val_env, macro_agent, n_episodes=5)
            tracker.update(eval_metrics, step=global_step)

            if eval_metrics["eval_return_mean"] > best_eval_return:
                best_eval_return = eval_metrics["eval_return_mean"]
                macro_agent.save(best_path)

            if early_stop.step(eval_metrics["eval_sharpe_mean"]):
                logger.info("Early stopping at episode %d", episode_count)
                break

        # Periodic save
        if episode_count % save_interval == 0:
            ckpt_path = os.path.join(log_dir, f"macro_ep{episode_count}.pt")
            macro_agent.save(ckpt_path)

        if episode_count % 20 == 0:
            logger.info(
                "Episode %d | return=%.4f | sharpe=%.4f | steps=%d | stage=%s",
                episode_count, episode_return, episode_sharpe,
                global_step, curriculum.stage_name,
            )

    tracker.save("phase3_macro_metrics.json")
    micro_agent.unfreeze()

    return {
        "episodes_trained": episode_count,
        "total_steps": global_step,
        "best_eval_return": best_eval_return,
        "best_checkpoint": best_path,
        "curriculum_status": curriculum.get_status(),
        "metrics_summary": tracker.summary(),
    }
