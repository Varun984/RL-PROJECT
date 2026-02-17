"""
File: joint_finetune.py
Module: training
Description: Phase 5 — Joint fine-tuning of both Macro and Micro agents with
    alternating gradient updates. Both agents are unfrozen and trained together
    in the full hierarchical environment.
Design Decisions: Alternating updates (Macro for M steps, then Micro for N steps)
    prevent the co-adaptation instability of simultaneous updates. Lower learning
    rates than individual phases. Gradient scaling reduces Macro LR to keep its
    slowly-changing goals stable as Micro adapts.
References: HIRO (Nachum 2018), HAM (Vezhnevets 2017)
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


def joint_finetune(
    macro_agent: MacroAgent,
    micro_agent: MicroAgent,
    env,
    val_env=None,
    macro_config_path: str = "config/macro_agent_config.yaml",
    micro_config_path: str = "config/micro_agent_config.yaml",
    log_dir: str = "logs/phase5_joint",
    seed: int = 42,
) -> Dict[str, Any]:
    """Phase 5: Joint fine-tuning of Macro + Micro agents.

    Training protocol:
        1. Both agents unfrozen with reduced learning rates (0.1x)
        2. Run full hierarchical episodes
        3. Alternate: collect Macro rollout → PPO update → collect Micro data → TD3 update
        4. Macro updates every macro_update_interval episodes
        5. Micro updates every step (off-policy)

    Args:
        macro_agent: Pre-trained MacroAgent.
        micro_agent: Pre-trained MicroAgent.
        env: HierarchicalEnv wrapping both agent environments.
        val_env: Optional validation environment.
        macro_config_path: Macro config path.
        micro_config_path: Micro config path.
        log_dir: Directory for logs.
        seed: Random seed.

    Returns:
        Training summary dict.
    """
    set_global_seed(seed)
    os.makedirs(log_dir, exist_ok=True)

    with open(macro_config_path, "r") as f:
        macro_cfg = yaml.safe_load(f)
    with open(micro_config_path, "r") as f:
        micro_cfg = yaml.safe_load(f)

    # Joint training parameters
    joint_cfg = macro_cfg.get("training", {}).get("joint", {})
    total_episodes: int = joint_cfg.get("total_episodes", 5000)
    macro_update_interval: int = joint_cfg.get("macro_update_interval", 10)
    micro_updates_per_step: int = joint_cfg.get("micro_updates_per_step", 1)
    lr_scale: float = joint_cfg.get("lr_scale", 0.1)
    eval_interval: int = joint_cfg.get("eval_interval", 50)
    save_interval: int = joint_cfg.get("save_interval", 200)

    # Reduce learning rates for fine-tuning stability
    _scale_learning_rate(macro_agent.optimizer, lr_scale)
    _scale_learning_rate(micro_agent.actor_optimizer, lr_scale)
    _scale_learning_rate(micro_agent.critic_optimizer, lr_scale)

    # Ensure both agents are unfrozen
    macro_agent.unfreeze()
    micro_agent.unfreeze()

    tracker = MetricsTracker(log_dir=log_dir)
    early_stop = EarlyStopping(patience=50, mode="max")
    curriculum = CurriculumManager(config_path=macro_config_path)
    # Start curriculum at the hard stage for joint fine-tuning
    curriculum.set_stage(3)

    best_eval_return = float("-inf")
    best_path_macro = os.path.join(log_dir, "best_joint_macro.pt")
    best_path_micro = os.path.join(log_dir, "best_joint_micro.pt")
    global_step = 0

    logger.info(
        "Phase 5: Joint fine-tuning | episodes=%d | LR_scale=%.2f",
        total_episodes, lr_scale,
    )

    for episode in range(1, total_episodes + 1):
        obs, info = env.reset()
        done = False
        episode_return = 0.0
        episode_macro_steps = 0
        episode_micro_steps = 0

        # Track whether we're in a macro or micro step
        macro_acting = True
        current_macro_goal = None

        while not done:
            if _is_macro_step(info):
                # ── Macro step (weekly) ──
                macro_state, sector_emb = _split_macro_obs(obs, macro_agent)
                action, log_prob, value = macro_agent.select_action(
                    macro_state, sector_emb
                )

                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                macro_agent.store_transition(
                    obs=obs,
                    action=action,
                    reward=reward,
                    value=value,
                    log_prob=log_prob,
                    done=done,
                    sector_embedding=sector_emb,
                )

                current_macro_goal = macro_agent.get_goal_embedding(action)
                episode_macro_steps += 1
            else:
                # ── Micro step (daily) ──
                if current_macro_goal is None:
                    current_macro_goal = _default_goal(macro_agent)

                stock_features, goal, mask = _parse_micro_obs_joint(
                    obs, micro_agent, current_macro_goal
                )
                action = micro_agent.select_action(
                    stock_features, current_macro_goal, mask
                )

                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Achieved goal from environment
                achieved_goal = info.get(
                    "achieved_sector_weights", current_macro_goal
                )

                flat_state = _flatten_state(obs, current_macro_goal, micro_agent)
                flat_next = _flatten_state(next_obs, current_macro_goal, micro_agent)

                micro_agent.store_transition(
                    state=flat_state,
                    action=action,
                    reward=reward,
                    next_state=flat_next,
                    done=done,
                    desired_goal=current_macro_goal,
                    achieved_goal=achieved_goal,
                )

                # Micro TD3 updates (off-policy)
                for _ in range(micro_updates_per_step):
                    micro_metrics = micro_agent.update()
                    if micro_metrics:
                        tracker.update(micro_metrics, step=global_step)

                episode_micro_steps += 1

            obs = next_obs
            episode_return += reward
            global_step += 1

        # ── Macro PPO update (periodic) ──
        if episode % macro_update_interval == 0 and macro_agent.buffer.full:
            # Bootstrap
            if not done:
                ms, se = _split_macro_obs(next_obs, macro_agent)
                _, _, last_value = macro_agent.select_action(ms, se, deterministic=True)
            else:
                last_value = 0.0

            macro_update_metrics = macro_agent.update(last_value, done)
            tracker.update(
                {f"macro/{k}": v for k, v in macro_update_metrics.items()},
                step=global_step,
            )

        # Episode metrics
        episode_sharpe = info.get("sharpe", 0.0)
        ep_metrics = {
            "joint/return": episode_return,
            "joint/sharpe": episode_sharpe,
            "joint/drawdown": info.get("drawdown", 0.0),
            "joint/macro_steps": episode_macro_steps,
            "joint/micro_steps": episode_micro_steps,
        }
        tracker.update(ep_metrics, step=global_step)

        # Curriculum
        curriculum.on_episode_end(episode_return, episode_sharpe)

        # Periodic evaluation
        if val_env is not None and episode % eval_interval == 0:
            eval_metrics = _evaluate_joint(
                macro_agent, micro_agent, val_env
            )
            tracker.update(eval_metrics, step=global_step)

            if eval_metrics["joint_eval/return_mean"] > best_eval_return:
                best_eval_return = eval_metrics["joint_eval/return_mean"]
                macro_agent.save(best_path_macro)
                micro_agent.save(best_path_micro)

            if early_stop.step(eval_metrics.get("joint_eval/sharpe_mean", 0.0)):
                logger.info("Early stopping at episode %d", episode)
                break

        # Periodic save
        if episode % save_interval == 0:
            macro_agent.save(os.path.join(log_dir, f"joint_macro_ep{episode}.pt"))
            micro_agent.save(os.path.join(log_dir, f"joint_micro_ep{episode}.pt"))

        if episode % 20 == 0:
            logger.info(
                "Joint Ep %d | return=%.4f | sharpe=%.4f | macro=%d micro=%d steps",
                episode, episode_return, episode_sharpe,
                episode_macro_steps, episode_micro_steps,
            )

    tracker.save("phase5_joint_metrics.json")

    return {
        "episodes_trained": episode,
        "total_steps": global_step,
        "best_eval_return": best_eval_return,
        "best_macro_checkpoint": best_path_macro,
        "best_micro_checkpoint": best_path_micro,
        "curriculum_status": curriculum.get_status(),
        "metrics_summary": tracker.summary(),
    }


# ══════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════


def _scale_learning_rate(optimizer, scale: float) -> None:
    """Scale all param group learning rates."""
    for pg in optimizer.param_groups:
        pg["lr"] *= scale
    logger.info("LR scaled by %.2f → new LR=%.6f", scale, optimizer.param_groups[0]["lr"])


def _is_macro_step(info: Dict) -> bool:
    """Determine if the current step should be a Macro decision.

    HierarchicalEnv sets 'is_macro_step' flag in info.
    """
    return info.get("is_macro_step", False)


def _split_macro_obs(obs: np.ndarray, macro_agent: MacroAgent):
    """Split flat observation into macro_state and sector_embeddings."""
    macro_state = obs[:macro_agent.macro_state_dim]
    sector_emb = obs[macro_agent.macro_state_dim:].reshape(
        macro_agent.num_sectors, macro_agent.sector_emb_dim
    )
    return macro_state, sector_emb


def _parse_micro_obs_joint(obs, micro_agent, goal):
    """Parse micro observation for joint training."""
    max_s = micro_agent.max_stocks
    feat_dim = micro_agent.stock_feature_dim
    stock_flat = obs[:max_s * feat_dim]
    stock_features = stock_flat.reshape(max_s, feat_dim)
    mask = (np.abs(stock_features).sum(axis=-1) > 1e-6).astype(np.float32)
    return stock_features, goal, mask


def _flatten_state(obs, goal, micro_agent):
    """Flatten observation + goal for buffer."""
    max_s = micro_agent.max_stocks
    feat_dim = micro_agent.stock_feature_dim
    stock_flat = obs[:max_s * feat_dim]
    return np.concatenate([stock_flat, goal]).astype(np.float32)


def _default_goal(macro_agent: MacroAgent) -> np.ndarray:
    """Generate a default uniform goal."""
    n_sectors = macro_agent.num_sectors
    n_regime = macro_agent.regime_classes
    goal = np.zeros(n_sectors + n_regime, dtype=np.float32)
    goal[:n_sectors] = 1.0 / n_sectors
    goal[n_sectors] = 1.0
    return goal


@torch.no_grad()
def _evaluate_joint(
    macro_agent: MacroAgent,
    micro_agent: MicroAgent,
    env,
    n_episodes: int = 5,
) -> Dict[str, float]:
    """Evaluate the joint Macro + Micro system."""
    returns = []
    sharpes = []
    drawdowns = []

    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_return = 0.0
        current_goal = None

        while not done:
            if _is_macro_step(info):
                ms, se = _split_macro_obs(obs, macro_agent)
                action, _, _ = macro_agent.select_action(ms, se, deterministic=True)
                current_goal = macro_agent.get_goal_embedding(action)
                obs, reward, terminated, truncated, info = env.step(action)
            else:
                if current_goal is None:
                    current_goal = _default_goal(macro_agent)
                sf, g, m = _parse_micro_obs_joint(obs, micro_agent, current_goal)
                action = micro_agent.select_action(sf, current_goal, m, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)

            ep_return += reward
            done = terminated or truncated

        returns.append(ep_return)
        sharpes.append(info.get("sharpe", 0.0))
        drawdowns.append(info.get("drawdown", 0.0))

    return {
        "joint_eval/return_mean": float(np.mean(returns)),
        "joint_eval/return_std": float(np.std(returns)),
        "joint_eval/sharpe_mean": float(np.mean(sharpes)),
        "joint_eval/max_drawdown": float(np.max(drawdowns)),
    }
