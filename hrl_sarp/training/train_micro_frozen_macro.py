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
import re
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

from agents.macro_agent import MacroAgent
from agents.micro_agent import MicroAgent
from training.curriculum_manager import CurriculumManager
from training.trainer_utils import (
    EarlyStopping,
    MetricsTracker,
    set_global_seed,
)

logger = logging.getLogger(__name__)


def _get_project_root() -> str:
    """Get hrl_sarp project root directory for config paths."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _find_latest_macro_checkpoint(phase3_log_dir: str) -> Optional[str]:
    """Find latest phase3 macro checkpoint by episode number."""
    if not os.path.isdir(phase3_log_dir):
        return None
    best_path = None
    best_ep = -1
    for name in os.listdir(phase3_log_dir):
        if not (name.startswith("macro_ep") and name.endswith(".pt")):
            continue
        ep_str = name[len("macro_ep"):-len(".pt")]
        if not ep_str.isdigit():
            continue
        ep = int(ep_str)
        if ep > best_ep:
            best_ep = ep
            best_path = os.path.join(phase3_log_dir, name)
    return best_path


def _extract_episode_from_micro_checkpoint_name(path: str) -> int:
    """Extract episode number from checkpoint name like micro_ep600.pt."""
    m = re.search(r"micro_ep(\d+)\.pt$", os.path.basename(path))
    return int(m.group(1)) if m else 0


def _find_latest_phase4_checkpoint(log_dir: str) -> Optional[str]:
    """Return newest Phase 4 checkpoint by episode number, if any."""
    if not os.path.isdir(log_dir):
        return None
    best_path = None
    best_ep = -1
    for name in os.listdir(log_dir):
        m = re.match(r"micro_ep(\d+)\.pt$", name)
        if not m:
            continue
        ep = int(m.group(1))
        if ep > best_ep:
            best_ep = ep
            best_path = os.path.join(log_dir, name)
    return best_path


class GoalConditionedMicroEnv:
    """Wrap MicroEnv and inject macro_state/sector_embeddings into reset info.

    This keeps Phase 4 helpers unchanged: they can derive Macro goals from `info`.
    """

    def __init__(
        self,
        micro_env,
        macro_states: np.ndarray,
        sector_embeddings: np.ndarray,
        seed: int = 42,
    ) -> None:
        self.micro_env = micro_env
        self.macro_states = macro_states
        self.sector_embeddings = sector_embeddings
        self.rng = np.random.default_rng(seed)
        self.observation_space = micro_env.observation_space
        self.action_space = micro_env.action_space

    def reset(self):
        obs, info = self.micro_env.reset()
        if len(self.macro_states) > 0:
            idx = int(self.rng.integers(0, len(self.macro_states)))
            info["macro_state"] = self.macro_states[idx]
            info["sector_embeddings"] = self.sector_embeddings[idx]
        return obs, info

    def step(self, action):
        return self.micro_env.step(action)


def train_micro_frozen_macro(
    configs: Dict[str, Any],
    device: torch.device,
    seed: int = 42,
) -> Dict[str, Any]:
    """Phase 4: RL training of Micro agent with frozen Macro.
    
    This is a wrapper that initializes agents and environment, then calls
    the actual training function.

    Args:
        configs: Dictionary containing all config files (macro, micro, data, risk).
        device: Torch device (cpu or cuda).
        seed: Random seed.

    Returns:
        Training summary dict.
    """
    set_global_seed(seed)
    
    logger.info("Initializing agents and environment for Phase 4...")
    root = _get_project_root()

    from data.training_data_loader import TrainingDataLoader
    from environment.micro_env import MicroEnv

    # Config paths
    macro_config = os.path.join(root, "config", "macro_agent_config.yaml")
    micro_config = os.path.join(root, "config", "micro_agent_config.yaml")
    data_config_path = os.path.join(root, "config", "data_config.yaml")
    risk_config = os.path.join(root, "config", "risk_config.yaml")

    # Initialize agents
    logger.info("Initializing MacroAgent...")
    macro_agent = MacroAgent(config_path=macro_config, device=device)

    logger.info("Initializing MicroAgent...")
    micro_agent = MicroAgent(config_path=micro_config, device=device)

    # Resume settings from micro config (loaded by main.py)
    micro_cfg = configs.get("micro_agent_config", {})
    train_cfg = micro_cfg.get("training", {})
    resume_phase4 = bool(train_cfg.get("resume_phase4", True))
    resume_checkpoint_cfg = train_cfg.get("resume_checkpoint", None)
    initial_episode_count = 0
    initial_global_step = 0
    phase4_log_dir = os.path.join(root, "logs", "phase4_micro")

    # Optional resume from latest Phase 4 checkpoint
    resume_checkpoint_path: Optional[str] = None
    if resume_phase4:
        if isinstance(resume_checkpoint_cfg, str) and resume_checkpoint_cfg.strip():
            candidate = resume_checkpoint_cfg.strip()
            if not os.path.isabs(candidate):
                candidate = os.path.join(root, candidate)
            if os.path.exists(candidate):
                resume_checkpoint_path = candidate
            else:
                logger.warning("Configured resume_checkpoint not found: %s", candidate)
        if resume_checkpoint_path is None:
            resume_checkpoint_path = _find_latest_phase4_checkpoint(phase4_log_dir)

    # Load Macro weights (prefer Phase 3 checkpoints, fallback to Phase 1)
    phase3_log_dir = os.path.join(root, "logs", "phase3_macro")
    macro_ckpt = os.path.join(phase3_log_dir, "best_macro.pt")
    if not os.path.exists(macro_ckpt):
        latest = _find_latest_macro_checkpoint(phase3_log_dir)
        if latest is not None:
            macro_ckpt = latest
    if os.path.exists(macro_ckpt):
        try:
            macro_agent.load(macro_ckpt)
            logger.info("Loaded Macro checkpoint: %s", macro_ckpt)
        except Exception:
            checkpoint = torch.load(macro_ckpt, map_location=device)
            if "model_actor" in checkpoint:
                macro_agent.actor.load_state_dict(checkpoint["model_actor"])
            elif "models" in checkpoint and "actor" in checkpoint["models"]:
                macro_agent.actor.load_state_dict(checkpoint["models"]["actor"])
            elif "actor" in checkpoint:
                macro_agent.actor.load_state_dict(checkpoint["actor"])
            else:
                raise KeyError(f"Checkpoint missing actor weights. Keys: {list(checkpoint.keys())}")
            logger.info("Loaded Macro actor weights from: %s", macro_ckpt)
    else:
        pretrain_macro_ckpt = os.path.join(root, "logs", "pretrain_macro", "best_pretrain_macro.pt")
        if os.path.exists(pretrain_macro_ckpt):
            checkpoint = torch.load(pretrain_macro_ckpt, map_location=device)
            if "model_actor" in checkpoint:
                macro_agent.actor.load_state_dict(checkpoint["model_actor"])
            elif "models" in checkpoint and "actor" in checkpoint["models"]:
                macro_agent.actor.load_state_dict(checkpoint["models"]["actor"])
            else:
                raise KeyError(f"Checkpoint missing actor weights. Keys: {list(checkpoint.keys())}")
            logger.info("Loaded pre-trained Macro actor from Phase 1")
        else:
            logger.warning("No Macro checkpoint found; Phase 4 will use randomly initialized Macro")

    # Load Micro weights (prefer Phase 4 checkpoint, fallback to Phase 2)
    if resume_checkpoint_path is not None and os.path.exists(resume_checkpoint_path):
        micro_agent.load(resume_checkpoint_path)
        initial_episode_count = _extract_episode_from_micro_checkpoint_name(resume_checkpoint_path)
        initial_global_step = int(getattr(micro_agent, "total_steps", 0))
        logger.info(
            "Resuming Phase 4 from checkpoint: %s | episode=%d | total_steps=%d",
            resume_checkpoint_path,
            initial_episode_count,
            initial_global_step,
        )
    else:
        pretrain_micro_ckpt = os.path.join(root, "logs", "pretrain_micro", "best_pretrain_micro.pt")
        if os.path.exists(pretrain_micro_ckpt):
            checkpoint = torch.load(pretrain_micro_ckpt, map_location=device)
            if "model_actor" in checkpoint:
                micro_agent.actor.load_state_dict(checkpoint["model_actor"])
            elif "models" in checkpoint and "actor" in checkpoint["models"]:
                micro_agent.actor.load_state_dict(checkpoint["models"]["actor"])
            else:
                raise KeyError(f"Checkpoint missing actor weights. Keys: {list(checkpoint.keys())}")
            micro_agent.actor_target.load_state_dict(micro_agent.actor.state_dict())
            logger.info("Loaded pre-trained Micro actor from Phase 2")
        else:
            logger.warning("No pre-trained Micro checkpoint found; starting Micro from scratch")

    # Load data
    data_cfg = configs.get("data_config", {})
    dates = data_cfg.get("dates", {})
    train_start = dates.get("train_start", "2015-01-01")
    train_end = dates.get("train_end", "2022-12-31")
    val_start = dates.get("val_start", "2023-01-01")
    val_end = dates.get("val_end", "2023-12-31")

    loader = TrainingDataLoader(config_path=data_config_path)

    logger.info("Loading Micro training data...")
    micro_train = loader.load_micro_training_data(train_start, train_end, max_stocks=micro_agent.max_stocks)
    macro_train = loader.load_macro_training_data(train_start, train_end)

    logger.info("Loading Micro validation data...")
    micro_val = loader.load_micro_training_data(val_start, val_end, max_stocks=micro_agent.max_stocks)
    macro_val = loader.load_macro_training_data(val_start, val_end)

    # Base micro envs
    train_micro_env = MicroEnv(
        stock_returns_data=micro_train["stock_returns"],
        stock_features_data=micro_train["stock_features"],
        stock_to_sector_idx=micro_train["stock_to_sector"],
        micro_config_path=micro_config,
        risk_config_path=risk_config,
        max_stocks=micro_agent.max_stocks,
    )
    val_micro_env = MicroEnv(
        stock_returns_data=micro_val["stock_returns"],
        stock_features_data=micro_val["stock_features"],
        stock_to_sector_idx=micro_val["stock_to_sector"],
        micro_config_path=micro_config,
        risk_config_path=risk_config,
        max_stocks=micro_agent.max_stocks,
    )

    # Wrapped envs that provide macro context for goal generation
    train_env = GoalConditionedMicroEnv(
        train_micro_env,
        macro_states=macro_train["macro_states"],
        sector_embeddings=macro_train["sector_embeddings"],
        seed=seed,
    )
    val_env = GoalConditionedMicroEnv(
        val_micro_env,
        macro_states=macro_val["macro_states"],
        sector_embeddings=macro_val["sector_embeddings"],
        seed=seed + 1,
    )

    logger.info("Starting RL training (Phase 4)...")
    log_dir = phase4_log_dir
    result = _train_micro_frozen_macro_impl(
        micro_agent=micro_agent,
        macro_agent=macro_agent,
        env=train_env,
        val_env=val_env,
        config_path=micro_config,
        log_dir=log_dir,
        seed=seed,
        initial_episode_count=initial_episode_count,
        initial_global_step=initial_global_step,
    )
    logger.info("Phase 4 complete: %d episodes trained", result["episodes_trained"])
    return result


def _train_micro_frozen_macro_impl(
    micro_agent: MicroAgent,
    macro_agent: MacroAgent,
    env,
    val_env=None,
    config_path: str = "config/micro_agent_config.yaml",
    log_dir: str = "logs/phase4_micro",
    seed: int = 42,
    initial_episode_count: int = 0,
    initial_global_step: int = 0,
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

    with open(config_path, "r", encoding="utf-8") as f:cfg = yaml.safe_load(f)

    train_cfg = cfg.get("training", {})
    total_timesteps: int = int(train_cfg.get("total_timesteps", 1_000_000))
    warmup_steps: int = int(train_cfg.get("warmup_steps", train_cfg.get("learning_starts", 10_000)))
    update_every: int = int(train_cfg.get("update_every", 1))

    # Support both episode-based and step-based interval keys.
    eval_interval_episodes: int = int(train_cfg.get("eval_interval_episodes", 50))
    save_interval_episodes: int = int(train_cfg.get("save_interval_episodes", 100))
    eval_interval_steps = train_cfg.get("eval_interval", None)
    checkpoint_interval_steps = train_cfg.get("checkpoint_interval", None)

    eval_interval_steps = int(eval_interval_steps) if eval_interval_steps is not None else None
    checkpoint_interval_steps = int(checkpoint_interval_steps) if checkpoint_interval_steps is not None else None
    next_eval_step = None
    if eval_interval_steps and eval_interval_steps > 0:
        next_eval_step = ((initial_global_step // eval_interval_steps) + 1) * eval_interval_steps
    next_save_step = None
    if checkpoint_interval_steps and checkpoint_interval_steps > 0:
        next_save_step = ((initial_global_step // checkpoint_interval_steps) + 1) * checkpoint_interval_steps

    # Freeze Macro
    macro_agent.freeze()

    tracker = MetricsTracker(log_dir=log_dir)
    early_stop = EarlyStopping(patience=30, mode="max")

    best_eval_return = float("-inf")
    best_path = os.path.join(log_dir, "best_micro.pt")
    episode_count = initial_episode_count
    global_step = initial_global_step

    logger.info(
        "Phase 4: Micro RL training (Macro frozen) | total_steps=%d | warmup=%d | "
        "eval_ep=%d | eval_steps=%s | save_ep=%d | save_steps=%s | "
        "resume_ep=%d | resume_step=%d",
        total_timesteps,
        warmup_steps,
        eval_interval_episodes,
        str(eval_interval_steps),
        save_interval_episodes,
        str(checkpoint_interval_steps),
        initial_episode_count,
        initial_global_step,
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
        should_eval = False
        if val_env is not None:
            if eval_interval_episodes > 0 and episode_count % eval_interval_episodes == 0:
                should_eval = True
            if next_eval_step is not None and global_step >= next_eval_step:
                should_eval = True
                while next_eval_step is not None and global_step >= next_eval_step:
                    next_eval_step += eval_interval_steps

        if val_env is not None and should_eval:
            eval_metrics = _evaluate_micro(micro_agent, macro_agent, val_env)
            tracker.update(eval_metrics, step=global_step)

            if eval_metrics["eval_return_mean"] > best_eval_return:
                best_eval_return = eval_metrics["eval_return_mean"]
                micro_agent.save(best_path)

            if early_stop.step(eval_metrics.get("eval_sharpe_mean", 0.0)):
                logger.info("Early stopping at episode %d", episode_count)
                break

        # Periodic save
        should_save = False
        if save_interval_episodes > 0 and episode_count % save_interval_episodes == 0:
            should_save = True
        if next_save_step is not None and global_step >= next_save_step:
            should_save = True
            while next_save_step is not None and global_step >= next_save_step:
                next_save_step += checkpoint_interval_steps

        if should_save:
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
    if "goal" in info and info["goal"] is not None:
        return np.array(info["goal"], dtype=np.float32)
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
