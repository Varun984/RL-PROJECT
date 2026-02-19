"""
File: train_macro_frozen_micro.py
Module: training
Description: Phase 3 - RL training of the Macro agent (PPO) with the Micro agent's
    weights frozen. Supports resuming from latest phase3 checkpoints.
"""

import logging
import os
import re
from typing import Any, Dict, Optional

import torch
import yaml

from agents.macro_agent import MacroAgent
from agents.micro_agent import MicroAgent
from training.curriculum_manager import CurriculumManager
from training.trainer_utils import EarlyStopping, MetricsTracker, evaluate_agent, set_global_seed

logger = logging.getLogger(__name__)


def _get_project_root() -> str:
    """Get hrl_sarp project root directory for config paths."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _extract_episode_from_checkpoint_name(path: str) -> int:
    """Extract episode number from checkpoint name like macro_ep600.pt."""
    m = re.search(r"macro_ep(\d+)\.pt$", os.path.basename(path))
    return int(m.group(1)) if m else 0


def _find_latest_phase3_checkpoint(log_dir: str) -> Optional[str]:
    """Return newest phase3 checkpoint by episode number, if any."""
    if not os.path.isdir(log_dir):
        return None
    best_path = None
    best_ep = -1
    for name in os.listdir(log_dir):
        m = re.match(r"macro_ep(\d+)\.pt$", name)
        if not m:
            continue
        ep = int(m.group(1))
        if ep > best_ep:
            best_ep = ep
            best_path = os.path.join(log_dir, name)
    return best_path


def _infer_last_logged_stage(root: str) -> Optional[str]:
    """Infer latest curriculum stage from recent train log lines."""
    logs_dir = os.path.join(root, "logs")
    if not os.path.isdir(logs_dir):
        return None

    log_files = [
        os.path.join(logs_dir, f)
        for f in os.listdir(logs_dir)
        if f.startswith("train_") and f.endswith(".log")
    ]
    log_files.sort(key=lambda p: os.path.getmtime(p), reverse=True)

    stage_pattern = re.compile(r"Episode\s+\d+\s+\|.*\|\s+stage=([a-zA-Z_]+)")
    for path in log_files[:5]:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            for line in reversed(lines):
                m = stage_pattern.search(line)
                if m:
                    return m.group(1)
        except OSError:
            continue
    return None


def train_macro_frozen_micro(
    configs: Dict[str, Any],
    device: torch.device,
    seed: int = 42,
) -> Dict[str, Any]:
    """Phase 3: RL training of Macro agent with frozen Micro."""
    set_global_seed(seed)
    root = _get_project_root()

    logger.info("Initializing agents and environment for Phase 3...")

    from data.training_data_loader import TrainingDataLoader
    from environment.macro_env import MacroEnv

    # Config paths (relative to hrl_sarp root)
    macro_config = os.path.join(root, "config", "macro_agent_config.yaml")
    micro_config = os.path.join(root, "config", "micro_agent_config.yaml")
    data_config_path = os.path.join(root, "config", "data_config.yaml")
    risk_config = os.path.join(root, "config", "risk_config.yaml")
    phase3_log_dir = os.path.join(root, "logs", "phase3_macro")

    # Initialize Macro
    logger.info("Initializing MacroAgent...")
    macro_agent = MacroAgent(
        config_path=macro_config,
        device=device,
    )

    # Resume settings from macro config (loaded by main.py)
    macro_cfg = configs.get("macro_agent_config", {})
    train_cfg = macro_cfg.get("training", {})
    resume_phase3 = bool(train_cfg.get("resume_phase3", True))
    resume_checkpoint_cfg = train_cfg.get("resume_checkpoint", None)
    resume_stage_name: Optional[str] = None
    initial_episode_count = 0
    initial_global_step = 0

    # Optional resume from latest Phase 3 checkpoint
    resume_checkpoint_path: Optional[str] = None
    if resume_phase3:
        if isinstance(resume_checkpoint_cfg, str) and resume_checkpoint_cfg.strip():
            candidate = resume_checkpoint_cfg.strip()
            if not os.path.isabs(candidate):
                candidate = os.path.join(root, candidate)
            if os.path.exists(candidate):
                resume_checkpoint_path = candidate
            else:
                logger.warning("Configured resume_checkpoint not found: %s", candidate)
        if resume_checkpoint_path is None:
            resume_checkpoint_path = _find_latest_phase3_checkpoint(phase3_log_dir)

    if resume_checkpoint_path is not None and os.path.exists(resume_checkpoint_path):
        macro_agent.load(resume_checkpoint_path)
        initial_episode_count = _extract_episode_from_checkpoint_name(resume_checkpoint_path)
        initial_global_step = int(getattr(macro_agent, "total_steps", 0))
        resume_stage_name = _infer_last_logged_stage(root)
        logger.info(
            "Resuming Phase 3 from checkpoint: %s | episode=%d | total_steps=%d | stage=%s",
            resume_checkpoint_path,
            initial_episode_count,
            initial_global_step,
            resume_stage_name or "unknown",
        )
    else:
        # Fallback: load pre-trained Macro weights from Phase 1
        macro_checkpoint_path = os.path.join(root, "logs", "pretrain_macro", "best_pretrain_macro.pt")
        if os.path.exists(macro_checkpoint_path):
            checkpoint = torch.load(macro_checkpoint_path, map_location=device)
            if "model_actor" in checkpoint:
                macro_agent.actor.load_state_dict(checkpoint["model_actor"])
            elif "models" in checkpoint and "actor" in checkpoint["models"]:
                macro_agent.actor.load_state_dict(checkpoint["models"]["actor"])
            else:
                raise KeyError(f"Checkpoint missing actor weights. Keys: {list(checkpoint.keys())}")
            logger.info("Loaded pre-trained Macro weights from Phase 1")
        else:
            logger.warning("No pre-trained Macro weights found, starting from scratch")

    # Initialize Micro
    logger.info("Initializing MicroAgent...")
    micro_agent = MicroAgent(
        config_path=micro_config,
        device=device,
    )

    # Load pre-trained Micro weights from Phase 2
    micro_checkpoint_path = os.path.join(root, "logs", "pretrain_micro", "best_pretrain_micro.pt")
    if os.path.exists(micro_checkpoint_path):
        checkpoint = torch.load(micro_checkpoint_path, map_location=device)
        if "model_actor" in checkpoint:
            micro_agent.actor.load_state_dict(checkpoint["model_actor"])
        elif "models" in checkpoint and "actor" in checkpoint["models"]:
            micro_agent.actor.load_state_dict(checkpoint["models"]["actor"])
        else:
            raise KeyError(f"Checkpoint missing actor weights. Keys: {list(checkpoint.keys())}")
        logger.info("Loaded pre-trained Micro weights from Phase 2")
    else:
        logger.warning("No pre-trained Micro weights found")

    # Load environment data
    data_cfg = configs.get("data_config", {})
    dates = data_cfg.get("dates", {})
    train_start = dates.get("train_start", "2015-01-01")
    train_end = dates.get("train_end", "2022-12-31")
    val_start = dates.get("val_start", "2023-01-01")
    val_end = dates.get("val_end", "2023-12-31")

    loader = TrainingDataLoader(config_path=data_config_path)

    logger.info("Loading training environment data...")
    train_data = loader.load_macro_training_data(train_start, train_end)

    train_env = MacroEnv(
        sector_returns_data=train_data["sector_returns"],
        benchmark_returns_data=train_data["sector_returns"].mean(axis=1),
        macro_states_data=train_data["macro_states"],
        sector_gnn_embeddings_data=train_data["sector_embeddings"],
        regime_labels=train_data["regime_labels"],
        macro_config_path=macro_config,
        risk_config_path=risk_config,
    )
    logger.info("Training environment created")

    logger.info("Loading validation environment data...")
    val_data = loader.load_macro_training_data(val_start, val_end)
    val_env = MacroEnv(
        sector_returns_data=val_data["sector_returns"],
        benchmark_returns_data=val_data["sector_returns"].mean(axis=1),
        macro_states_data=val_data["macro_states"],
        sector_gnn_embeddings_data=val_data["sector_embeddings"],
        regime_labels=val_data["regime_labels"],
        macro_config_path=macro_config,
        risk_config_path=risk_config,
    )
    logger.info("Validation environment created")

    logger.info("Starting RL training (Phase 3)...")
    result = _train_macro_frozen_micro_impl(
        macro_agent=macro_agent,
        micro_agent=micro_agent,
        env=train_env,
        val_env=val_env,
        config_path=macro_config,
        log_dir=phase3_log_dir,
        seed=seed,
        initial_episode_count=initial_episode_count,
        initial_global_step=initial_global_step,
        resume_stage_name=resume_stage_name,
    )

    logger.info("Phase 3 complete: %d episodes trained", result["episodes_trained"])
    return result


def _train_macro_frozen_micro_impl(
    macro_agent: MacroAgent,
    micro_agent: MicroAgent,
    env,
    val_env=None,
    config_path: str = "config/macro_agent_config.yaml",
    log_dir: str = "logs/phase3_macro",
    seed: int = 42,
    initial_episode_count: int = 0,
    initial_global_step: int = 0,
    resume_stage_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Phase 3: RL training of Macro agent with frozen Micro."""
    set_global_seed(seed)
    os.makedirs(log_dir, exist_ok=True)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg.get("training", {})
    total_timesteps: int = train_cfg.get("total_timesteps", 500_000)
    eval_interval: int = train_cfg.get("eval_interval_episodes", 50)
    save_interval: int = train_cfg.get("save_interval_episodes", 100)
    early_stopping_patience: int = train_cfg.get("early_stopping_patience", 30)
    num_eval_episodes: int = train_cfg.get("num_eval_episodes", 5)
    early_stopping_metric: str = train_cfg.get("early_stopping_metric", "eval_sharpe_mean")

    micro_agent.freeze()

    tracker = MetricsTracker(log_dir=log_dir)
    early_stop_mode = "min" if "drawdown" in early_stopping_metric.lower() else "max"
    early_stop = EarlyStopping(
        patience=early_stopping_patience,
        mode=early_stop_mode,
    )
    curriculum = CurriculumManager(config_path=config_path)
    if initial_episode_count > 0:
        curriculum.total_episodes = initial_episode_count
    if resume_stage_name:
        stage_map = {s.name: idx for idx, s in enumerate(curriculum.stages)}
        if resume_stage_name in stage_map:
            curriculum.set_stage(stage_map[resume_stage_name])

    best_eval_return = float("-inf")
    best_path = os.path.join(log_dir, "best_macro.pt")
    episode_count = initial_episode_count
    global_step = initial_global_step

    logger.info(
        (
            "Phase 3: Macro RL training (Micro frozen) | total_steps=%d | "
            "eval_every=%d ep | eval_n=%d | early_stop=%s (%s, patience=%d) | "
            "resume_ep=%d | resume_step=%d | stage=%s"
        ),
        total_timesteps,
        eval_interval,
        num_eval_episodes,
        early_stopping_metric,
        early_stop_mode,
        early_stopping_patience,
        initial_episode_count,
        initial_global_step,
        curriculum.stage_name,
    )

    while global_step < total_timesteps:
        obs, info = env.reset()
        done = False
        episode_return = 0.0
        episode_steps = 0

        while not done and global_step < total_timesteps:
            macro_state = obs[:macro_agent.macro_state_dim]
            sector_emb = obs[macro_agent.macro_state_dim:].reshape(
                macro_agent.num_sectors, macro_agent.sector_emb_dim
            )

            action, log_prob, value = macro_agent.select_action(macro_state, sector_emb)
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

            obs = next_obs
            episode_return += reward
            episode_steps += 1
            global_step += 1

            if macro_agent.buffer.full:
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

        episode_count += 1
        episode_sharpe = info.get("sharpe", 0.0)

        ep_metrics = {
            "episode/return": episode_return,
            "episode/length": episode_steps,
            "episode/sharpe": episode_sharpe,
            "episode/drawdown": info.get("drawdown", 0.0),
        }
        tracker.update(ep_metrics, step=global_step)

        curriculum.on_episode_end(episode_return, episode_sharpe)

        if val_env is not None and episode_count % eval_interval == 0:
            eval_metrics = evaluate_agent(
                val_env,
                macro_agent,
                n_episodes=num_eval_episodes,
            )
            tracker.update(eval_metrics, step=global_step)

            if eval_metrics["eval_return_mean"] > best_eval_return:
                best_eval_return = eval_metrics["eval_return_mean"]
                macro_agent.save(best_path)

            monitor_value = float(eval_metrics.get(early_stopping_metric, 0.0))
            if early_stop.step(monitor_value):
                logger.info("Early stopping at episode %d", episode_count)
                break

        if episode_count % save_interval == 0:
            ckpt_path = os.path.join(log_dir, f"macro_ep{episode_count}.pt")
            macro_agent.save(ckpt_path)

        if episode_count % 20 == 0:
            logger.info(
                "Episode %d | return=%.4f | sharpe=%.4f | steps=%d | stage=%s",
                episode_count,
                episode_return,
                episode_sharpe,
                global_step,
                curriculum.stage_name,
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

