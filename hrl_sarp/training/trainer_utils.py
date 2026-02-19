"""
File: trainer_utils.py
Module: training
Description: Shared utilities for all training phases — seed setup, gradient clipping,
    LR scheduling, early stopping, checkpoint management, and metrics tracking.
    Integrates with MLflow for experiment logging.
Design Decisions: Centralised utilities ensure consistent behaviour across all 5 phases.
    EarlyStopping uses patience-based monitoring on a configurable metric.
References: MLflow tracking API, PyTorch best practices
Author: HRL-SARP Framework
"""

import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# SEED & REPRODUCIBILITY
# ══════════════════════════════════════════════════════════════════════


def set_global_seed(seed: int = 42) -> None:
    """Set seeds for Python, NumPy, PyTorch, and CUDA for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info("Global seed set to %d", seed)


def setup_training(
    seed: int = 42,
    log_dir: str = "logs",
    experiment_name: str = "hrl_sarp",
    use_mlflow: bool = True,
) -> Dict[str, Any]:
    """One-call training setup: seed, logging, directories, MLflow.

    Returns:
        Dict with 'device', 'log_dir', 'mlflow_run_id' keys.
    """
    set_global_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Create directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "metrics"), exist_ok=True)

    result = {"device": str(device), "log_dir": log_dir, "mlflow_run_id": None}

    # MLflow setup
    if use_mlflow:
        try:
            import mlflow
            mlflow.set_experiment(experiment_name)
            run = mlflow.start_run()
            result["mlflow_run_id"] = run.info.run_id
            mlflow.log_param("seed", seed)
            mlflow.log_param("device", str(device))
            logger.info("MLflow run started: %s", run.info.run_id)
        except ImportError:
            logger.warning("MLflow not installed; skipping experiment tracking")
        except Exception as e:
            logger.warning("MLflow setup failed: %s", e)

    return result


# ══════════════════════════════════════════════════════════════════════
# CHECKPOINT MANAGEMENT
# ══════════════════════════════════════════════════════════════════════


def save_checkpoint(
    path: str,
    epoch: int,
    models: Dict[str, nn.Module],
    optimizers: Dict[str, torch.optim.Optimizer],
    metrics: Optional[Dict[str, float]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a training checkpoint with all model and optimizer states."""
    checkpoint = {
        "epoch": epoch,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    for name, model in models.items():
        checkpoint[f"model_{name}"] = model.state_dict()
    for name, opt in optimizers.items():
        checkpoint[f"optimizer_{name}"] = opt.state_dict()
    if metrics:
        checkpoint["metrics"] = metrics
    if extra:
        checkpoint["extra"] = extra

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)
    logger.info("Checkpoint saved to %s (epoch %d)", path, epoch)


def load_checkpoint(
    path: str,
    models: Dict[str, nn.Module],
    optimizers: Optional[Dict[str, torch.optim.Optimizer]] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Load a training checkpoint, restoring model and optimizer states.

    Returns:
        Dict with 'epoch', 'metrics', and any 'extra' data.
    """
    checkpoint = torch.load(path, map_location=device)

    for name, model in models.items():
        key = f"model_{name}"
        if key in checkpoint:
            model.load_state_dict(checkpoint[key])
            logger.info("Restored model '%s'", name)

    if optimizers:
        for name, opt in optimizers.items():
            key = f"optimizer_{name}"
            if key in checkpoint:
                opt.load_state_dict(checkpoint[key])

    result = {
        "epoch": checkpoint.get("epoch", 0),
        "metrics": checkpoint.get("metrics", {}),
        "extra": checkpoint.get("extra", {}),
    }
    logger.info("Checkpoint loaded from %s (epoch %d)", path, result["epoch"])
    return result


# ══════════════════════════════════════════════════════════════════════
# EARLY STOPPING
# ══════════════════════════════════════════════════════════════════════


class EarlyStopping:
    """Patience-based early stopping on a monitored metric."""

    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 1e-4,
        mode: str = "max",
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_value: Optional[float] = None
        self.counter: int = 0
        self.should_stop: bool = False

    def step(self, value: float) -> bool:
        """Update with new metric value. Returns True if should stop."""
        if self.best_value is None:
            self.best_value = value
            return False

        if self.mode == "max":
            improved = value > self.best_value + self.min_delta
        else:
            improved = value < self.best_value - self.min_delta

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    "Early stopping triggered after %d checks without improvement",
                    self.patience,
                )

        return self.should_stop

    def reset(self) -> None:
        self.best_value = None
        self.counter = 0
        self.should_stop = False


# ══════════════════════════════════════════════════════════════════════
# METRICS TRACKER
# ══════════════════════════════════════════════════════════════════════


class MetricsTracker:
    """Track and log training metrics across epochs."""

    def __init__(self, log_dir: str = "logs/metrics") -> None:
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.history: Dict[str, List[float]] = {}

    def update(self, metrics: Dict[str, float], step: int) -> None:
        """Record metrics for a given step."""
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            # Convert numpy types to native Python for JSON serialization
            self.history[key].append(float(value) if hasattr(value, "item") else value)

        # Log to MLflow if available
        try:
            import mlflow
            if mlflow.active_run():
                mlflow.log_metrics(metrics, step=step)
        except (ImportError, Exception):
            pass

    def get(self, key: str) -> List[float]:
        return self.history.get(key, [])

    def get_best(self, key: str, mode: str = "max") -> Tuple[float, int]:
        """Return best value and its index."""
        values = self.get(key)
        if not values:
            return 0.0, 0
        if mode == "max":
            idx = int(np.argmax(values))
        else:
            idx = int(np.argmin(values))
        return values[idx], idx

    def save(self, filename: str = "metrics.json") -> None:
        path = os.path.join(self.log_dir, filename)
        # Convert any numpy types to native Python for JSON serialization
        serializable = self._to_serializable(self.history)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)
        logger.info("Metrics saved to %s", path)

    @staticmethod
    def _to_serializable(obj: Any) -> Any:
        """Recursively convert numpy types to native Python for JSON."""
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: MetricsTracker._to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [MetricsTracker._to_serializable(v) for v in obj]
        return obj

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Return summary statistics for each metric."""
        summary = {}
        for key, values in self.history.items():
            arr = np.array(values)
            summary[key] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "last": float(arr[-1]),
            }
        return summary


# ══════════════════════════════════════════════════════════════════════
# GRADIENT UTILITIES
# ══════════════════════════════════════════════════════════════════════


def clip_gradients(model: nn.Module, max_norm: float) -> float:
    """Clip gradients and return the total gradient norm before clipping."""
    total_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    return float(total_norm)


def compute_grad_norm(model: nn.Module) -> float:
    """Compute total L2 gradient norm for monitoring."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


# ══════════════════════════════════════════════════════════════════════
# EVALUATION HELPER
# ══════════════════════════════════════════════════════════════════════


def evaluate_agent(
    env,
    agent,
    n_episodes: int = 10,
    deterministic: bool = True,
) -> Dict[str, float]:
    """Run evaluation episodes and return aggregate metrics."""
    returns = []
    sharpes = []
    drawdowns = []

    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_return = 0.0

        while not done:
            if hasattr(agent, "select_action"):
                # Need to handle macro vs micro differently
                action = _get_agent_action(agent, obs, deterministic)
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            done = terminated or truncated

        returns.append(ep_return)
        sharpes.append(info.get("sharpe", 0.0))
        drawdowns.append(info.get("drawdown", 0.0))

    return {
        "eval_return_mean": float(np.mean(returns)),
        "eval_return_std": float(np.std(returns)),
        "eval_sharpe_mean": float(np.mean(sharpes)),
        "eval_max_drawdown": float(np.max(drawdowns)),
    }


def _get_agent_action(agent, obs: np.ndarray, deterministic: bool) -> np.ndarray:
    """Extract action from agent, handling different agent interfaces."""
    if hasattr(agent, "macro_state_dim"):
        # MacroAgent: split obs into state + embeddings
        state = obs[:agent.macro_state_dim]
        emb = obs[agent.macro_state_dim:].reshape(
            agent.num_sectors, agent.sector_emb_dim
        )
        action, _, _ = agent.select_action(state, emb, deterministic)
    else:
        # MicroAgent: pass full obs + default goal
        n = agent.stock_feature_dim
        max_s = agent.max_stocks
        feats = obs[:max_s * n].reshape(max_s, n)
        goal = obs[max_s * n:max_s * n + agent.goal_input_dim]
        action = agent.select_action(feats, goal, deterministic=deterministic)
    return action


# ══════════════════════════════════════════════════════════════════════
# AGENT EVALUATION
# ══════════════════════════════════════════════════════════════════════


def evaluate_agent(
    env,
    agent,
    n_episodes: int = 5,
    deterministic: bool = True,
) -> Dict[str, float]:
    """Evaluate an agent on an environment for n episodes.
    
    Args:
        env: Gymnasium environment
        agent: Agent with select_action method
        n_episodes: Number of episodes to evaluate
        deterministic: Use deterministic policy
        
    Returns:
        Dict with evaluation metrics
    """
    episode_returns = []
    episode_lengths = []
    episode_sharpes = []
    
    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_return = 0.0
        episode_length = 0
        
        while not done:
            # Parse observation for macro agent
            if hasattr(agent, 'macro_state_dim'):
                macro_state = obs[:agent.macro_state_dim]
                sector_emb = obs[agent.macro_state_dim:].reshape(
                    agent.num_sectors, agent.sector_emb_dim
                )
                action, _, _ = agent.select_action(macro_state, sector_emb, deterministic=deterministic)
            else:
                # For micro agent or other agents
                action, _, _ = agent.select_action(obs, deterministic=deterministic)
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_return += reward
            episode_length += 1
        
        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
        episode_sharpes.append(info.get('sharpe', 0.0))
    
    return {
        "eval_return_mean": float(np.mean(episode_returns)),
        "eval_return_std": float(np.std(episode_returns)),
        "eval_length_mean": float(np.mean(episode_lengths)),
        "eval_sharpe_mean": float(np.mean(episode_sharpes)),
    }
