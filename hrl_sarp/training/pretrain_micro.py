"""
File: pretrain_micro.py
Module: training
Description: Phase 2 — Supervised pre-training of the Micro agent on historical
    stock returns with mock goals. Behavioural cloning on oracle allocations derived
    from momentum or equal-weight strategies. This provides the Micro agent with
    a reasonable starting policy before TD3 RL training begins.
Design Decisions: Uses MSE loss between actor output and oracle allocation (returns-
    ranked softmax). Goal embeddings come from pre-trained Macro or random sampling
    during pre-training. Separate pre-training avoids catastrophic interference with
    TD3 critic.
References: Behavioural Cloning for portfolio allocation (Jiang 2017)
Author: HRL-SARP Framework
"""

import logging
import os
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml

from agents.micro_agent import MicroAgent
from training.trainer_utils import (
    EarlyStopping,
    MetricsTracker,
    save_checkpoint,
    set_global_seed,
)

logger = logging.getLogger(__name__)


def pretrain_micro(
    configs: Dict[str, Any],
    device: torch.device,
    seed: int = 42,
) -> Dict[str, Any]:
    """Phase 2: Supervised pre-training of the Micro agent.
    
    This is a wrapper that initializes the micro agent and calls the actual
    pre-training function with prepared data.

    Args:
        configs: Dictionary containing all config files (macro, micro, data, risk).
        device: Torch device (cpu or cuda).
        seed: Random seed.

    Returns:
        Dict with training metrics and best checkpoint path.
    """
    set_global_seed(seed)
    
    logger.info("Initializing Micro agent for pre-training...")
    
    # Import data loader
    from data.training_data_loader import TrainingDataLoader
    
    # Initialize MicroAgent
    micro_config = configs.get("micro_agent_config", {})
    max_stocks = micro_config.get("environment", {}).get("max_stocks", 50)
    
    micro_agent = MicroAgent(
        config_path="config/micro_agent_config.yaml",
        device=device,
    )
    logger.info("✓ MicroAgent initialized")
    
    # Load training data
    data_config = configs.get("data_config", {})
    dates = data_config.get("dates", {})
    train_start = dates.get("train_start", "2015-01-01")
    train_end = dates.get("train_end", "2022-12-31")
    val_start = dates.get("val_start", "2023-01-01")
    val_end = dates.get("val_end", "2023-12-31")
    
    loader = TrainingDataLoader(config_path="config/data_config.yaml")
    
    logger.info("Loading training data: %s to %s", train_start, train_end)
    train_data_raw = loader.load_micro_training_data(train_start, train_end, max_stocks)
    
    # Prepare data with random goals for pre-training
    N = len(train_data_raw["stock_returns"])
    random_goals = np.random.randn(N, 14).astype(np.float32)
    random_goals[:, :11] = np.abs(random_goals[:, :11])  # Sector weights positive
    random_goals[:, :11] /= random_goals[:, :11].sum(axis=1, keepdims=True) + 1e-8
    
    train_data = {
        "stock_features": train_data_raw["stock_features"],
        "stock_returns": train_data_raw["stock_returns"],
        "goals": random_goals,
        "masks": train_data_raw["stock_masks"],
    }
    logger.info("✓ Training data loaded: %d samples", N)
    
    logger.info("Loading validation data: %s to %s", val_start, val_end)
    val_data_raw = loader.load_micro_training_data(val_start, val_end, max_stocks)
    
    N_val = len(val_data_raw["stock_returns"])
    random_goals_val = np.random.randn(N_val, 14).astype(np.float32)
    random_goals_val[:, :11] = np.abs(random_goals_val[:, :11])
    random_goals_val[:, :11] /= random_goals_val[:, :11].sum(axis=1, keepdims=True) + 1e-8
    
    val_data = {
        "stock_features": val_data_raw["stock_features"],
        "stock_returns": val_data_raw["stock_returns"],
        "goals": random_goals_val,
        "masks": val_data_raw["stock_masks"],
    }
    logger.info("✓ Validation data loaded: %d samples", N_val)
    
    # Call actual pre-training implementation
    logger.info("Starting supervised pre-training...")
    result = _pretrain_micro_impl(
        micro_agent=micro_agent,
        train_data=train_data,
        val_data=val_data,
        config_path="config/micro_agent_config.yaml",
        log_dir="logs/pretrain_micro",
        seed=seed,
    )
    
    logger.info("✓ Micro pre-training complete: best_loss=%.6f", result["best_loss"])
    return result


def _pretrain_micro_impl(
    micro_agent: MicroAgent,
    train_data: Dict[str, np.ndarray],
    val_data: Optional[Dict[str, np.ndarray]] = None,
    config_path: str = "config/micro_agent_config.yaml",
    log_dir: str = "logs/pretrain_micro",
    seed: int = 42,
) -> Dict[str, Any]:
    """Phase 2: Supervised pre-training of the Micro agent.

    Expected data format:
        train_data = {
            'stock_features': (N, max_stocks, 22),  # Per-stock features
            'stock_returns': (N, max_stocks),        # One-step-ahead returns
            'goals': (N, 14),                        # Macro goal embeddings
            'masks': (N, max_stocks),                # Stock validity masks
        }

    Args:
        micro_agent: MicroAgent instance to pre-train.
        train_data: Training dataset as dict of numpy arrays.
        val_data: Optional validation set (same format).
        config_path: Path to micro agent config.
        log_dir: Directory for logs and checkpoints.
        seed: Random seed.

    Returns:
        Dict with training metrics and best checkpoint path.
    """
    set_global_seed(seed)
    os.makedirs(log_dir, exist_ok=True)

    with open(config_path, "r", encoding="utf-8") as f:cfg = yaml.safe_load(f)

    train_cfg = cfg.get("training", {})
    n_epochs: int = train_cfg.get("pretrain_epochs", 30)
    batch_size: int = train_cfg.get("pretrain_batch_size", 64)
    lr: float = train_cfg.get("pretrain_lr", 1e-3)

    device = micro_agent.device
    actor = micro_agent.actor
    actor.train()

    pretrain_optimizer = optim.Adam(actor.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(pretrain_optimizer, T_max=n_epochs)

    early_stop = EarlyStopping(patience=10, mode="min")
    tracker = MetricsTracker(log_dir=log_dir)

    # Convert data to tensors
    feats_t = torch.tensor(train_data["stock_features"], dtype=torch.float32, device=device)
    returns_t = torch.tensor(train_data["stock_returns"], dtype=torch.float32, device=device)
    goals_t = torch.tensor(train_data["goals"], dtype=torch.float32, device=device)
    masks_t = torch.tensor(train_data["masks"], dtype=torch.float32, device=device)

    n_samples = len(feats_t)
    best_loss = float("inf")
    best_path = os.path.join(log_dir, "best_pretrain_micro.pt")

    logger.info(
        "Phase 2: Micro pre-training | samples=%d | epochs=%d | batch=%d",
        n_samples, n_epochs, batch_size,
    )

    for epoch in range(n_epochs):
        perm = torch.randperm(n_samples, device=device)
        epoch_loss = 0.0
        epoch_portfolio_return = 0.0
        n_batches = 0

        for start in range(0, n_samples - batch_size, batch_size):
            idx = perm[start:start + batch_size]

            # Forward pass: get predicted portfolio weights
            pred_weights = actor(feats_t[idx], goals_t[idx], masks_t[idx])

            # ── Oracle allocation ──
            # Use next-step returns to compute "ideal" allocation
            batch_returns = returns_t[idx]
            # Mask invalid stocks
            masked_returns = batch_returns * masks_t[idx]
            # Softmax over returns → overweight winners
            oracle_weights = F.softmax(masked_returns * 20.0, dim=-1)
            # Zero out padded stocks
            oracle_weights = oracle_weights * masks_t[idx]
            oracle_sums = oracle_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            oracle_weights = oracle_weights / oracle_sums

            # ── Loss: MSE between predicted and oracle weights ──
            loss = F.mse_loss(pred_weights * masks_t[idx], oracle_weights)

            # ── Portfolio return as monitoring metric ──
            portfolio_return = (pred_weights * batch_returns).sum(dim=-1).mean()

            pretrain_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            pretrain_optimizer.step()

            epoch_loss += loss.item()
            epoch_portfolio_return += portfolio_return.item()
            n_batches += 1

        scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_return = epoch_portfolio_return / max(n_batches, 1)

        metrics = {
            "pretrain_micro/loss": avg_loss,
            "pretrain_micro/portfolio_return": avg_return,
            "pretrain_micro/lr": pretrain_optimizer.param_groups[0]["lr"],
        }

        # Validation
        if val_data is not None:
            val_metrics = _validate_micro(actor, val_data, device)
            metrics.update(val_metrics)
            monitor_loss = val_metrics["pretrain_micro/val_loss"]
        else:
            monitor_loss = avg_loss

        tracker.update(metrics, step=epoch)

        if monitor_loss < best_loss:
            best_loss = monitor_loss
            save_checkpoint(
                best_path,
                epoch=epoch,
                models={"actor": actor},
                optimizers={"pretrain_opt": pretrain_optimizer},
                metrics=metrics,
            )

        if (epoch + 1) % 5 == 0:
            logger.info(
                "Epoch %d/%d | loss=%.6f | port_return=%.4f",
                epoch + 1, n_epochs, avg_loss, avg_return,
            )

        if early_stop.step(monitor_loss):
            logger.info("Early stopping at epoch %d", epoch + 1)
            break

    tracker.save("pretrain_micro_metrics.json")

    return {
        "best_loss": best_loss,
        "best_checkpoint": best_path,
        "epochs_trained": epoch + 1,
        "metrics_summary": tracker.summary(),
    }


@torch.no_grad()
def _validate_micro(
    actor: nn.Module,
    val_data: Dict[str, np.ndarray],
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate pre-trained Micro actor on validation data."""
    actor.eval()
    feats = torch.tensor(val_data["stock_features"], dtype=torch.float32, device=device)
    returns = torch.tensor(val_data["stock_returns"], dtype=torch.float32, device=device)
    goals = torch.tensor(val_data["goals"], dtype=torch.float32, device=device)
    masks = torch.tensor(val_data["masks"], dtype=torch.float32, device=device)

    pred_weights = actor(feats, goals, masks)

    # Oracle
    masked_returns = returns * masks
    oracle_weights = F.softmax(masked_returns * 20.0, dim=-1) * masks
    oracle_sums = oracle_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    oracle_weights = oracle_weights / oracle_sums

    loss = F.mse_loss(pred_weights * masks, oracle_weights).item()
    port_return = float((pred_weights * returns).sum(dim=-1).mean().item())

    actor.train()
    return {
        "pretrain_micro/val_loss": loss,
        "pretrain_micro/val_portfolio_return": port_return,
    }
