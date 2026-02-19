"""
File: pretrain_macro.py
Module: training
Description: Phase 1 — Supervised pre-training of the Macro agent on historical
    regime labels and sector returns. Trains the regime classification head with
    cross-entropy loss and the sector allocation head with a portfolio return
    maximisation objective. This initialises the Macro policy near reasonable
    allocations before RL fine-tuning.
Design Decisions: Two-head supervised loss (regime CE + allocation MSE against oracle
    momentum-based allocation). Uses teacher forcing with historical data.
References: Pre-training for RL (Levine 2020), Behavioural Cloning
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

from agents.macro_agent import MacroAgent
from training.trainer_utils import (
    EarlyStopping,
    MetricsTracker,
    save_checkpoint,
    set_global_seed,
)

logger = logging.getLogger(__name__)


def pretrain_macro(
    configs: Dict[str, Any],
    device: torch.device,
    seed: int = 42,
) -> Dict[str, Any]:
    """Phase 1: Supervised pre-training of the Macro agent.
    
    This is a wrapper that initializes the macro agent and calls the actual
    pre-training function with prepared data.

    Args:
        configs: Dictionary containing all config files (macro, micro, data, risk).
        device: Torch device (cpu or cuda).
        seed: Random seed.

    Returns:
        Dict with training metrics and best checkpoint path.
    """
    set_global_seed(seed)
    
    logger.info("Initializing Macro agent for pre-training...")
    
    # Import data loader
    from data.training_data_loader import TrainingDataLoader
    
    # Initialize MacroAgent
    macro_config = configs.get("macro_agent_config", {})
    macro_agent = MacroAgent(
        config_path="config/macro_agent_config.yaml",
        device=device,
    )
    logger.info("✓ MacroAgent initialized")
    
    # Load training data
    data_config = configs.get("data_config", {})
    dates = data_config.get("dates", {})
    train_start = dates.get("train_start", "2015-01-01")
    train_end = dates.get("train_end", "2022-12-31")
    val_start = dates.get("val_start", "2023-01-01")
    val_end = dates.get("val_end", "2023-12-31")
    
    loader = TrainingDataLoader(config_path="config/data_config.yaml")
    
    logger.info("Loading training data: %s to %s", train_start, train_end)
    train_data = loader.load_macro_training_data(train_start, train_end)
    logger.info("✓ Training data loaded: %d samples", len(train_data["macro_states"]))
    
    logger.info("Loading validation data: %s to %s", val_start, val_end)
    val_data = loader.load_macro_training_data(val_start, val_end)
    logger.info("✓ Validation data loaded: %d samples", len(val_data["macro_states"]))
    
    # Call actual pre-training implementation
    logger.info("Starting supervised pre-training...")
    result = _pretrain_macro_impl(
        macro_agent=macro_agent,
        train_data=train_data,
        val_data=val_data,
        config_path="config/macro_agent_config.yaml",
        log_dir="logs/pretrain_macro",
        seed=seed,
    )
    
    logger.info("✓ Macro pre-training complete: best_loss=%.4f", result["best_loss"])
    return result


def _pretrain_macro_impl(
    macro_agent: MacroAgent,
    train_data: Dict[str, np.ndarray],
    val_data: Optional[Dict[str, np.ndarray]] = None,
    config_path: str = "config/macro_agent_config.yaml",
    log_dir: str = "logs/pretrain_macro",
    seed: int = 42,
) -> Dict[str, Any]:
    """Phase 1: Supervised pre-training of the Macro agent.

    Expected data format:
        train_data = {
            'macro_states': (N, 18),           # Macro feature vectors
            'sector_embeddings': (N, 11, 64),   # GNN sector embeddings
            'sector_returns': (N, 11),           # One-step-ahead sector returns
            'regime_labels': (N,),               # Expert regime labels (0/1/2)
        }

    Args:
        macro_agent: MacroAgent instance to pre-train.
        train_data: Training dataset as dict of numpy arrays.
        val_data: Optional validation dataset (same format).
        config_path: Path to macro agent config.
        log_dir: Directory for logs and checkpoints.
        seed: Random seed.

    Returns:
        Dict with training metrics and best checkpoint path.
    """
    set_global_seed(seed)
    os.makedirs(log_dir, exist_ok=True)

    with open(config_path, "r", encoding="utf-8") as f:cfg = yaml.safe_load(f)

    train_cfg = cfg.get("training", {})
    n_epochs: int = train_cfg.get("pretrain_epochs", 50)
    batch_size: int = train_cfg.get("pretrain_batch_size", 64)
    lr: float = train_cfg.get("pretrain_lr", 1e-3)
    regime_loss_weight: float = 0.3
    alloc_loss_weight: float = 0.7

    device = macro_agent.device
    actor = macro_agent.actor
    actor.train()

    # Supervised optimizer (separate from PPO optimizer)
    pretrain_optimizer = optim.Adam(actor.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(pretrain_optimizer, T_max=n_epochs)

    early_stop = EarlyStopping(patience=15, mode="min")
    tracker = MetricsTracker(log_dir=log_dir)

    # Convert data to tensors
    states_t = torch.tensor(train_data["macro_states"], dtype=torch.float32, device=device)
    embs_t = torch.tensor(train_data["sector_embeddings"], dtype=torch.float32, device=device)
    returns_t = torch.tensor(train_data["sector_returns"], dtype=torch.float32, device=device)
    regimes_t = torch.tensor(train_data["regime_labels"], dtype=torch.long, device=device)

    n_samples = len(states_t)
    best_loss = float("inf")
    best_path = os.path.join(log_dir, "best_pretrain_macro.pt")

    logger.info(
        "Phase 1: Macro pre-training | samples=%d | epochs=%d | batch=%d",
        n_samples, n_epochs, batch_size,
    )

    for epoch in range(n_epochs):
        perm = torch.randperm(n_samples, device=device)
        epoch_regime_loss = 0.0
        epoch_alloc_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples - batch_size, batch_size):
            idx = perm[start:start + batch_size]

            # Forward pass
            sector_weights, regime_logits, _ = actor(states_t[idx], embs_t[idx])

            # ── Loss 1: Regime classification (cross-entropy) ──
            regime_loss = F.cross_entropy(regime_logits, regimes_t[idx])

            # ── Loss 2: Allocation quality (reward-weighted MSE) ──
            # Oracle allocation: overweight sectors with positive returns
            target_returns = returns_t[idx]
            # Softmax over returns → target allocation that favours winning sectors
            oracle_weights = F.softmax(target_returns * 10.0, dim=-1)
            alloc_loss = F.mse_loss(sector_weights, oracle_weights)

            # Combined loss
            total_loss = regime_loss_weight * regime_loss + alloc_loss_weight * alloc_loss

            pretrain_optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            pretrain_optimizer.step()

            epoch_regime_loss += regime_loss.item()
            epoch_alloc_loss += alloc_loss.item()
            n_batches += 1

        scheduler.step()

        # Epoch metrics
        avg_regime = epoch_regime_loss / max(n_batches, 1)
        avg_alloc = epoch_alloc_loss / max(n_batches, 1)
        avg_total = regime_loss_weight * avg_regime + alloc_loss_weight * avg_alloc

        metrics = {
            "pretrain/regime_loss": avg_regime,
            "pretrain/alloc_loss": avg_alloc,
            "pretrain/total_loss": avg_total,
            "pretrain/lr": pretrain_optimizer.param_groups[0]["lr"],
        }

        # Validation
        if val_data is not None:
            val_metrics = _validate_macro(actor, val_data, device)
            metrics.update(val_metrics)
            monitor_loss = val_metrics["pretrain/val_total_loss"]
        else:
            monitor_loss = avg_total

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

        if (epoch + 1) % 10 == 0:
            logger.info(
                "Epoch %d/%d | regime_loss=%.4f | alloc_loss=%.4f | total=%.4f",
                epoch + 1, n_epochs, avg_regime, avg_alloc, avg_total,
            )

        if early_stop.step(monitor_loss):
            logger.info("Early stopping at epoch %d", epoch + 1)
            break

    tracker.save("pretrain_macro_metrics.json")

    return {
        "best_loss": best_loss,
        "best_checkpoint": best_path,
        "epochs_trained": epoch + 1,
        "metrics_summary": tracker.summary(),
    }


@torch.no_grad()
def _validate_macro(
    actor: nn.Module,
    val_data: Dict[str, np.ndarray],
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate pre-trained Macro actor on validation data."""
    actor.eval()
    states = torch.tensor(val_data["macro_states"], dtype=torch.float32, device=device)
    embs = torch.tensor(val_data["sector_embeddings"], dtype=torch.float32, device=device)
    returns = torch.tensor(val_data["sector_returns"], dtype=torch.float32, device=device)
    regimes = torch.tensor(val_data["regime_labels"], dtype=torch.long, device=device)

    sector_weights, regime_logits, _ = actor(states, embs)

    regime_loss = F.cross_entropy(regime_logits, regimes).item()
    oracle_weights = F.softmax(returns * 10.0, dim=-1)
    alloc_loss = F.mse_loss(sector_weights, oracle_weights).item()

    # Regime accuracy
    pred_regimes = regime_logits.argmax(dim=-1)
    accuracy = float((pred_regimes == regimes).float().mean().item())

    actor.train()
    return {
        "pretrain/val_regime_loss": regime_loss,
        "pretrain/val_alloc_loss": alloc_loss,
        "pretrain/val_total_loss": 0.3 * regime_loss + 0.7 * alloc_loss,
        "pretrain/val_regime_accuracy": accuracy,
    }
