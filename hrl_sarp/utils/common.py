"""
File: common.py
Module: utils
Description: Common utility functions used across the HRL-SARP framework.
    Includes seed management, device selection, logging setup, and helpers.
Author: HRL-SARP Framework
"""

import logging
import os
import random
import sys
from datetime import datetime
from typing import Optional

import numpy as np
import torch


def set_global_seed(seed: int = 42) -> None:
    """Set random seed for full reproducibility across all libraries.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Deterministic mode for CUDA (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logging.getLogger(__name__).info("Global seed set to %d", seed)


def get_device(prefer_gpu: bool = True) -> torch.device:
    """Get the best available compute device.

    Args:
        prefer_gpu: Whether to prefer CUDA GPU if available.

    Returns:
        torch.device instance.
    """
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        logging.getLogger(__name__).info(
            "Using GPU: %s (%.1f GB)", gpu_name, gpu_mem
        )
    elif prefer_gpu and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.getLogger(__name__).info("Using Apple MPS backend")
    else:
        device = torch.device("cpu")
        logging.getLogger(__name__).info("Using CPU")

    return device


def setup_logging(
    log_dir: str = "logs",
    log_level: str = "INFO",
    experiment_name: Optional[str] = None,
) -> logging.Logger:
    """Configure structured logging for the framework.

    Args:
        log_dir: Directory for log files.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
        experiment_name: Optional experiment name for log file.

    Returns:
        Root logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)

    # Timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = experiment_name or "hrl_sarp"
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")

    # Format
    fmt = "%(asctime)s | %(levelname)-7s | %(name)-25s | %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=date_fmt)

    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    root_logger.info("Logging initialised -> %s", log_file)

    return root_logger


def count_parameters(model: torch.nn.Module) -> int:
    """Count total and trainable parameters in a PyTorch model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.getLogger(__name__).info(
        "Model params: %d total, %d trainable", total, trainable
    )
    return trainable


def soft_update(
    target: torch.nn.Module,
    source: torch.nn.Module,
    tau: float = 0.005,
) -> None:
    """Polyak-averaged soft update of target network parameters.

    θ_target = τ·θ_source + (1-τ)·θ_target
    """
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)


def hard_update(
    target: torch.nn.Module,
    source: torch.nn.Module,
) -> None:
    """Hard copy of source parameters to target."""
    target.load_state_dict(source.state_dict())


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute explained variance (useful for value function diagnostics).

    EV = 1 - Var(y_true - y_pred) / Var(y_true)
    Returns -inf if y_true is constant, 1.0 for perfect prediction.
    """
    var_true = np.var(y_true)
    if var_true < 1e-10:
        return float("-inf")
    return float(1.0 - np.var(y_true - y_pred) / var_true)
