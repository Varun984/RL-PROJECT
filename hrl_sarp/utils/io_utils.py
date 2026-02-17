"""
File: io_utils.py
Module: utils
Description: File I/O utilities for YAML, JSON, and checkpoint management.
Author: HRL-SARP Framework
"""

import json
import logging
import os
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)


def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    logger.debug("Loaded YAML: %s", path)
    return data or {}


def save_yaml(data: Dict[str, Any], path: str) -> None:
    """Save data to a YAML file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    logger.debug("Saved YAML: %s", path)


def load_json(path: str) -> Dict[str, Any]:
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.debug("Loaded JSON: %s", path)
    return data


def save_json(data: Any, path: str, indent: int = 2) -> None:
    """Save data to a JSON file with numpy type handling."""
    import numpy as np

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return str(obj)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=_convert)
    logger.debug("Saved JSON: %s", path)


def ensure_dir(path: str) -> str:
    """Ensure directory exists, creating it if necessary."""
    os.makedirs(path, exist_ok=True)
    return path
