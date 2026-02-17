"""
File: __init__.py
Module: training
Description: Training package exposing all 5 training phases, curriculum manager,
    and shared trainer utilities.
Author: HRL-SARP Framework
"""

from training.pretrain_macro import pretrain_macro
from training.pretrain_micro import pretrain_micro
from training.train_micro_frozen_macro import train_micro_frozen_macro
from training.train_macro_frozen_micro import train_macro_frozen_micro
from training.joint_finetune import joint_finetune
from training.curriculum_manager import CurriculumManager
from training.trainer_utils import (
    setup_training,
    save_checkpoint,
    load_checkpoint,
    EarlyStopping,
    MetricsTracker,
)

__all__ = [
    "pretrain_macro",
    "pretrain_micro",
    "train_micro_frozen_macro",
    "train_macro_frozen_micro",
    "joint_finetune",
    "CurriculumManager",
    "setup_training",
    "save_checkpoint",
    "load_checkpoint",
    "EarlyStopping",
    "MetricsTracker",
]
