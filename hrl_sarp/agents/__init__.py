"""
File: __init__.py
Module: agents
Description: Agent package exposing Macro (PPO) and Micro (TD3+HER) agents,
    network architectures, replay buffers, and regime detector.
Author: HRL-SARP Framework
"""

from agents.macro_agent import MacroAgent
from agents.micro_agent import MicroAgent
from agents.networks import (
    MacroActorNet,
    MacroCriticNet,
    MicroActorNet,
    TwinCriticNet,
    MultiTimeframeAttentionEncoder,
    GoalConditionedEncoder,
)
from agents.replay_buffer import ReplayBuffer, HERReplayBuffer
from agents.regime_detector import RegimeDetector

__all__ = [
    "MacroAgent",
    "MicroAgent",
    "MacroActorNet",
    "MacroCriticNet",
    "MicroActorNet",
    "TwinCriticNet",
    "MultiTimeframeAttentionEncoder",
    "GoalConditionedEncoder",
    "ReplayBuffer",
    "HERReplayBuffer",
    "RegimeDetector",
]
