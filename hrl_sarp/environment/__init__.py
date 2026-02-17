"""
File: __init__.py
Module: environment
Description: Environment package exposing all Gymnasium environments,
    reward functions, India calendar, and market simulator.
Author: HRL-SARP Framework
"""

from environment.base_env import BasePortfolioEnv
from environment.macro_env import MacroEnv
from environment.micro_env import MicroEnv
from environment.hierarchical_env import HierarchicalEnv
from environment.reward_functions import (
    sector_alpha_reward,
    value_discovery_reward,
    drawdown_penalty,
    regime_accuracy_reward,
    turnover_penalty,
    portfolio_calmar_reward,
    compute_total_macro_reward,
    compute_total_micro_reward,
)
from environment.india_calendar import IndiaCalendar
from environment.market_simulator import MarketSimulator

__all__ = [
    "BasePortfolioEnv",
    "MacroEnv",
    "MicroEnv",
    "HierarchicalEnv",
    "sector_alpha_reward",
    "value_discovery_reward",
    "drawdown_penalty",
    "regime_accuracy_reward",
    "turnover_penalty",
    "portfolio_calmar_reward",
    "compute_total_macro_reward",
    "compute_total_micro_reward",
    "IndiaCalendar",
    "MarketSimulator",
]
