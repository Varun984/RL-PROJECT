"""
File: reward_functions.py
Module: environment
Description: All reward components as standalone, pure functions. These are composed
    into total rewards by the environments. Each function maps directly to the
    reward shaping formulas specified in the HRL-SARP design.
Design Decisions: Standalone functions enable unit testing, ablation studies, and
    easy reweighting without touching environment logic.
References: PPO reward design (Schulman 2017), Hierarchical goal conditioning (Nachum 2018)
Author: HRL-SARP Framework
"""

import numpy as np
from typing import Optional


# ══════════════════════════════════════════════════════════════════════
# INDIVIDUAL REWARD COMPONENTS
# ══════════════════════════════════════════════════════════════════════


def sector_alpha_reward(
    sector_returns: np.ndarray,
    sector_weights: np.ndarray,
    benchmark_return: float,
    herfindahl_penalty_coef: float = 0.1,
    turnover_cost: float = 0.0,
    turnover_penalty_coef: float = 0.001,
) -> float:
    """R_macro = (sector_alpha_vs_nifty) - 0.1 * HHI - 0.001 * turnover_cost

    Args:
        sector_returns: Array of per-sector returns for the period.
        sector_weights: Agent's sector allocation weights.
        benchmark_return: Nifty 50 return for the same period.
        herfindahl_penalty_coef: Penalises concentration.
        turnover_cost: Total turnover cost this period.
        turnover_penalty_coef: Weight on turnover cost penalty.
    """
    portfolio_return = float(np.dot(sector_weights, sector_returns))
    alpha = portfolio_return - benchmark_return

    # Herfindahl-Hirschman Index measures concentration
    hhi = float(np.sum(sector_weights ** 2))

    reward = alpha - herfindahl_penalty_coef * hhi - turnover_penalty_coef * turnover_cost
    return reward


def value_discovery_reward(
    weekly_return: float,
    pe_zscore: float,
    pe_threshold: float = -0.5,
) -> float:
    """R_value_bonus = return_this_week * I(PE_z < threshold) * I(return > 0)

    Rewards the agent for picking undervalued stocks that subsequently appreciate.
    """
    if pe_zscore < pe_threshold and weekly_return > 0:
        return weekly_return
    return 0.0


def drawdown_penalty(
    current_drawdown: float,
    threshold: float = 0.05,
) -> float:
    """Quadratic penalty for drawdown exceeding threshold.

    Returns: -0.2 * max(0, drawdown - threshold)^2
    """
    excess = max(0.0, current_drawdown - threshold)
    return -(excess ** 2)


def regime_accuracy_reward(
    predicted_regime: int,
    realised_regime: int,
    correct_reward: float = 0.3,
    incorrect_penalty: float = -0.1,
) -> float:
    """R_regime = +0.3 if correct, -0.1 if wrong."""
    return correct_reward if predicted_regime == realised_regime else incorrect_penalty


def turnover_penalty(
    old_weights: np.ndarray,
    new_weights: np.ndarray,
) -> float:
    """L1 turnover between consecutive allocations."""
    return float(np.sum(np.abs(new_weights - old_weights)))


def portfolio_calmar_reward(
    calmar_ratio: float,
    cvar_95: float,
    stt_cost: float,
) -> float:
    """R_portfolio = calmar_ratio * (1 - CVaR_95%) - STT_cost"""
    return calmar_ratio * (1.0 - cvar_95) - stt_cost


def goal_alignment_reward(
    achieved_sector_weights: np.ndarray,
    target_sector_weights: np.ndarray,
) -> float:
    """Cosine similarity between achieved and target sector allocations.

    Used by the Micro agent to measure how closely its stock picks
    align with the Macro agent's sector recommendations.
    """
    dot = float(np.dot(achieved_sector_weights, target_sector_weights))
    norm_a = float(np.linalg.norm(achieved_sector_weights))
    norm_b = float(np.linalg.norm(target_sector_weights))
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return dot / (norm_a * norm_b)


def sharpe_reward(
    returns: np.ndarray,
    annualise_factor: float = np.sqrt(52),
) -> float:
    """Weekly Sharpe ratio (annualised by sqrt(52) for weekly data)."""
    if len(returns) < 2:
        return 0.0
    mean_r = float(np.mean(returns))
    std_r = float(np.std(returns))
    if std_r < 1e-8:
        return 0.0
    return mean_r / std_r * annualise_factor


# ══════════════════════════════════════════════════════════════════════
# COMPOSITE REWARD FUNCTIONS
# ══════════════════════════════════════════════════════════════════════


def compute_total_macro_reward(
    sector_returns: np.ndarray,
    sector_weights: np.ndarray,
    benchmark_return: float,
    calmar_ratio: float,
    cvar_95: float,
    stt_cost: float,
    predicted_regime: int,
    realised_regime: int,
    pe_zscore: float,
    weekly_return: float,
    turnover_cost: float = 0.0,
    w_macro: float = 0.4,
    w_portfolio: float = 0.3,
    w_regime: float = 0.2,
    w_value: float = 0.1,
    herfindahl_coef: float = 0.1,
    turnover_coef: float = 0.001,
    regime_correct: float = 0.3,
    regime_incorrect: float = -0.1,
    pe_threshold: float = -0.5,
) -> float:
    """R_total_macro = 0.4*R_macro + 0.3*R_portfolio + 0.2*R_regime + 0.1*R_value"""
    r_macro = sector_alpha_reward(
        sector_returns, sector_weights, benchmark_return,
        herfindahl_coef, turnover_cost, turnover_coef,
    )
    r_portfolio = portfolio_calmar_reward(calmar_ratio, cvar_95, stt_cost)
    r_regime = regime_accuracy_reward(
        predicted_regime, realised_regime, regime_correct, regime_incorrect,
    )
    r_value = value_discovery_reward(weekly_return, pe_zscore, pe_threshold)

    return w_macro * r_macro + w_portfolio * r_portfolio + w_regime * r_regime + w_value * r_value


def compute_total_micro_reward(
    weekly_returns: np.ndarray,
    achieved_sector_weights: np.ndarray,
    target_sector_weights: np.ndarray,
    current_drawdown: float,
    calmar_ratio: float,
    cvar_95: float,
    stt_cost: float,
    pe_zscore: float,
    weekly_return: float,
    w_micro: float = 0.5,
    w_portfolio: float = 0.3,
    w_value: float = 0.2,
    sharpe_weight: float = 0.5,
    goal_weight: float = 0.3,
    dd_weight: float = 0.2,
    dd_threshold: float = 0.05,
    pe_threshold: float = -0.5,
) -> float:
    """R_total_micro = 0.5*R_micro + 0.3*R_portfolio + 0.2*R_value

    R_micro = 0.5*sharpe + 0.3*goal_cosine - 0.2*drawdown_penalty
    """
    s = sharpe_reward(weekly_returns)
    g = goal_alignment_reward(achieved_sector_weights, target_sector_weights)
    d = drawdown_penalty(current_drawdown, dd_threshold)

    r_micro = sharpe_weight * s + goal_weight * g + dd_weight * d
    r_portfolio = portfolio_calmar_reward(calmar_ratio, cvar_95, stt_cost)
    r_value = value_discovery_reward(weekly_return, pe_zscore, pe_threshold)

    return w_micro * r_micro + w_portfolio * r_portfolio + w_value * r_value
