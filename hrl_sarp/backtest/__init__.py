"""
File: __init__.py
Module: backtest
Description: Backtesting package exposing walk-forward backtester,
    performance metrics, and benchmark comparison utilities.
Author: HRL-SARP Framework
"""

from backtest.backtester import Backtester
from backtest.performance_metrics import PerformanceMetrics
from backtest.benchmark_comparison import BenchmarkComparison

__all__ = [
    "Backtester",
    "PerformanceMetrics",
    "BenchmarkComparison",
]
