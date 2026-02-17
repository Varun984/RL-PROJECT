"""
File: __init__.py
Module: data
Description: Data layer package for HRL-SARP. Exposes all data fetchers, the feature store,
             and the orchestration pipeline for unified imports.
Design Decisions: Centralised exports enable clean imports like `from data import MarketDataFetcher`.
References: Python packaging best practices (PEP 420, implicit namespace packages)
Author: HRL-SARP Framework
"""

from data.market_data_fetcher import MarketDataFetcher
from data.fundamental_fetcher import FundamentalFetcher
from data.news_fetcher import NewsFetcher
from data.macro_fetcher import MacroFetcher
from data.feature_store import FeatureStore
from data.data_pipeline import DataPipeline

__all__ = [
    "MarketDataFetcher",
    "FundamentalFetcher",
    "NewsFetcher",
    "MacroFetcher",
    "FeatureStore",
    "DataPipeline",
]
