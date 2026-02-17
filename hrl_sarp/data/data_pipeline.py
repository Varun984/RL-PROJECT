"""
File: data_pipeline.py
Module: data
Description: Orchestrates all data fetchers (market, fundamental, news, macro) in a
             scheduled pipeline. Supports simple Python-based scheduling (via `schedule`)
             or Apache Airflow DAG definition for production deployments.
Design Decisions:
    - Simple scheduler as default for research use (no external dependencies).
    - Airflow DAG definition included for production (commented out, easy to enable).
    - Pipeline runs fetchers sequentially with error isolation per step.
    - Market-hours-aware: only fetches during NSE trading hours (09:15–15:30 IST).
References:
    - schedule library: https://github.com/dbader/schedule
    - Apache Airflow: https://airflow.apache.org/
Author: HRL-SARP Framework
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pytz
import schedule
import yaml

from data.feature_store import FeatureStore
from data.fundamental_fetcher import FundamentalFetcher
from data.macro_fetcher import MacroFetcher
from data.market_data_fetcher import MarketDataFetcher
from data.news_fetcher import NewsFetcher

logger = logging.getLogger(__name__)

# ── India Standard Time ──────────────────────────────────────────────
IST = pytz.timezone("Asia/Kolkata")


class DataPipeline:
    """
    Orchestrates all data fetchers into a unified pipeline.

    The pipeline coordinates:
        1. Market OHLCV data fetching (daily, post-market close)
        2. Fundamental data refresh (weekly, during off-hours)
        3. News scraping (every 30 minutes during market hours)
        4. Macro data aggregation (daily, post-market)
        5. Feature store persistence (after each fetch)

    Scheduling modes:
        - "simple": Python `schedule` library (default, for research)
        - "airflow": Generates Airflow DAG (for production orchestration)

    Design: Each pipeline step is isolated — failure in one step does not
    block others. All errors are logged, and the pipeline continues.
    """

    def __init__(self, config_path: str = "config/data_config.yaml") -> None:
        with open(config_path, "r") as f:
            self.config: Dict[str, Any] = yaml.safe_load(f)

        self.pipeline_config = self.config["pipeline"]
        self.dates_config = self.config["dates"]

        # ── Initialise all fetchers ──────────────────────────────────
        self.market_fetcher = MarketDataFetcher(config_path)
        self.fundamental_fetcher = FundamentalFetcher(config_path)
        self.news_fetcher = NewsFetcher(config_path)
        self.macro_fetcher = MacroFetcher(config_path)
        self.feature_store = FeatureStore(config_path)

        self._is_running: bool = False
        logger.info("DataPipeline initialised | scheduler=%s", self.pipeline_config["scheduler"])

    # ══════════════════════════════════════════════════════════════════
    # PIPELINE STEPS
    # ══════════════════════════════════════════════════════════════════
    def run_market_data_fetch(self, start_date: Optional[str] = None,
                              end_date: Optional[str] = None) -> Dict[str, int]:
        """
        Fetch OHLCV market data for all stocks in the universe and persist.

        Returns:
            Dict with success/failure counts.
        """
        start = start_date or self.dates_config["train_start"]
        end = end_date or datetime.now().strftime("%Y-%m-%d")

        logger.info("Starting market data fetch: %s to %s", start, end)
        all_symbols = self.market_fetcher.get_all_symbols()
        results = self.market_fetcher.fetch_ohlcv_batch(all_symbols, start, end)

        # ── Persist to feature store ─────────────────────────────────
        persisted = 0
        for symbol, df in results.items():
            try:
                for date_idx, row in df.iterrows():
                    date_str = date_idx.strftime("%Y-%m-%d")
                    features = {
                        "open": row.get("Open", 0), "high": row.get("High", 0),
                        "low": row.get("Low", 0), "close": row.get("Close", 0),
                        "volume": row.get("Volume", 0),
                    }
                    self.feature_store.write_stock_features(date_str, symbol, features)
                persisted += 1
            except Exception as e:
                logger.warning("Failed to persist %s: %s", symbol, e)

        stats = {"total": len(all_symbols), "fetched": len(results), "persisted": persisted}
        logger.info("Market data fetch complete: %s", stats)
        return stats

    def run_index_data_fetch(self, start_date: Optional[str] = None,
                             end_date: Optional[str] = None) -> int:
        """Fetch all sector index data and store as sector features."""
        start = start_date or self.dates_config["train_start"]
        end = end_date or datetime.now().strftime("%Y-%m-%d")

        index_data = self.market_fetcher.fetch_index_data(start, end)
        count = 0
        for name, df in index_data.items():
            for date_idx, row in df.iterrows():
                date_str = date_idx.strftime("%Y-%m-%d")
                features = {"index_close": row.get("Close", 0), "index_volume": row.get("Volume", 0)}
                self.feature_store.write_sector_features(date_str, name, features)
                count += 1

        logger.info("Index data fetch complete: %d records persisted", count)
        return count

    def run_fundamental_fetch(self) -> int:
        """Fetch fundamental data for all stocks (run weekly)."""
        all_symbols = self.market_fetcher.get_all_symbols()
        logger.info("Starting fundamental fetch for %d symbols", len(all_symbols))

        df = self.fundamental_fetcher.fetch_fundamentals_batch(all_symbols)
        if df.empty:
            logger.warning("No fundamental data fetched.")
            return 0

        # Persist fundamentals
        today = datetime.now().strftime("%Y-%m-%d")
        count = 0
        for symbol, row in df.iterrows():
            features = {k: v for k, v in row.items() if isinstance(v, (int, float)) and not pd.isna(v)}
            self.feature_store.write_stock_features(today, str(symbol), features)
            count += 1

        logger.info("Fundamental fetch complete: %d symbols persisted", count)
        return count

    def run_news_fetch(self) -> int:
        """Fetch news articles and compute per-sector/stock sentiment."""
        articles = self.news_fetcher.fetch_all_news()
        if not articles:
            logger.warning("No news articles fetched.")
            return 0

        # ── Aggregate sentiment by sector ────────────────────────────
        # Simple approach: average positive/negative tagging per sector
        # Full FinBERT scoring happens in sentiment_features.py
        today = datetime.now().strftime("%Y-%m-%d")
        sector_counts: Dict[str, int] = {}
        for article in articles:
            for sector in article.sectors:
                sector_counts[sector] = sector_counts.get(sector, 0) + 1

        for sector, count in sector_counts.items():
            self.feature_store.write_sentiment_score(today, sector, "sector", 0.0, count)

        logger.info("News fetch complete: %d articles, %d sectors tagged", len(articles), len(sector_counts))
        return len(articles)

    def run_macro_fetch(self, start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> int:
        """Fetch all macro data and persist to feature store."""
        start = start_date or (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        end = end_date or datetime.now().strftime("%Y-%m-%d")

        try:
            macro_df = self.macro_fetcher.fetch_all_macro(start, end)
        except Exception as e:
            logger.error("Macro fetch failed: %s", e)
            return 0

        count = 0
        for date_idx, row in macro_df.iterrows():
            date_str = date_idx.strftime("%Y-%m-%d")
            features = {k: v for k, v in row.items() if not pd.isna(v)}
            self.feature_store.write_macro_features(date_str, features)
            count += 1

        logger.info("Macro fetch complete: %d dates persisted", count)
        return count

    # ══════════════════════════════════════════════════════════════════
    # FULL PIPELINE RUN
    # ══════════════════════════════════════════════════════════════════
    def run_full_pipeline(self, start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the complete data pipeline: market → index → macro → news → fundamentals.

        Each step is error-isolated; failure in one step doesn't block others.

        Args:
            start_date: Start date for historical data fetch.
            end_date: End date for fetching.
        Returns:
            Dictionary with per-step results.
        """
        logger.info("═══ Starting full data pipeline ═══")
        results: Dict[str, Any] = {}

        steps = [
            ("market_data", lambda: self.run_market_data_fetch(start_date, end_date)),
            ("index_data", lambda: self.run_index_data_fetch(start_date, end_date)),
            ("macro_data", lambda: self.run_macro_fetch(start_date, end_date)),
            ("news_data", lambda: self.run_news_fetch()),
            ("fundamental_data", lambda: self.run_fundamental_fetch()),
        ]

        for step_name, step_fn in steps:
            try:
                logger.info("Running pipeline step: %s", step_name)
                result = step_fn()
                results[step_name] = {"status": "success", "result": result}
            except Exception as e:
                logger.error("Pipeline step '%s' failed: %s", step_name, e)
                results[step_name] = {"status": "failed", "error": str(e)}

        logger.info("═══ Full pipeline complete ═══ Results: %s", {k: v["status"] for k, v in results.items()})
        return results

    # ══════════════════════════════════════════════════════════════════
    # SCHEDULER (SIMPLE MODE)
    # ══════════════════════════════════════════════════════════════════
    def _is_market_hours(self) -> bool:
        """Check if current IST time is within NSE market hours."""
        now = datetime.now(IST)
        mh = self.pipeline_config["market_hours"]
        open_time = datetime.strptime(mh["open"], "%H:%M").time()
        close_time = datetime.strptime(mh["close"], "%H:%M").time()
        return open_time <= now.time() <= close_time and now.weekday() < 5

    def start_scheduler(self) -> None:
        """
        Start the simple scheduler for continuous data fetching.

        Schedule:
            - Market data: daily at 16:00 IST (after market close)
            - Macro data: daily at 16:30 IST
            - News: every 30 minutes during market hours
            - Fundamentals: every Saturday at 06:00 IST
        """
        logger.info("Starting data pipeline scheduler...")
        self._is_running = True

        # Daily post-market
        schedule.every().day.at("16:00").do(self.run_market_data_fetch)
        schedule.every().day.at("16:15").do(self.run_index_data_fetch)
        schedule.every().day.at("16:30").do(self.run_macro_fetch)

        # News every 30 minutes during market hours
        schedule.every(30).minutes.do(
            lambda: self.run_news_fetch() if self._is_market_hours() else None
        )

        # Weekly fundamentals
        schedule.every().saturday.at("06:00").do(self.run_fundamental_fetch)

        logger.info("Scheduler started. Press Ctrl+C to stop.")
        try:
            while self._is_running:
                schedule.run_pending()
                time.sleep(60)
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user.")
            self._is_running = False

    def stop_scheduler(self) -> None:
        """Stop the scheduler."""
        self._is_running = False
        schedule.clear()
        logger.info("Scheduler stopped.")


# ══════════════════════════════════════════════════════════════════════
# AIRFLOW DAG DEFINITION (OPTIONAL)
# ══════════════════════════════════════════════════════════════════════
# Uncomment and configure for Apache Airflow deployment
#
# from airflow import DAG
# from airflow.operators.python import PythonOperator
# from datetime import timedelta
#
# default_args = {
#     "owner": "hrl_sarp",
#     "depends_on_past": False,
#     "email_on_failure": True,
#     "email_on_retry": False,
#     "retries": 2,
#     "retry_delay": timedelta(minutes=5),
# }
#
# dag = DAG(
#     "hrl_sarp_data_pipeline",
#     default_args=default_args,
#     description="HRL-SARP data ingestion pipeline",
#     schedule_interval="0 16 * * 1-5",  # 4 PM IST on weekdays
#     start_date=datetime(2024, 1, 1),
#     catchup=False,
# )
#
# pipeline = DataPipeline()
#
# market_task = PythonOperator(task_id="fetch_market_data", python_callable=pipeline.run_market_data_fetch, dag=dag)
# index_task = PythonOperator(task_id="fetch_index_data", python_callable=pipeline.run_index_data_fetch, dag=dag)
# macro_task = PythonOperator(task_id="fetch_macro_data", python_callable=pipeline.run_macro_fetch, dag=dag)
# news_task = PythonOperator(task_id="fetch_news", python_callable=pipeline.run_news_fetch, dag=dag)
# fundamental_task = PythonOperator(task_id="fetch_fundamentals", python_callable=pipeline.run_fundamental_fetch, dag=dag)
#
# market_task >> index_task >> macro_task >> news_task >> fundamental_task


# Need pd for isna checks in pipeline steps
import pandas as pd
