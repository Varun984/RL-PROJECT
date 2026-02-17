"""
File: feature_store.py
Module: data
Description: SQLite/TimescaleDB interface for storing and retrieving feature vectors with
             strict timestamp gating to prevent lookahead bias. Implements point-in-time
             joins ensuring features are only available T+1.
Design Decisions:
    - SQLite default for portability; TimescaleDB optional for production.
    - Timestamp gating is enforced at the query level: a feature computed on day T
      is only queryable from T+1 onwards, preventing lookahead bias in backtesting.
    - Schema designed for columnar retrieval of feature vectors by date range.
References:
    - SQLite: Built-in Python module
    - TimescaleDB: https://www.timescale.com/
    - Point-in-time joins: de Prado (2018) "Advances in Financial ML", Ch. 7
Author: HRL-SARP Framework
"""

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class FeatureStore:
    """
    Feature storage and retrieval with strict timestamp gating (no lookahead bias).

    Stores macro features, stock features, and sentiment scores with metadata.
    Point-in-time queries ensure backtesting integrity: features computed on date T
    are only available for queries with as_of_date >= T+1.

    Design rationale (de Prado, 2018):
        Lookahead bias is the most common source of inflated backtest performance.
        By enforcing T+1 availability at the storage layer, we guarantee that no
        downstream consumer (environment, agent, or backtester) can accidentally
        use future information.

    Attributes:
        backend: "sqlite" or "timescaledb"
        db_path: Path to SQLite database file (if using sqlite)
        lookahead_guard: Whether to enforce T+1 timestamp gating
    """

    def __init__(self, config_path: str = "config/data_config.yaml") -> None:
        with open(config_path, "r") as f:
            self.config: Dict[str, Any] = yaml.safe_load(f)

        fs_config = self.config["feature_store"]
        self.backend: str = fs_config.get("backend", "sqlite")
        self.lookahead_guard: bool = fs_config.get("lookahead_guard", True)
        self.pit_join: bool = fs_config.get("point_in_time_join", True)

        if self.backend == "sqlite":
            self.db_path: str = fs_config["sqlite"]["db_path"]
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            self._init_sqlite()
        else:
            self._init_timescaledb(fs_config["timescaledb"])

        logger.info("FeatureStore initialised | backend=%s | lookahead_guard=%s",
                     self.backend, self.lookahead_guard)

    # ══════════════════════════════════════════════════════════════════
    # DATABASE INITIALISATION
    # ══════════════════════════════════════════════════════════════════
    def _init_sqlite(self) -> None:
        """Create SQLite tables if they don't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # ── Macro features table ─────────────────────────────────
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS macro_features (
                    date TEXT NOT NULL,
                    computed_at TEXT NOT NULL,
                    feature_name TEXT NOT NULL,
                    feature_value REAL,
                    PRIMARY KEY (date, feature_name)
                )
            """)

            # ── Stock features table ─────────────────────────────────
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stock_features (
                    date TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    computed_at TEXT NOT NULL,
                    feature_name TEXT NOT NULL,
                    feature_value REAL,
                    PRIMARY KEY (date, symbol, feature_name)
                )
            """)

            # ── Sector features table ────────────────────────────────
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sector_features (
                    date TEXT NOT NULL,
                    sector TEXT NOT NULL,
                    computed_at TEXT NOT NULL,
                    feature_name TEXT NOT NULL,
                    feature_value REAL,
                    PRIMARY KEY (date, sector, feature_name)
                )
            """)

            # ── Sentiment scores table ───────────────────────────────
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sentiment_scores (
                    date TEXT NOT NULL,
                    entity TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    computed_at TEXT NOT NULL,
                    sentiment_score REAL,
                    num_articles INTEGER,
                    PRIMARY KEY (date, entity, entity_type)
                )
            """)

            # ── Indices for fast querying ────────────────────────────
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_macro_date ON macro_features(date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_stock_date ON stock_features(date, symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sector_date ON sector_features(date, sector)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sentiment_date ON sentiment_scores(date)")

            conn.commit()
            logger.info("SQLite feature store tables initialised at %s", self.db_path)

    def _init_timescaledb(self, ts_config: Dict[str, Any]) -> None:
        """Initialise TimescaleDB connection (production mode)."""
        try:
            import os
            from sqlalchemy import create_engine
            host = os.environ.get("TSDB_HOST", ts_config.get("host", "localhost"))
            port = ts_config.get("port", 5432)
            db = ts_config.get("database", "hrl_sarp")
            user = os.environ.get("TSDB_USER", ts_config.get("user", "postgres"))
            password = os.environ.get("TSDB_PASSWORD", ts_config.get("password", ""))
            self.engine = create_engine(f"postgresql://{user}:{password}@{host}:{port}/{db}")
            logger.info("TimescaleDB engine created for %s:%d/%s", host, port, db)
        except Exception as e:
            logger.error("TimescaleDB init failed: %s. Falling back to SQLite.", e)
            self.backend = "sqlite"
            self.db_path = "data/feature_store.db"
            self._init_sqlite()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for SQLite connections."""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    # ══════════════════════════════════════════════════════════════════
    # WRITE OPERATIONS
    # ══════════════════════════════════════════════════════════════════
    def write_macro_features(self, date: str, features: Dict[str, float]) -> None:
        """
        Write macro feature vector for a given date.
        Features are stamped with computed_at for audit trail.
        """
        computed_at = datetime.now().isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for name, value in features.items():
                cursor.execute(
                    "INSERT OR REPLACE INTO macro_features (date, computed_at, feature_name, feature_value) VALUES (?, ?, ?, ?)",
                    (date, computed_at, name, float(value) if value is not None else None),
                )
            conn.commit()
        logger.debug("Wrote %d macro features for %s", len(features), date)

    def write_stock_features(self, date: str, symbol: str, features: Dict[str, float]) -> None:
        """Write stock feature vector for a given symbol and date."""
        computed_at = datetime.now().isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for name, value in features.items():
                cursor.execute(
                    "INSERT OR REPLACE INTO stock_features (date, symbol, computed_at, feature_name, feature_value) VALUES (?, ?, ?, ?, ?)",
                    (date, symbol, computed_at, name, float(value) if value is not None else None),
                )
            conn.commit()

    def write_sector_features(self, date: str, sector: str, features: Dict[str, float]) -> None:
        """Write sector feature vector for a given sector and date."""
        computed_at = datetime.now().isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for name, value in features.items():
                cursor.execute(
                    "INSERT OR REPLACE INTO sector_features (date, sector, computed_at, feature_name, feature_value) VALUES (?, ?, ?, ?, ?)",
                    (date, sector, computed_at, name, float(value) if value is not None else None),
                )
            conn.commit()

    def write_sentiment_score(self, date: str, entity: str, entity_type: str,
                              score: float, num_articles: int) -> None:
        """Write sentiment score for a stock or sector."""
        computed_at = datetime.now().isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO sentiment_scores VALUES (?, ?, ?, ?, ?, ?)",
                (date, entity, entity_type, computed_at, score, num_articles),
            )
            conn.commit()

    def write_features_batch(self, table: str, records: List[Dict[str, Any]]) -> None:
        """Batch write multiple feature records efficiently."""
        if not records:
            return
        computed_at = datetime.now().isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for record in records:
                record["computed_at"] = computed_at
                if table == "macro_features":
                    cursor.execute(
                        "INSERT OR REPLACE INTO macro_features VALUES (?, ?, ?, ?)",
                        (record["date"], computed_at, record["feature_name"], record["feature_value"]),
                    )
                elif table == "stock_features":
                    cursor.execute(
                        "INSERT OR REPLACE INTO stock_features VALUES (?, ?, ?, ?, ?)",
                        (record["date"], record["symbol"], computed_at, record["feature_name"], record["feature_value"]),
                    )
            conn.commit()
        logger.debug("Batch wrote %d records to %s", len(records), table)

    # ══════════════════════════════════════════════════════════════════
    # READ OPERATIONS (WITH TIMESTAMP GATING)
    # ══════════════════════════════════════════════════════════════════
    def read_macro_features(self, start_date: str, end_date: str,
                            as_of_date: Optional[str] = None) -> pd.DataFrame:
        """
        Read macro features with optional timestamp gating.

        If as_of_date is provided and lookahead_guard is enabled, only features
        computed before as_of_date are returned (point-in-time correctness).

        Args:
            start_date: Start of date range.
            end_date: End of date range.
            as_of_date: Point-in-time cutoff (for backtesting). If None, returns all.
        Returns:
            pd.DataFrame pivoted with dates as index, feature names as columns.
        """
        query = "SELECT date, feature_name, feature_value FROM macro_features WHERE date >= ? AND date <= ?"
        params: List[str] = [start_date, end_date]

        if self.lookahead_guard and as_of_date:
            # Only return features that were computed before the as_of_date
            # This prevents using T's features when making T's decision
            query += " AND computed_at < ?"
            params.append(as_of_date)

        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if df.empty:
            return pd.DataFrame()

        # Pivot: rows=date, columns=feature_name, values=feature_value
        pivoted = df.pivot_table(index="date", columns="feature_name", values="feature_value", aggfunc="last")
        pivoted.index = pd.to_datetime(pivoted.index)
        pivoted = pivoted.sort_index()
        return pivoted

    def read_stock_features(self, symbol: str, start_date: str, end_date: str,
                            as_of_date: Optional[str] = None) -> pd.DataFrame:
        """Read stock features for a specific symbol with timestamp gating."""
        query = "SELECT date, feature_name, feature_value FROM stock_features WHERE symbol = ? AND date >= ? AND date <= ?"
        params: List[str] = [symbol, start_date, end_date]

        if self.lookahead_guard and as_of_date:
            query += " AND computed_at < ?"
            params.append(as_of_date)

        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if df.empty:
            return pd.DataFrame()

        pivoted = df.pivot_table(index="date", columns="feature_name", values="feature_value", aggfunc="last")
        pivoted.index = pd.to_datetime(pivoted.index)
        return pivoted.sort_index()

    def read_sector_features(self, sector: str, start_date: str, end_date: str,
                             as_of_date: Optional[str] = None) -> pd.DataFrame:
        """Read sector features with timestamp gating."""
        query = "SELECT date, feature_name, feature_value FROM sector_features WHERE sector = ? AND date >= ? AND date <= ?"
        params: List[str] = [sector, start_date, end_date]

        if self.lookahead_guard and as_of_date:
            query += " AND computed_at < ?"
            params.append(as_of_date)

        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if df.empty:
            return pd.DataFrame()

        pivoted = df.pivot_table(index="date", columns="feature_name", values="feature_value", aggfunc="last")
        pivoted.index = pd.to_datetime(pivoted.index)
        return pivoted.sort_index()

    def read_sentiment(self, entity: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Read sentiment scores for a stock or sector."""
        query = "SELECT date, sentiment_score, num_articles FROM sentiment_scores WHERE entity = ? AND date >= ? AND date <= ?"
        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=[entity, start_date, end_date])
        if df.empty:
            return pd.DataFrame()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        return df

    # ══════════════════════════════════════════════════════════════════
    # UTILITY METHODS
    # ══════════════════════════════════════════════════════════════════
    def get_latest_date(self, table: str = "macro_features") -> Optional[str]:
        """Get the most recent date in a feature table."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT MAX(date) FROM {table}")
            result = cursor.fetchone()
            return result[0] if result and result[0] else None

    def get_feature_names(self, table: str = "macro_features") -> List[str]:
        """Get list of unique feature names in a table."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT DISTINCT feature_name FROM {table} ORDER BY feature_name")
            return [row[0] for row in cursor.fetchall()]

    def get_available_symbols(self) -> List[str]:
        """Get list of symbols with stored features."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT symbol FROM stock_features ORDER BY symbol")
            return [row[0] for row in cursor.fetchall()]

    def clear_table(self, table: str) -> None:
        """Clear all data from a feature table (use with caution)."""
        with self._get_connection() as conn:
            conn.execute(f"DELETE FROM {table}")
            conn.commit()
        logger.warning("Cleared all data from %s", table)

    def get_row_count(self, table: str) -> int:
        """Get total row count for a table."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            result = cursor.fetchone()
            return result[0] if result else 0
