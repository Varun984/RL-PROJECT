"""
File: market_data_fetcher.py
Module: data
Description: Fetches OHLCV market data, F&O option chain data, and index data for Indian
             equities. Uses yfinance as the primary source with Zerodha Kite API and
             jugaad-data as fallbacks. Handles NSE-specific quirks (symbol suffixes,
             delivery volume, circuit limits).
Design Decisions:
    - yfinance chosen as primary because it is free, reliable for historical data, and
      handles stock splits / bonus adjustments automatically. However, it lacks delivery
      volume and real-time F&O data, so Kite API supplements when available.
    - Fallback chain: yfinance → jugaad-data → raise DataFetchError.
    - All fetched data is returned as pandas DataFrames with DatetimeIndex for seamless
      downstream feature engineering.
    - Rate limiting and retry logic built in to respect API/website constraints.
References:
    - yfinance: https://github.com/ranaroussi/yfinance
    - Kite Connect: https://kite.trade/docs/connect/v3/
    - jugaad-data: https://github.com/jugaad-py/jugaad-data
Author: HRL-SARP Framework
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

# ══════════════════════════════════════════════════════════════════════
# LOGGER SETUP
# ══════════════════════════════════════════════════════════════════════
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# CUSTOM EXCEPTIONS
# ══════════════════════════════════════════════════════════════════════
class DataFetchError(Exception):
    """Raised when all data sources fail to return requested data."""
    pass


class SymbolNotFoundError(DataFetchError):
    """Raised when a stock symbol is not found in any data source."""
    pass


# ══════════════════════════════════════════════════════════════════════
# MARKET DATA FETCHER CLASS
# ══════════════════════════════════════════════════════════════════════
class MarketDataFetcher:
    """
    Fetches OHLCV data, F&O option chain data, and index data for Indian equities.

    The fetcher implements a fallback chain:
        1. yfinance (primary, free, historical)
        2. Zerodha Kite API (if credentials available, for live + F&O)
        3. jugaad-data (NSE direct scraping fallback)

    All returned DataFrames use a DatetimeIndex and are adjusted for splits/bonuses.

    Design rationale:
        - Decoupled from downstream consumers via clean DataFrame interface
        - Rate limiting prevents API bans during batch fetches
        - Retry logic with exponential backoff handles transient network failures

    Attributes:
        config: Parsed data_config.yaml dictionary
        kite_enabled: Whether Zerodha Kite API is available
        kite_client: KiteConnect client instance (if enabled)
    """

    def __init__(self, config_path: str = "config/data_config.yaml") -> None:
        """
        Initialise MarketDataFetcher with configuration.

        Args:
            config_path: Path to data_config.yaml file.
        """
        with open(config_path, "r") as f:
            self.config: Dict[str, Any] = yaml.safe_load(f)

        self.market_config = self.config["market_data"]
        self.kite_enabled: bool = self.market_config["kite_api"].get("enabled", False)
        self.kite_client: Optional[Any] = None

        # ── Initialise Kite client if enabled ────────────────────────
        if self.kite_enabled:
            self._init_kite_client()

        logger.info(
            "MarketDataFetcher initialised | primary=%s | kite_enabled=%s",
            self.market_config["primary_source"],
            self.kite_enabled,
        )

    # ══════════════════════════════════════════════════════════════════
    # KITE API INITIALISATION
    # ══════════════════════════════════════════════════════════════════
    def _init_kite_client(self) -> None:
        """
        Initialise Zerodha Kite Connect client.

        Uses environment-variable-based credentials to avoid hardcoding secrets.
        If import or auth fails, falls back gracefully to yfinance-only mode.
        """
        try:
            import os
            from kiteconnect import KiteConnect

            api_key = os.environ.get("KITE_API_KEY", "")
            access_token = os.environ.get("KITE_ACCESS_TOKEN", "")

            if not api_key or not access_token:
                logger.warning(
                    "Kite API credentials not found in environment. Disabling Kite."
                )
                self.kite_enabled = False
                return

            self.kite_client = KiteConnect(api_key=api_key)
            self.kite_client.set_access_token(access_token)
            logger.info("Kite Connect client initialised successfully.")
        except ImportError:
            logger.warning("kiteconnect package not installed. Disabling Kite API.")
            self.kite_enabled = False
        except Exception as e:
            logger.error("Failed to initialise Kite client: %s", e)
            self.kite_enabled = False

    # ══════════════════════════════════════════════════════════════════
    # OHLCV DATA FETCHING
    # ══════════════════════════════════════════════════════════════════
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    def fetch_ohlcv(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
        source: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV (Open, High, Low, Close, Volume) data for a given symbol.

        Implements fallback chain: yfinance → kite → jugaad-data.
        All data is adjusted for stock splits and bonuses.

        Args:
            symbol: NSE stock symbol (e.g., "RELIANCE", "TCS").
                    Automatically appends ".NS" suffix for yfinance.
            start_date: Start date in "YYYY-MM-DD" format.
            end_date: End date in "YYYY-MM-DD" format.
            interval: Candle interval ("1d", "1wk", "1mo").
            source: Force a specific source ("yfinance", "kite", "jugaad").
                    If None, uses fallback chain.

        Returns:
            pd.DataFrame with columns [Open, High, Low, Close, Volume, Adj_Close]
            and DatetimeIndex.

        Raises:
            DataFetchError: If all sources fail.
            SymbolNotFoundError: If symbol not found in any source.
        """
        # Determine source priority
        sources_to_try = (
            [source] if source else [self.market_config["primary_source"], "kite", self.market_config["fallback_source"]]
        )

        last_error: Optional[Exception] = None

        for src in sources_to_try:
            try:
                if src == "yfinance":
                    df = self._fetch_yfinance(symbol, start_date, end_date, interval)
                elif src == "kite" and self.kite_enabled:
                    df = self._fetch_kite(symbol, start_date, end_date, interval)
                elif src == "jugaad_data":
                    df = self._fetch_jugaad(symbol, start_date, end_date)
                else:
                    continue

                if df is not None and not df.empty:
                    # ── Standardise column names ─────────────────────
                    df = self._standardise_columns(df)
                    logger.info(
                        "Fetched OHLCV | symbol=%s | source=%s | rows=%d | range=[%s, %s]",
                        symbol, src, len(df), df.index.min(), df.index.max(),
                    )
                    return df

            except Exception as e:
                last_error = e
                logger.warning(
                    "Source %s failed for %s: %s. Trying next...", src, symbol, e
                )
                continue

        raise DataFetchError(
            f"All sources failed for symbol '{symbol}'. Last error: {last_error}"
        )

    def _fetch_yfinance(
        self, symbol: str, start_date: str, end_date: str, interval: str
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from yfinance.

        yfinance automatically handles:
            - Stock split adjustments
            - Bonus issue adjustments
            - Dividend adjustments (in Adj Close)

        NSE symbols require ".NS" suffix (e.g., "RELIANCE.NS").

        Args:
            symbol: Raw NSE symbol (without .NS suffix).
            start_date: Start date string.
            end_date: End date string.
            interval: Candle interval.

        Returns:
            pd.DataFrame with OHLCV data.
        """
        # ── Append .NS suffix for NSE stocks ────────────────────────
        # Index symbols (starting with ^) don't need the suffix
        yf_symbol = symbol if symbol.startswith("^") else f"{symbol}.NS"

        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(
            start=start_date,
            end=end_date,
            interval=interval,
            auto_adjust=self.market_config["ohlcv"]["adjust_splits"],
        )

        if df.empty:
            raise SymbolNotFoundError(f"No data found for {yf_symbol} on yfinance.")

        return df

    def _fetch_kite(
        self, symbol: str, start_date: str, end_date: str, interval: str
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from Zerodha Kite Connect API.

        Kite API provides real-time and historical data with delivery-volume info.
        Requires valid API credentials (access token refreshed daily).

        Design note: Kite historical data API has a limit of 2000 candles per request,
        so we chunk the date range for longer periods.

        Args:
            symbol: NSE symbol.
            start_date: Start date string.
            end_date: End date string.
            interval: Candle interval (mapped to Kite's format).

        Returns:
            pd.DataFrame with OHLCV data.
        """
        if not self.kite_client:
            raise DataFetchError("Kite client not initialised.")

        # ── Map interval to Kite format ──────────────────────────────
        interval_map = {
            "1d": "day",
            "1wk": "week",
            "1mo": "month",
            "5m": "5minute",
            "15m": "15minute",
        }
        kite_interval = interval_map.get(interval, "day")

        # ── Get instrument token ─────────────────────────────────────
        # Kite API uses instrument tokens, not symbols directly
        instruments = self.kite_client.instruments("NSE")
        instrument_token = None
        for inst in instruments:
            if inst["tradingsymbol"] == symbol:
                instrument_token = inst["instrument_token"]
                break

        if instrument_token is None:
            raise SymbolNotFoundError(f"Symbol {symbol} not found in Kite instruments.")

        # ── Fetch data with date chunking (2000 candle limit) ────────
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        all_data: List[Dict] = []
        chunk_size = timedelta(days=400)  # Conservative chunk to stay under 2000 limit

        current_start = start_dt
        while current_start < end_dt:
            current_end = min(current_start + chunk_size, end_dt)
            data = self.kite_client.historical_data(
                instrument_token,
                from_date=current_start,
                to_date=current_end,
                interval=kite_interval,
            )
            all_data.extend(data)
            current_start = current_end + timedelta(days=1)
            time.sleep(0.5)  # Rate limiting

        if not all_data:
            raise DataFetchError(f"No data returned from Kite for {symbol}.")

        df = pd.DataFrame(all_data)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        return df

    def _fetch_jugaad(
        self, symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from jugaad-data (NSE direct scraping).

        jugaad-data is a Python library that scrapes data directly from NSE India.
        It's useful as a last-resort fallback when yfinance and Kite are unavailable.

        Args:
            symbol: NSE symbol.
            start_date: Start date string.
            end_date: End date string.

        Returns:
            pd.DataFrame with OHLCV data.
        """
        try:
            from jugaad_data.nse import stock_df

            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")

            df = stock_df(
                symbol=symbol,
                from_date=start_dt.date(),
                to_date=end_dt.date(),
                series="EQ",
            )

            if df.empty:
                raise DataFetchError(f"No data from jugaad-data for {symbol}.")

            df = df.set_index("DATE")
            df.index = pd.to_datetime(df.index)
            return df

        except ImportError:
            raise DataFetchError("jugaad-data package not installed.")

    # ══════════════════════════════════════════════════════════════════
    # BATCH OHLCV FETCHING
    # ══════════════════════════════════════════════════════════════════
    def fetch_ohlcv_batch(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        interval: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for multiple symbols with rate limiting.

        This is the recommended method for batch fetching as it:
            - Implements rate limiting between requests
            - Logs progress for long batch operations
            - Continues on individual symbol failures (logs warnings)

        Args:
            symbols: List of NSE stock symbols.
            start_date: Start date in "YYYY-MM-DD" format.
            end_date: End date in "YYYY-MM-DD" format.
            interval: Candle interval.

        Returns:
            Dictionary mapping symbol → OHLCV DataFrame.
            Symbols that failed are excluded (with warning logged).
        """
        results: Dict[str, pd.DataFrame] = {}
        failed: List[str] = []

        for i, symbol in enumerate(symbols):
            try:
                df = self.fetch_ohlcv(symbol, start_date, end_date, interval)
                results[symbol] = df

                # ── Rate limiting: 0.5s between requests ─────────────
                if i < len(symbols) - 1:
                    time.sleep(0.5)

                if (i + 1) % 10 == 0:
                    logger.info(
                        "Batch fetch progress: %d/%d symbols completed.",
                        i + 1, len(symbols),
                    )

            except DataFetchError as e:
                logger.warning("Failed to fetch %s: %s", symbol, e)
                failed.append(symbol)
                continue

        if failed:
            logger.warning(
                "Batch fetch completed with %d failures: %s",
                len(failed), failed,
            )

        logger.info(
            "Batch OHLCV fetch complete | success=%d | failed=%d",
            len(results), len(failed),
        )
        return results

    # ══════════════════════════════════════════════════════════════════
    # INDEX DATA FETCHING
    # ══════════════════════════════════════════════════════════════════
    def fetch_index_data(
        self,
        start_date: str,
        end_date: str,
        interval: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for all configured NSE sector indices.

        Indices configured in data_config.yaml under market_data.indices.
        These include Nifty 50, Bank Nifty, and all 11 sector indices.

        Args:
            start_date: Start date in "YYYY-MM-DD" format.
            end_date: End date in "YYYY-MM-DD" format.
            interval: Candle interval.

        Returns:
            Dictionary mapping index_name → OHLCV DataFrame.
        """
        indices = self.market_config["indices"]
        results: Dict[str, pd.DataFrame] = {}

        for idx_config in indices:
            symbol = idx_config["symbol"]
            name = idx_config["name"]
            try:
                df = self.fetch_ohlcv(symbol, start_date, end_date, interval, source="yfinance")
                results[name] = df
                logger.info("Fetched index: %s (%s) | rows=%d", name, symbol, len(df))
                time.sleep(0.3)  # Rate limit
            except DataFetchError as e:
                logger.warning("Failed to fetch index %s: %s", name, e)

        return results

    # ══════════════════════════════════════════════════════════════════
    # F&O OPTION CHAIN DATA
    # ══════════════════════════════════════════════════════════════════
    def fetch_option_chain(
        self,
        symbol: str = "NIFTY",
        expiry_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch F&O option chain data for a given symbol.

        Option chain data includes strike prices, OI, volume, IV for
        both call and put options. This data is critical for computing
        the Put-Call Ratio (PCR) and implied volatility features.

        Primary source: Kite API (real-time OI).
        Fallback: NSE India website scraping via jugaad-data.

        Args:
            symbol: Underlying symbol ("NIFTY", "BANKNIFTY", or stock symbol).
            expiry_date: Specific expiry in "YYYY-MM-DD". If None, uses nearest monthly.

        Returns:
            pd.DataFrame with columns:
                [strike, call_oi, call_volume, call_iv, call_ltp,
                 put_oi, put_volume, put_iv, put_ltp, pcr_at_strike]
        """
        if self.kite_enabled and self.kite_client:
            return self._fetch_option_chain_kite(symbol, expiry_date)
        else:
            return self._fetch_option_chain_nse(symbol, expiry_date)

    def _fetch_option_chain_kite(
        self, symbol: str, expiry_date: Optional[str]
    ) -> pd.DataFrame:
        """
        Fetch option chain from Kite Connect API.

        Kite provides real-time option chain with OI, volume, Greeks, and IV.
        This is the preferred source for live trading scenarios.

        Args:
            symbol: Underlying symbol.
            expiry_date: Specific expiry date.

        Returns:
            pd.DataFrame with option chain data.
        """
        try:
            # Get instruments for the symbol's F&O segment
            instruments = self.kite_client.instruments("NFO")

            # Filter for the specific underlying and expiry
            options = [
                inst for inst in instruments
                if inst["name"] == symbol
                and inst["instrument_type"] in ("CE", "PE")
            ]

            if expiry_date:
                expiry_dt = datetime.strptime(expiry_date, "%Y-%m-%d").date()
                options = [o for o in options if o["expiry"] == expiry_dt]
            else:
                # ── Use nearest monthly expiry ───────────────────────
                expiries = sorted(set(o["expiry"] for o in options))
                if expiries:
                    nearest_expiry = expiries[0]
                    options = [o for o in options if o["expiry"] == nearest_expiry]

            if not options:
                logger.warning("No options found for %s on Kite.", symbol)
                return pd.DataFrame()

            # ── Build option chain DataFrame ─────────────────────────
            chain_data: List[Dict] = []
            strikes = sorted(set(o["strike"] for o in options))

            for strike in strikes:
                row: Dict[str, Any] = {"strike": strike}

                # Find CE and PE for this strike
                ce = next((o for o in options if o["strike"] == strike and o["instrument_type"] == "CE"), None)
                pe = next((o for o in options if o["strike"] == strike and o["instrument_type"] == "PE"), None)

                if ce:
                    try:
                        ce_quote = self.kite_client.quote(f"NFO:{ce['tradingsymbol']}")
                        ce_data = list(ce_quote.values())[0]
                        row["call_oi"] = ce_data.get("oi", 0)
                        row["call_volume"] = ce_data.get("volume", 0)
                        row["call_ltp"] = ce_data.get("last_price", 0.0)
                    except Exception:
                        row["call_oi"] = 0
                        row["call_volume"] = 0
                        row["call_ltp"] = 0.0

                if pe:
                    try:
                        pe_quote = self.kite_client.quote(f"NFO:{pe['tradingsymbol']}")
                        pe_data = list(pe_quote.values())[0]
                        row["put_oi"] = pe_data.get("oi", 0)
                        row["put_volume"] = pe_data.get("volume", 0)
                        row["put_ltp"] = pe_data.get("last_price", 0.0)
                    except Exception:
                        row["put_oi"] = 0
                        row["put_volume"] = 0
                        row["put_ltp"] = 0.0

                # ── PCR at each strike ───────────────────────────────
                # Put-Call Ratio = Put OI / Call OI
                # A high PCR (>1.0) is considered bullish (more put writing → support)
                call_oi = row.get("call_oi", 0)
                put_oi = row.get("put_oi", 0)
                row["pcr_at_strike"] = put_oi / call_oi if call_oi > 0 else 0.0

                chain_data.append(row)

            return pd.DataFrame(chain_data)

        except Exception as e:
            logger.error("Failed to fetch option chain from Kite: %s", e)
            return pd.DataFrame()

    def _fetch_option_chain_nse(
        self, symbol: str, expiry_date: Optional[str]
    ) -> pd.DataFrame:
        """
        Fetch option chain from NSE India website (fallback).

        Scrapes the NSE option chain page using requests with appropriate headers.
        NSE requires specific User-Agent and cookie handling.

        Args:
            symbol: Underlying symbol.
            expiry_date: Specific expiry date (not used in NSE scraping; uses current).

        Returns:
            pd.DataFrame with option chain data.
        """
        import requests

        url = "https://www.nseindia.com/api/option-chain-indices"
        params = {"symbol": symbol}
        headers = {
            "User-Agent": self.config.get("news", {}).get("scraping", {}).get(
                "user_agent", "Mozilla/5.0"
            ),
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nseindia.com/option-chain",
        }

        try:
            # ── NSE requires a session with cookies ──────────────────
            session = requests.Session()
            # First hit the main page to get cookies
            session.get("https://www.nseindia.com", headers=headers, timeout=10)
            time.sleep(1)

            # Then fetch the API
            response = session.get(url, params=params, headers=headers, timeout=15)
            response.raise_for_status()
            data = response.json()

            records = data.get("records", {}).get("data", [])
            if not records:
                return pd.DataFrame()

            chain_data: List[Dict] = []
            for record in records:
                row: Dict[str, Any] = {"strike": record.get("strikePrice", 0)}

                ce = record.get("CE", {})
                pe = record.get("PE", {})

                row["call_oi"] = ce.get("openInterest", 0)
                row["call_volume"] = ce.get("totalTradedVolume", 0)
                row["call_iv"] = ce.get("impliedVolatility", 0.0)
                row["call_ltp"] = ce.get("lastPrice", 0.0)

                row["put_oi"] = pe.get("openInterest", 0)
                row["put_volume"] = pe.get("totalTradedVolume", 0)
                row["put_iv"] = pe.get("impliedVolatility", 0.0)
                row["put_ltp"] = pe.get("lastPrice", 0.0)

                call_oi = row["call_oi"]
                put_oi = row["put_oi"]
                row["pcr_at_strike"] = put_oi / call_oi if call_oi > 0 else 0.0

                chain_data.append(row)

            return pd.DataFrame(chain_data)

        except Exception as e:
            logger.error("Failed to fetch option chain from NSE: %s", e)
            return pd.DataFrame()

    # ══════════════════════════════════════════════════════════════════
    # DELIVERY VOLUME DATA
    # ══════════════════════════════════════════════════════════════════
    def fetch_delivery_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Fetch delivery volume percentage data for a stock.

        Delivery volume percentage is a key India-specific feature:
            - High delivery % (>60%) indicates genuine buying interest
            - Low delivery % (<30%) suggests speculative / intraday activity
            - Useful for filtering quality trades and avoiding speculative stocks

        Primary source: jugaad-data (NSE bhavcopy includes delivery data).

        Args:
            symbol: NSE stock symbol.
            start_date: Start date string.
            end_date: End date string.

        Returns:
            pd.DataFrame with columns [traded_qty, deliverable_qty, delivery_pct].
        """
        try:
            from jugaad_data.nse import stock_df

            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")

            df = stock_df(
                symbol=symbol,
                from_date=start_dt.date(),
                to_date=end_dt.date(),
                series="EQ",
            )

            if df.empty:
                return pd.DataFrame()

            df = df.set_index("DATE")
            df.index = pd.to_datetime(df.index)

            # ── Extract delivery columns ─────────────────────────────
            delivery_cols = ["TOTAL_TRADED_QUANTITY", "DELIVERABLE_QTY", "DELIV_PER"]
            available_cols = [c for c in delivery_cols if c in df.columns]

            if not available_cols:
                logger.warning(
                    "Delivery data columns not found for %s. Columns: %s",
                    symbol, list(df.columns),
                )
                return pd.DataFrame()

            result = df[available_cols].copy()
            result.columns = ["traded_qty", "deliverable_qty", "delivery_pct"][
                : len(available_cols)
            ]

            return result

        except ImportError:
            logger.warning("jugaad-data not installed. Cannot fetch delivery data.")
            return pd.DataFrame()
        except Exception as e:
            logger.error("Failed to fetch delivery data for %s: %s", symbol, e)
            return pd.DataFrame()

    # ══════════════════════════════════════════════════════════════════
    # PUT-CALL RATIO COMPUTATION
    # ══════════════════════════════════════════════════════════════════
    def compute_pcr(self, option_chain: pd.DataFrame) -> float:
        """
        Compute the aggregate Put-Call Ratio (PCR) from option chain data.

        PCR = Total Put OI / Total Call OI

        Interpretation in Indian markets:
            - PCR > 1.2: Bullish (heavy put writing → strong support)
            - PCR 0.8–1.2: Neutral
            - PCR < 0.8: Bearish (excess call writing → resistance)

        Args:
            option_chain: DataFrame from fetch_option_chain().

        Returns:
            float: Aggregate PCR value.
        """
        if option_chain.empty:
            return 1.0  # Neutral default

        total_call_oi = option_chain["call_oi"].sum()
        total_put_oi = option_chain["put_oi"].sum()

        if total_call_oi == 0:
            return 0.0

        pcr = total_put_oi / total_call_oi
        logger.debug("PCR computed: %.4f (Put OI=%d, Call OI=%d)", pcr, total_put_oi, total_call_oi)
        return pcr

    # ══════════════════════════════════════════════════════════════════
    # COLUMN STANDARDISATION
    # ══════════════════════════════════════════════════════════════════
    @staticmethod
    def _standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardise DataFrame column names to a consistent format.

        Different data sources return columns with different naming conventions.
        This method maps all variations to: [Open, High, Low, Close, Volume, Adj_Close].

        Args:
            df: Raw OHLCV DataFrame.

        Returns:
            pd.DataFrame with standardised column names.
        """
        # ── Common column name mappings ──────────────────────────────
        column_map = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
            "adj close": "Adj_Close",
            "adj_close": "Adj_Close",
            "Adj Close": "Adj_Close",
            # Kite Connect specific
            "date": "Date",
            # jugaad-data specific
            "OPEN": "Open",
            "HIGH": "High",
            "LOW": "Low",
            "CLOSE": "Close",
            "TOTAL_TRADED_QUANTITY": "Volume",
            "PREV_CLOSE": "Prev_Close",
            "LAST": "Last",
        }

        # Apply mapping (case-insensitive)
        new_columns = {}
        for col in df.columns:
            mapped = column_map.get(col, column_map.get(col.lower(), col))
            new_columns[col] = mapped

        df = df.rename(columns=new_columns)

        # ── Ensure Adj_Close exists ──────────────────────────────────
        if "Adj_Close" not in df.columns and "Close" in df.columns:
            df["Adj_Close"] = df["Close"]

        # ── Keep only standard columns ───────────────────────────────
        standard_cols = ["Open", "High", "Low", "Close", "Volume", "Adj_Close"]
        existing = [c for c in standard_cols if c in df.columns]
        df = df[existing]

        # ── Ensure numeric types ─────────────────────────────────────
        for col in existing:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    # ══════════════════════════════════════════════════════════════════
    # UNIVERSE CONSTRUCTION
    # ══════════════════════════════════════════════════════════════════
    def get_sector_universe(self) -> Dict[str, List[str]]:
        """
        Return the sector-to-stocks mapping from configuration.

        This mapping defines the investable universe for the Micro agent.
        Stocks are grouped by NSE 11-sector classification.

        Returns:
            Dictionary mapping sector_name → list of stock symbols.
        """
        sectors_config = self.config.get("sectors", {}).get("mapping", {})
        return dict(sectors_config)

    def get_all_symbols(self) -> List[str]:
        """
        Return a flat list of all stock symbols across all sectors.

        Returns:
            List of all stock symbols in the configured universe.
        """
        sector_map = self.get_sector_universe()
        all_symbols: List[str] = []
        for sector_stocks in sector_map.values():
            all_symbols.extend(sector_stocks)
        return sorted(list(set(all_symbols)))
