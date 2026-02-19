"""
File: macro_fetcher.py
Module: data
Description: Fetches macroeconomic data for India: RBI repo rate, India VIX, USD/INR,
             crude oil, FII/DII flows, and yield curve data. These drive the Macro agent's
             state vector for regime detection and sector allocation.
Design Decisions:
    - yfinance for market-traded instruments (VIX, USD/INR, crude, bonds).
    - NSDL website scraping for FII/DII flow data (India-specific, no public API).
    - RBI data from published bulletin pages.
References:
    - NSDL FPI data: https://www.fpi.nsdl.co.in
    - RBI Bulletin: https://www.rbi.org.in
Author: HRL-SARP Framework
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import yaml
import yfinance as yf
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class MacroDataError(Exception):
    """Raised when macro data fetch fails."""
    pass


class MacroFetcher:
    """
    Fetches macroeconomic signals for the Macro agent's state vector.

    Data sources:
        - India VIX: yfinance (^INDIAVIX)
        - FII/DII flows: NSDL/Moneycontrol scraping
        - USD/INR: yfinance (INR=X)
        - Crude Oil (Brent): yfinance (BZ=F)
        - US 10Y Treasury: yfinance (^TNX)
        - RBI repo rate: manual config / RBI site scraping

    These features form the 18D macro state vector that the PPO-based
    Macro agent uses for sector allocation and regime classification.
    """

    def __init__(self, config_path: str = "config/data_config.yaml") -> None:
        with open(config_path, "r", encoding="utf-8") as f:self.config: Dict[str, Any] = yaml.safe_load(f)
        self.macro_config = self.config["macro"]
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        })
        logger.info("MacroFetcher initialised.")

    # ══════════════════════════════════════════════════════════════════
    # INDIA VIX
    # ══════════════════════════════════════════════════════════════════
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=15))
    def fetch_india_vix(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch India VIX (volatility index) data.

        India VIX measures the market's expectation of 30-day volatility,
        derived from Nifty option prices. It is the single most important
        indicator for regime detection:
            - VIX < 13: Low volatility regime (Bull / Sideways)
            - VIX 13-20: Normal volatility
            - VIX > 20: High volatility regime (Bear / Stress)
            - VIX > 30: Extreme fear (COVID-level)

        Args:
            start_date: Start date "YYYY-MM-DD".
            end_date: End date "YYYY-MM-DD".
        Returns:
            pd.DataFrame with Close (VIX level) and DatetimeIndex.
        """
        symbol = self.macro_config["india_vix"]["symbol"]
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval="1d")
        if df.empty:
            raise MacroDataError("India VIX data empty from yfinance.")
        df = df[["Close"]].rename(columns={"Close": "india_vix"})
        logger.info("Fetched India VIX: %d rows", len(df))
        return df

    # ══════════════════════════════════════════════════════════════════
    # FII/DII FLOWS
    # ══════════════════════════════════════════════════════════════════
    def fetch_fii_dii_flows(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch FII (Foreign Institutional Investor) and DII (Domestic) flow data.

        FII/DII flows are critical drivers of Indian equity markets:
            - Sustained FII buying: bullish signal for large-caps and financials
            - FII selling + DII buying: rotation from foreign to domestic sentiment
            - Dual selling: bearish, liquidity withdrawal
            - Flows normalised by market cap for cross-period comparability

        Primary: NSDL FPI data. Fallback: Moneycontrol aggregated data.

        Args:
            start_date: Start date.
            end_date: End date.
        Returns:
            pd.DataFrame with columns [fii_net_buy, dii_net_buy] in ₹ Crores.
        """
        try:
            return self._fetch_fii_dii_nsdl(start_date, end_date)
        except MacroDataError:
            logger.warning("NSDL FII/DII failed, trying Moneycontrol fallback.")
            return self._fetch_fii_dii_moneycontrol(start_date, end_date)

    def _fetch_fii_dii_nsdl(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch FII data from NSDL FPI reports."""
        url = self.macro_config["fii_dii"]["url"]
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "lxml")
            tables = soup.find_all("table")
            if not tables:
                raise MacroDataError("No tables found on NSDL FPI page.")

            # Parse the main data table
            rows_data: List[Dict[str, Any]] = []
            for table in tables:
                for row in table.find_all("tr")[1:]:  # Skip header
                    cells = row.find_all("td")
                    if len(cells) >= 4:
                        try:
                            date_str = cells[0].get_text(strip=True)
                            buy_val = self._parse_cr_value(cells[1].get_text(strip=True))
                            sell_val = self._parse_cr_value(cells[2].get_text(strip=True))
                            net_val = self._parse_cr_value(cells[3].get_text(strip=True))
                            if net_val is not None:
                                rows_data.append({
                                    "date": date_str,
                                    "fii_net_buy": net_val,
                                })
                        except Exception:
                            continue

            if not rows_data:
                raise MacroDataError("Could not parse NSDL FPI data.")

            df = pd.DataFrame(rows_data)
            df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
            df = df.dropna(subset=["date"]).set_index("date").sort_index()

            # Filter date range
            df = df.loc[start_date:end_date]
            logger.info("Fetched FII/DII flows from NSDL: %d rows", len(df))
            return df

        except Exception as e:
            raise MacroDataError(f"NSDL FII/DII fetch failed: {e}")

    def _fetch_fii_dii_moneycontrol(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fallback: Fetch FII/DII data from Moneycontrol."""
        url = self.macro_config["fii_dii"]["fallback_url"]
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()

            # Parse Moneycontrol's FII/DII activity table
            soup = BeautifulSoup(response.text, "lxml")
            table = soup.find("table", class_="tbldata14")
            if not table:
                # Try alternate selectors
                tables = pd.read_html(response.text)
                if tables:
                    df = tables[0]
                    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
                    return df
                raise MacroDataError("No FII/DII table found on Moneycontrol.")

            rows_data: List[Dict[str, Any]] = []
            for row in table.find_all("tr")[1:]:
                cells = row.find_all("td")
                if len(cells) >= 6:
                    rows_data.append({
                        "date": cells[0].get_text(strip=True),
                        "fii_net_buy": self._parse_cr_value(cells[2].get_text(strip=True)),
                        "dii_net_buy": self._parse_cr_value(cells[5].get_text(strip=True)),
                    })

            df = pd.DataFrame(rows_data)
            df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
            df = df.dropna(subset=["date"]).set_index("date").sort_index()
            df = df.loc[start_date:end_date]
            return df

        except Exception as e:
            raise MacroDataError(f"Moneycontrol FII/DII fallback failed: {e}")

    # ══════════════════════════════════════════════════════════════════
    # CURRENCY — USD/INR
    # ══════════════════════════════════════════════════════════════════
    def fetch_usd_inr(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch USD/INR exchange rate.
        Rupee depreciation is bearish for IT (revenue in USD = tailwind) but bearish
        for importers (Energy, Pharma APIs). USD/INR momentum is a key macro feature.
        """
        symbol = self.macro_config["usd_inr"]["symbol"]
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval="1d")
        if df.empty:
            raise MacroDataError("USD/INR data empty.")
        df = df[["Close"]].rename(columns={"Close": "usd_inr"})
        logger.info("Fetched USD/INR: %d rows", len(df))
        return df

    # ══════════════════════════════════════════════════════════════════
    # CRUDE OIL (BRENT)
    # ══════════════════════════════════════════════════════════════════
    def fetch_crude_oil(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch Brent crude oil futures price.
        Crude is critical for India (net oil importer):
            - Rising crude: bearish for current account, bearish for OMCs (BPCL, IOC)
            - Falling crude: positive for fiscal deficit, positive for consumer spending
        Regime encoding: Low (<$60), Medium ($60-$80), High (>$80), Extreme (>$100)
        """
        symbol = self.macro_config["crude_oil"]["symbol"]
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval="1d")
        if df.empty:
            raise MacroDataError("Crude oil data empty.")
        df = df[["Close"]].rename(columns={"Close": "crude_oil"})

        # ── Encode crude oil regime ──────────────────────────────────
        df["crude_regime"] = pd.cut(
            df["crude_oil"],
            bins=[0, 60, 80, 100, float("inf")],
            labels=[0, 1, 2, 3],  # Low, Medium, High, Extreme
        ).astype(float)

        logger.info("Fetched Crude Oil: %d rows", len(df))
        return df

    # ══════════════════════════════════════════════════════════════════
    # US 10-YEAR TREASURY YIELD
    # ══════════════════════════════════════════════════════════════════
    def fetch_us_treasury(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch US 10-Year Treasury yield.
        Rising US yields → FII outflows from emerging markets (India) → bearish.
        Yield spread (US10Y - India sovereign bond) indicates relative attractiveness.
        """
        symbol = self.macro_config["us_10y"]["symbol"]
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval="1d")
        if df.empty:
            raise MacroDataError("US 10Y Treasury data empty.")
        df = df[["Close"]].rename(columns={"Close": "us_10y_yield"})
        logger.info("Fetched US 10Y: %d rows", len(df))
        return df

    # ══════════════════════════════════════════════════════════════════
    # AGGREGATE MACRO DATA
    # ══════════════════════════════════════════════════════════════════
    def fetch_all_macro(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch and merge all macroeconomic data into a single DataFrame.
        Handles missing data via forward-fill (macro data has holidays/gaps).

        Args:
            start_date: Start date "YYYY-MM-DD".
            end_date: End date "YYYY-MM-DD".
        Returns:
            pd.DataFrame with all macro features, DatetimeIndex, forward-filled.
        """
        dfs: Dict[str, pd.DataFrame] = {}

        # Fetch each macro series independently (graceful degradation)
        fetch_methods = [
            ("india_vix", self.fetch_india_vix),
            ("usd_inr", self.fetch_usd_inr),
            ("crude_oil", self.fetch_crude_oil),
            ("us_10y", self.fetch_us_treasury),
        ]

        for name, method in fetch_methods:
            try:
                dfs[name] = method(start_date, end_date)
            except MacroDataError as e:
                logger.warning("Macro fetch '%s' failed: %s", name, e)

        try:
            dfs["fii_dii"] = self.fetch_fii_dii_flows(start_date, end_date)
        except MacroDataError as e:
            logger.warning("FII/DII fetch failed: %s", e)

        if not dfs:
            raise MacroDataError("All macro data fetches failed.")

        # ── Merge on date index ──────────────────────────────────────
        combined = None
        for name, df in dfs.items():
            if combined is None:
                combined = df
            else:
                combined = combined.join(df, how="outer")

        # Forward-fill gaps (weekends, holidays produce NaNs in outer join)
        combined = combined.ffill().bfill()

        logger.info("Aggregated macro data: %d rows, %d columns", len(combined), len(combined.columns))
        return combined

    # ══════════════════════════════════════════════════════════════════
    # RBI REPO RATE
    # ══════════════════════════════════════════════════════════════════
    def get_rbi_repo_rate(self) -> float:
        """
        Get current RBI repo rate.
        As of Feb 2025, repo rate is 6.50% (after 25bps cut from 6.50%).
        This is updated manually or via RBI bulletin scraping.
        The repo rate directly impacts banking sector profitability (NIM)
        and overall market liquidity.
        """
        # Default to latest known rate, updated from config or scraping
        default_rate = 6.25  # As of Feb 2025 (after 7 Feb 2025 MPC cut)
        try:
            url = self.macro_config["rbi"].get("repo_rate_url", "")
            if not url:
                return default_rate
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "lxml")
                # Attempt to find repo rate in RBI bulletin
                text = soup.get_text()
                import re
                match = re.search(r"repo\s*rate.*?(\d+\.?\d*)\s*%", text, re.IGNORECASE)
                if match:
                    return float(match.group(1))
            return default_rate
        except Exception:
            return default_rate

    # ══════════════════════════════════════════════════════════════════
    # UTILITY
    # ══════════════════════════════════════════════════════════════════
    @staticmethod
    def _parse_cr_value(text: str) -> Optional[float]:
        """Parse ₹ Crore values from text (handling commas, negatives)."""
        if not text or text.strip() in ("", "-", "N/A"):
            return None
        cleaned = text.replace(",", "").replace("₹", "").replace("(", "-").replace(")", "").strip()
        try:
            return float(cleaned)
        except (ValueError, TypeError):
            return None
