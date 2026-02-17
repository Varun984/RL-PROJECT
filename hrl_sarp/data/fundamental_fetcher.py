"""
File: fundamental_fetcher.py
Module: data
Description: Fetches fundamental financial data for Indian equities from Screener.in
             (web scraping) and Trendlyne API. Retrieves P/E, P/B, ROE, ROCE, D/E,
             EV/EBITDA, promoter pledge data, and earnings growth metrics.
Design Decisions:
    - Screener.in is primary (most comprehensive free fundamental data for India).
    - Rate limiting (2s) respects server limits. Retry with exponential backoff.
References:
    - Screener.in: https://www.screener.in
    - Trendlyne: https://trendlyne.com
Author: HRL-SARP Framework
"""

import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import yaml
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class FundamentalDataError(Exception):
    """Raised when fundamental data fetch fails."""
    pass


class FundamentalFetcher:
    """
    Fetches fundamental financial data for Indian equities.

    Primary: Screener.in scraping. Fallback: Trendlyne API.
    Retrieves valuation (P/E, P/B, EV/EBITDA), profitability (ROE, ROCE),
    leverage (D/E), governance (promoter pledge), and growth metrics.
    """

    def __init__(self, config_path: str = "config/data_config.yaml") -> None:
        with open(config_path, "r") as f:
            self.config: Dict[str, Any] = yaml.safe_load(f)
        self.fund_config = self.config["fundamentals"]
        self.rate_limit: float = self.fund_config["screener"].get("rate_limit_seconds", 2.0)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml",
        })
        logger.info("FundamentalFetcher initialised | rate_limit=%.1fs", self.rate_limit)

    # ══════════════════════════════════════════════════════════════════
    # SCREENER.IN SCRAPING
    # ══════════════════════════════════════════════════════════════════
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=15))
    def fetch_company_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch fundamental metrics for a single company from Screener.in.
        Extracts P/E, P/B, ROE, ROCE, D/E, EV/EBITDA, pledge data, earnings.

        Args:
            symbol: NSE stock symbol (e.g., "RELIANCE").
        Returns:
            Dict with fundamental metrics and fetch_timestamp.
        Raises:
            FundamentalDataError: If scraping fails.
        """
        base_url = self.fund_config["screener"]["base_url"]
        url = f"{base_url}/company/{symbol}/consolidated/"
        try:
            response = self.session.get(url, timeout=15)
            if response.status_code == 404:
                url = f"{base_url}/company/{symbol}/"
                response = self.session.get(url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "lxml")
            fundamentals: Dict[str, Any] = {
                "symbol": symbol, "fetch_timestamp": datetime.now().isoformat(),
            }
            fundamentals.update(self._parse_key_ratios(soup))
            fundamentals.update(self._parse_quarterly_results(soup))
            fundamentals.update(self._parse_shareholding(soup))
            logger.info("Fetched fundamentals: %s | P/E=%.2f", symbol, fundamentals.get("pe_ratio", 0.0))
            time.sleep(self.rate_limit)
            return fundamentals
        except requests.exceptions.HTTPError as e:
            raise FundamentalDataError(f"HTTP error for {symbol}: {e}")
        except Exception as e:
            raise FundamentalDataError(f"Parse failed for {symbol}: {e}")

    def _parse_key_ratios(self, soup: BeautifulSoup) -> Dict[str, float]:
        """Parse key ratios from Screener.in company page HTML."""
        ratios: Dict[str, float] = {}
        label_map = {
            "Stock P/E": "pe_ratio", "Price to book value": "pb_ratio",
            "Dividend Yield": "dividend_yield", "ROCE": "roce", "ROE": "roe",
            "Market Cap": "market_cap_cr", "Debt to equity": "debt_to_equity",
            "EPS": "eps", "Promoter holding": "promoter_holding_pct",
            "Book Value": "book_value", "Current Price": "current_price",
        }
        ratio_section = soup.find("div", id="top-ratios")
        elements = ratio_section.find_all("li") if ratio_section else soup.find_all("li", class_="flex")
        for el in elements:
            try:
                name_el = el.find("span", class_="name")
                value_el = el.find("span", class_="number")
                if not name_el or not value_el:
                    continue
                label = name_el.get_text(strip=True)
                if label in label_map:
                    val = self._parse_numeric(value_el.get_text(strip=True))
                    if val is not None:
                        ratios[label_map[label]] = val
            except Exception:
                continue
        return ratios

    def _parse_quarterly_results(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Parse quarterly results table for revenue/profit and YoY growth."""
        results: Dict[str, Any] = {}
        try:
            section = soup.find("section", id="quarters")
            if not section:
                return results
            table = section.find("table")
            if not table or not table.find("tbody"):
                return results
            for row in table.find("tbody").find_all("tr"):
                cells = row.find_all("td")
                if len(cells) < 2:
                    continue
                label = cells[0].get_text(strip=True).lower()
                latest = self._parse_numeric(cells[-1].get_text(strip=True))
                if "sales" in label or "revenue" in label:
                    results["revenue_latest_qtr"] = latest
                elif "net profit" in label:
                    results["net_profit_latest_qtr"] = latest
                # YoY growth: compare latest vs 4 quarters ago
                if len(cells) >= 6:
                    yoy_val = self._parse_numeric(cells[-5].get_text(strip=True))
                    if latest and yoy_val and yoy_val != 0:
                        growth = (latest - yoy_val) / abs(yoy_val) * 100
                        if "sales" in label or "revenue" in label:
                            results["revenue_growth_yoy"] = growth
                        elif "net profit" in label:
                            results["earnings_growth_yoy"] = growth
        except Exception as e:
            logger.warning("Quarterly results parse error: %s", e)
        return results

    def _parse_shareholding(self, soup: BeautifulSoup) -> Dict[str, float]:
        """Parse shareholding pattern: promoter, FII, DII, pledge data."""
        sh: Dict[str, float] = {}
        try:
            section = soup.find("section", id="shareholding")
            if not section:
                return sh
            table = section.find("table")
            if not table or not table.find("tbody"):
                return sh
            for row in table.find("tbody").find_all("tr"):
                cells = row.find_all("td")
                if len(cells) < 2:
                    continue
                label = cells[0].get_text(strip=True).lower()
                val = self._parse_numeric(cells[-1].get_text(strip=True))
                if "promoter" in label and "pledge" not in label:
                    sh["promoter_holding_pct"] = val or 0.0
                elif "pledge" in label:
                    sh["promoter_pledge_pct"] = val or 0.0
                elif "fii" in label or "foreign" in label:
                    sh["fii_holding_pct"] = val or 0.0
                elif "dii" in label or "domestic" in label:
                    sh["dii_holding_pct"] = val or 0.0
        except Exception as e:
            logger.warning("Shareholding parse error: %s", e)
        return sh

    # ══════════════════════════════════════════════════════════════════
    # BATCH FETCHING
    # ══════════════════════════════════════════════════════════════════
    def fetch_fundamentals_batch(self, symbols: List[str]) -> pd.DataFrame:
        """Fetch fundamentals for multiple symbols, returning a DataFrame."""
        all_data: List[Dict] = []
        for i, symbol in enumerate(symbols):
            try:
                data = self.fetch_company_fundamentals(symbol)
                all_data.append(data)
                if (i + 1) % 10 == 0:
                    logger.info("Fundamentals progress: %d/%d", i + 1, len(symbols))
            except FundamentalDataError as e:
                logger.warning("Failed: %s — %s", symbol, e)
        if not all_data:
            return pd.DataFrame()
        df = pd.DataFrame(all_data).set_index("symbol")
        return df

    # ══════════════════════════════════════════════════════════════════
    # TRENDLYNE API FALLBACK
    # ══════════════════════════════════════════════════════════════════
    def fetch_from_trendlyne(self, symbol: str) -> Dict[str, Any]:
        """Fallback: fetch fundamentals from Trendlyne API."""
        api_key = os.environ.get("TRENDLYNE_API_KEY", "")
        if not api_key:
            raise FundamentalDataError("Trendlyne API key not configured.")
        base_url = self.fund_config["trendlyne"]["base_url"]
        url = f"{base_url}/api/v1/stock/{symbol}/fundamentals/"
        headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            data = response.json()
            return {
                "symbol": symbol, "fetch_timestamp": datetime.now().isoformat(),
                "pe_ratio": data.get("pe", 0.0), "pb_ratio": data.get("pb", 0.0),
                "roe": data.get("roe", 0.0), "roce": data.get("roce", 0.0),
                "debt_to_equity": data.get("de_ratio", 0.0),
                "ev_ebitda": data.get("ev_ebitda", 0.0),
                "promoter_pledge_pct": data.get("pledge_pct", 0.0),
            }
        except Exception as e:
            raise FundamentalDataError(f"Trendlyne failed for {symbol}: {e}")

    # ══════════════════════════════════════════════════════════════════
    # DERIVED METRICS
    # ══════════════════════════════════════════════════════════════════
    @staticmethod
    def compute_ev_ebitda(market_cap: float, total_debt: float, cash: float, ebitda: float) -> float:
        """
        Compute EV/EBITDA = (Market Cap + Debt - Cash) / EBITDA.
        Preferred for capital-intensive sectors (Metals, Energy, Infra).
        Indian context: EV/EBITDA < 8 is cheap; > 20 indicates growth premium.
        """
        if ebitda <= 0:
            return 0.0
        return (market_cap + total_debt - cash) / ebitda

    @staticmethod
    def compute_earnings_surprise(actual_eps: float, consensus_eps: float) -> float:
        """
        Earnings Surprise % = (Actual - Consensus) / |Consensus| * 100.
        India: >+10% surprise drives 3-5 day positive momentum; <-10% causes sharp correction.
        """
        if consensus_eps == 0:
            return 0.0
        return ((actual_eps - consensus_eps) / abs(consensus_eps)) * 100.0

    @staticmethod
    def _parse_numeric(text: str) -> Optional[float]:
        """Parse numeric from text handling ₹, commas, %, Cr suffixes."""
        if not text or text.strip() in ("", "-", "N/A", "NA", "—"):
            return None
        cleaned = text.replace("₹", "").replace(",", "").replace("%", "")
        cleaned = cleaned.replace("Cr.", "").replace("Cr", "").strip()
        try:
            return float(cleaned)
        except (ValueError, TypeError):
            return None
