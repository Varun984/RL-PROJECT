"""
File: india_calendar.py
Module: environment
Description: IndiaCalendar tracks RBI MPC dates, Union Budget, NSE expiry cycles,
    results season, state elections, and NSE holidays. Returns event_risk_flag
    and days_to_next_event for the risk manager and macro state vector.
Design Decisions: All dates loaded from risk_config.yaml; F&O expiry and holidays
    are computed algorithmically. Caches results per date for fast environment stepping.
References: NSE circular on trading holidays, SEBI F&O expiry rules
Author: HRL-SARP Framework
"""

import logging
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════
# NSE HOLIDAYS (gazetted + exchange-declared, refreshed annually)
# ══════════════════════════════════════════════════════════════════════

# 2024–2025 NSE holidays (update yearly)
NSE_HOLIDAYS_2024 = [
    "2024-01-26", "2024-03-08", "2024-03-25", "2024-03-29",
    "2024-04-11", "2024-04-14", "2024-04-17", "2024-04-21",
    "2024-05-01", "2024-05-23", "2024-06-17", "2024-07-17",
    "2024-08-15", "2024-10-02", "2024-10-12", "2024-10-31",
    "2024-11-01", "2024-11-15", "2024-12-25",
]

NSE_HOLIDAYS_2025 = [
    "2025-01-26", "2025-02-26", "2025-03-14", "2025-03-31",
    "2025-04-10", "2025-04-14", "2025-04-18", "2025-05-01",
    "2025-08-15", "2025-08-27", "2025-10-02", "2025-10-20",
    "2025-10-21", "2025-10-22", "2025-11-05", "2025-11-26",
    "2025-12-25",
]


# ══════════════════════════════════════════════════════════════════════
# INDIA CALENDAR CLASS
# ══════════════════════════════════════════════════════════════════════


class IndiaCalendar:
    """Tracks Indian market events and trading calendar."""

    def __init__(self, risk_config_path: str = "config/risk_config.yaml") -> None:
        with open(risk_config_path, "r") as f:
            self.risk_cfg = yaml.safe_load(f)

        self.event_categories = self.risk_cfg["events"]["categories"]
        self.lookforward_days: int = self.risk_cfg["events"]["event_lookforward_days"]
        self.event_reduction: float = self.risk_cfg["events"]["event_risk_reduction"]

        # Parse all fixed event dates
        self.event_dates: List[Tuple[date, str, str]] = []
        self._parse_event_dates()

        # Build holiday set
        self.holidays = set()
        for d_str in NSE_HOLIDAYS_2024 + NSE_HOLIDAYS_2025:
            self.holidays.add(datetime.strptime(d_str, "%Y-%m-%d").date())

        # Cache for computed expiry dates
        self._expiry_cache: Dict[int, List[date]] = {}

        logger.info(
            "IndiaCalendar initialised | %d fixed events | %d holidays",
            len(self.event_dates),
            len(self.holidays),
        )

    def _parse_event_dates(self) -> None:
        """Parse event dates from config into structured list."""
        for category_name, category_data in self.event_categories.items():
            risk_level = category_data.get("risk_level", "medium")
            if "dates" in category_data:
                for d_str in category_data["dates"]:
                    dt = datetime.strptime(str(d_str), "%Y-%m-%d").date()
                    self.event_dates.append((dt, category_name, risk_level))
            if "periods" in category_data:
                for period in category_data["periods"]:
                    start = datetime.strptime(str(period["start"]), "%Y-%m-%d").date()
                    end = datetime.strptime(str(period["end"]), "%Y-%m-%d").date()
                    # Add start date as the event marker
                    self.event_dates.append((start, category_name, risk_level))
        # Sort chronologically
        self.event_dates.sort(key=lambda x: x[0])

    # ── F&O Expiry Computation ───────────────────────────────────────

    def get_monthly_expiry(self, year: int, month: int) -> date:
        """Last Thursday of the given month (NSE F&O monthly expiry).

        If last Thursday is a holiday, expiry shifts to previous trading day.
        """
        # Find last day of month
        if month == 12:
            last_day = date(year + 1, 1, 1) - timedelta(days=1)
        else:
            last_day = date(year, month + 1, 1) - timedelta(days=1)

        # Walk backward to find last Thursday (weekday 3)
        while last_day.weekday() != 3:
            last_day -= timedelta(days=1)

        # If it's a holiday, shift to previous trading day
        while last_day in self.holidays or last_day.weekday() >= 5:
            last_day -= timedelta(days=1)

        return last_day

    def get_expiries_for_year(self, year: int) -> List[date]:
        """Get all monthly F&O expiry dates for a year."""
        if year in self._expiry_cache:
            return self._expiry_cache[year]
        expiries = [self.get_monthly_expiry(year, m) for m in range(1, 13)]
        self._expiry_cache[year] = expiries
        return expiries

    # ── Event Risk Queries ───────────────────────────────────────────

    def get_event_risk(self, query_date: date) -> Dict[str, any]:
        """Get event risk information for a given date.

        Returns dict with:
            event_risk_flag (bool): True if an event is within lookforward window.
            days_to_next_event (int): Days until next event (-1 if none upcoming).
            next_event_name (str): Name of the next event.
            next_event_risk_level (str): Risk level of next event.
            is_results_season (bool): Whether within quarterly results period.
            days_to_expiry (int): Days until next F&O expiry.
            size_reduction (float): Recommended position size reduction factor.
        """
        if isinstance(query_date, str):
            query_date = datetime.strptime(query_date, "%Y-%m-%d").date()

        result = {
            "event_risk_flag": False,
            "days_to_next_event": -1,
            "next_event_name": "",
            "next_event_risk_level": "none",
            "is_results_season": False,
            "days_to_expiry": -1,
            "size_reduction": 1.0,
        }

        # Check fixed events
        min_days = float("inf")
        for evt_date, evt_name, evt_risk in self.event_dates:
            delta = (evt_date - query_date).days
            if 0 <= delta <= self.lookforward_days:
                result["event_risk_flag"] = True
                if delta < min_days:
                    min_days = delta
                    result["days_to_next_event"] = delta
                    result["next_event_name"] = evt_name
                    result["next_event_risk_level"] = evt_risk
            elif delta > self.lookforward_days:
                if result["days_to_next_event"] == -1:
                    result["days_to_next_event"] = delta
                    result["next_event_name"] = evt_name
                    result["next_event_risk_level"] = evt_risk
                break

        # Check F&O expiry
        year = query_date.year
        expiries = self.get_expiries_for_year(year)
        for exp in expiries:
            delta = (exp - query_date).days
            if 0 <= delta:
                result["days_to_expiry"] = delta
                if delta <= self.lookforward_days:
                    result["event_risk_flag"] = True
                break
        # If no expiry found in current year, check next year
        if result["days_to_expiry"] == -1:
            next_expiries = self.get_expiries_for_year(year + 1)
            if next_expiries:
                result["days_to_expiry"] = (next_expiries[0] - query_date).days

        # Check results season
        result["is_results_season"] = self._is_results_season(query_date)

        # Compute size reduction
        if result["event_risk_flag"]:
            if result["next_event_risk_level"] == "critical":
                result["size_reduction"] = self.event_reduction
            elif result["next_event_risk_level"] == "high":
                result["size_reduction"] = 1.0 - (1.0 - self.event_reduction) * 0.7
            else:
                result["size_reduction"] = 1.0 - (1.0 - self.event_reduction) * 0.4

        return result

    def _is_results_season(self, query_date: date) -> bool:
        """Check if date falls within a quarterly results season period."""
        if "results_season" in self.event_categories:
            periods = self.event_categories["results_season"].get("periods", [])
            for period in periods:
                start = datetime.strptime(str(period["start"]), "%Y-%m-%d").date()
                end = datetime.strptime(str(period["end"]), "%Y-%m-%d").date()
                if start <= query_date <= end:
                    return True
        return False

    # ── Trading Day Utilities ────────────────────────────────────────

    def is_trading_day(self, query_date: date) -> bool:
        """Check if a date is an NSE trading day (not weekend, not holiday)."""
        if query_date.weekday() >= 5:
            return False
        return query_date not in self.holidays

    def next_trading_day(self, query_date: date) -> date:
        """Find the next trading day after query_date."""
        d = query_date + timedelta(days=1)
        while not self.is_trading_day(d):
            d += timedelta(days=1)
        return d

    def prev_trading_day(self, query_date: date) -> date:
        """Find the previous trading day before query_date."""
        d = query_date - timedelta(days=1)
        while not self.is_trading_day(d):
            d -= timedelta(days=1)
        return d

    def trading_days_between(self, start: date, end: date) -> int:
        """Count trading days between two dates (exclusive of start, inclusive of end)."""
        count = 0
        d = start + timedelta(days=1)
        while d <= end:
            if self.is_trading_day(d):
                count += 1
            d += timedelta(days=1)
        return count

    def get_trading_dates(self, start: date, end: date) -> List[date]:
        """Return list of all trading dates in [start, end] range."""
        dates = []
        d = start
        while d <= end:
            if self.is_trading_day(d):
                dates.append(d)
            d += timedelta(days=1)
        return dates

    def is_week_end(self, query_date: date) -> bool:
        """Check if this is the last trading day of the week (for Macro agent stepping)."""
        next_td = self.next_trading_day(query_date)
        # If next trading day is in a different week
        return next_td.isocalendar()[1] != query_date.isocalendar()[1]
