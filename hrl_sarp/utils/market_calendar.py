"""
File: market_calendar.py
Module: utils
Description: Indian market (NSE/BSE) trading calendar utility.
    Provides trading day checks, holiday calendars, session timing,
    and business day arithmetic for backtesting and scheduling.
Design Decisions: Pre-populated NSE holiday list for 2020–2026 with
    automatic weekend handling. Extensible via YAML override.
References: NSE India official holiday calendar
Author: HRL-SARP Framework
"""

import logging
from datetime import date, datetime, timedelta
from typing import List, Optional, Set

logger = logging.getLogger(__name__)


class NSECalendar:
    """NSE India trading calendar with holiday management."""

    # NSE fixed holidays (approximate — combined for 2020–2026)
    FIXED_HOLIDAYS = {
        # Republic Day
        (1, 26),
        # Holi
        (3, 14), (3, 25),
        # Good Friday (varies, approximate)
        (3, 29), (4, 7), (4, 14), (4, 18),
        # Dr Ambedkar Jayanti
        (4, 14),
        # May Day
        (5, 1),
        # Independence Day
        (8, 15),
        # Mahatma Gandhi Jayanti
        (10, 2),
        # Diwali / Laxmi Puja (varies)
        (10, 24), (11, 1), (11, 12), (11, 13), (11, 14),
        # Guru Nanak Jayanti (varies)
        (11, 15), (11, 27),
        # Christmas
        (12, 25),
    }

    # Full trading session
    MARKET_OPEN = (9, 15)   # 09:15 IST
    MARKET_CLOSE = (15, 30) # 15:30 IST

    # Pre-open session
    PRE_OPEN_START = (9, 0)
    PRE_OPEN_END = (9, 8)

    def __init__(
        self,
        extra_holidays: Optional[List[date]] = None,
    ) -> None:
        self._extra_holidays: Set[date] = set()
        if extra_holidays:
            self._extra_holidays = set(extra_holidays)

        logger.info(
            "NSECalendar initialised | %d extra holidays",
            len(self._extra_holidays),
        )

    def is_trading_day(self, d: date) -> bool:
        """Check if a given date is a trading day."""
        # Weekend
        if d.weekday() >= 5:
            return False

        # Extra holidays (exact dates)
        if d in self._extra_holidays:
            return False

        # Fixed holidays (month, day) — approximate
        if (d.month, d.day) in self.FIXED_HOLIDAYS:
            return False

        return True

    def next_trading_day(self, d: date) -> date:
        """Get the next trading day after the given date."""
        current = d + timedelta(days=1)
        while not self.is_trading_day(current):
            current += timedelta(days=1)
        return current

    def prev_trading_day(self, d: date) -> date:
        """Get the previous trading day before the given date."""
        current = d - timedelta(days=1)
        while not self.is_trading_day(current):
            current -= timedelta(days=1)
        return current

    def trading_days_between(
        self,
        start: date,
        end: date,
    ) -> List[date]:
        """Get all trading days between start and end (inclusive)."""
        days = []
        current = start
        while current <= end:
            if self.is_trading_day(current):
                days.append(current)
            current += timedelta(days=1)
        return days

    def n_trading_days_between(
        self,
        start: date,
        end: date,
    ) -> int:
        """Count trading days between start and end (inclusive)."""
        return len(self.trading_days_between(start, end))

    def add_trading_days(self, d: date, n: int) -> date:
        """Add n trading days to date d."""
        current = d
        added = 0
        step = 1 if n >= 0 else -1
        target = abs(n)

        while added < target:
            current += timedelta(days=step)
            if self.is_trading_day(current):
                added += 1

        return current

    def is_month_end(self, d: date) -> bool:
        """Check if date is the last trading day of the month."""
        next_day = self.next_trading_day(d)
        return next_day.month != d.month

    def is_week_end(self, d: date) -> bool:
        """Check if date is the last trading day of the week (typically Friday)."""
        next_day = self.next_trading_day(d)
        return next_day.isocalendar()[1] != d.isocalendar()[1]

    def is_expiry_day(self, d: date) -> bool:
        """Check if date is F&O expiry (last Thursday of the month)."""
        if d.weekday() != 3:  # Thursday
            return False

        # Check if next Thursday is in a different month
        next_thursday = d + timedelta(days=7)
        return next_thursday.month != d.month

    def get_weekly_rebalance_dates(
        self,
        start: date,
        end: date,
    ) -> List[date]:
        """Get Friday (or last trading day of week) dates for weekly rebalancing."""
        trading_days = self.trading_days_between(start, end)
        rebalance_dates = []
        for d in trading_days:
            if self.is_week_end(d):
                rebalance_dates.append(d)
        return rebalance_dates

    def get_monthly_rebalance_dates(
        self,
        start: date,
        end: date,
    ) -> List[date]:
        """Get month-end trading dates for monthly rebalancing."""
        trading_days = self.trading_days_between(start, end)
        rebalance_dates = []
        for d in trading_days:
            if self.is_month_end(d):
                rebalance_dates.append(d)
        return rebalance_dates
