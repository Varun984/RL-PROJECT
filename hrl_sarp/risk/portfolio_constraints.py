"""
File: portfolio_constraints.py
Module: risk
Description: Enforcement layer for portfolio construction constraints. Provides
    projection operators that map unconstrained agent outputs into feasible
    portfolio allocations satisfying SEBI limits, liquidity filters, and
    governance requirements.
Design Decisions: Constraint enforcement is separated from risk monitoring to allow
    the environment to apply constraints pre-step (hard constraints) while risk
    monitoring runs post-step (soft alerts). Projection is differentiable where
    possible for compatibility with policy gradients.
References: Boyd & Vandenberghe convex optimisation, SEBI MF regulations
Author: HRL-SARP Framework
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

logger = logging.getLogger(__name__)


class PortfolioConstraints:
    """Enforce hard portfolio construction constraints.

    Provides projection operators:
        project_sector_weights() → feasible sector allocation
        project_stock_weights() → feasible stock allocation
        filter_stock_universe() → eligible stocks
    """

    def __init__(
        self,
        config_path: str = "config/risk_config.yaml",
    ) -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        # Constraints from config
        self.max_sector_pct: float = self.cfg["sector"]["max_single_sector_pct"]
        self.min_sector_pct: float = self.cfg["sector"]["min_single_sector_pct"]
        self.max_stock_pct: float = self.cfg["stock"]["max_single_stock_pct"]
        self.min_stock_pct: float = self.cfg["stock"]["min_single_stock_pct"]
        self.max_positions: int = self.cfg["portfolio"]["max_positions"]
        self.min_positions: int = self.cfg["portfolio"]["min_positions"]
        self.min_cash: float = self.cfg["portfolio"]["min_cash_pct"]

        # Liquidity filters
        liq_cfg = self.cfg["liquidity"]
        self.min_adtv: float = liq_cfg["min_daily_volume_cr"]
        self.min_delivery: float = liq_cfg["min_delivery_pct"]
        self.max_adv_participation: float = liq_cfg["max_adv_participation"]

        # Governance filters
        gov_cfg = self.cfg["governance"]
        self.max_promoter_pledge: float = gov_cfg["max_promoter_pledge_pct"]
        self.min_free_float: float = gov_cfg["min_free_float_pct"]
        self.exclude_asm: bool = gov_cfg["exclude_asm_stocks"]
        self.exclude_gsm: bool = gov_cfg["exclude_gsm_stocks"]
        self.min_listing_days: int = gov_cfg["min_listing_days"]

        logger.info("PortfolioConstraints initialised")

    # ── Sector Weight Projection ─────────────────────────────────────

    def project_sector_weights(
        self,
        raw_weights: np.ndarray,
        min_active: int = 3,
    ) -> np.ndarray:
        """Project raw sector weights into feasible region.

        Constraints:
            - Each sector weight in [0, max_sector_pct]
            - Non-zero weights ≥ min_sector_pct
            - Sum ≤ (1 - min_cash)
            - At least min_active sectors have non-zero weight
        """
        weights = raw_weights.copy()

        # Ensure non-negative
        weights = np.clip(weights, 0.0, None)

        # Cap individual sectors
        weights = np.clip(weights, 0.0, self.max_sector_pct)

        # Remove dust (below minimum threshold)
        dust_mask = (weights > 0) & (weights < self.min_sector_pct)
        weights[dust_mask] = 0.0

        # Ensure minimum active sectors
        active_count = (weights > 0).sum()
        if active_count < min_active:
            # Add weight to largest zero-weight sectors
            zero_idx = np.where(weights == 0)[0]
            # Use raw weights to decide which zero sectors to activate
            if len(zero_idx) > 0:
                raw_order = np.argsort(-raw_weights[zero_idx])
                n_activate = min(min_active - active_count, len(zero_idx))
                for i in range(int(n_activate)):
                    weights[zero_idx[raw_order[i]]] = self.min_sector_pct

        # Normalise to sum ≤ budget
        budget = 1.0 - self.min_cash
        total = weights.sum()
        if total > budget:
            weights *= (budget / total)
        elif total < 1e-8:
            # Fallback: equal weight across all sectors
            weights = np.full_like(weights, budget / len(weights))

        return weights

    # ── Stock Weight Projection ──────────────────────────────────────

    def project_stock_weights(
        self,
        raw_weights: np.ndarray,
        sector_targets: Optional[np.ndarray] = None,
        stock_to_sector: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Project raw stock weights into feasible region.

        Constraints:
            - Each stock in [0, max_stock_pct]
            - Non-zero weights ≥ min_stock_pct
            - Max positions ≤ max_positions
            - If sector_targets provided, sector exposures approximately match
        """
        weights = raw_weights.copy()
        weights = np.clip(weights, 0.0, None)

        # Step 1: Cap individual stocks
        weights = np.clip(weights, 0.0, self.max_stock_pct)

        # Step 2: Remove dust
        dust_mask = (weights > 0) & (weights < self.min_stock_pct)
        weights[dust_mask] = 0.0

        # Step 3: Enforce max positions
        n_active = (weights > 0).sum()
        if n_active > self.max_positions:
            sorted_idx = np.argsort(weights)
            n_remove = n_active - self.max_positions
            for i in range(int(n_remove)):
                weights[sorted_idx[i]] = 0.0

        # Step 4: Sector alignment (if targets given)
        if sector_targets is not None and stock_to_sector is not None:
            weights = self._align_to_sector_targets(
                weights, sector_targets, stock_to_sector
            )

        # Step 5: Normalise
        budget = 1.0 - self.min_cash
        total = weights.sum()
        if total > budget:
            weights *= (budget / total)
        elif total > 0:
            # Scale up to use available budget
            weights *= (budget / total)

        return weights

    def _align_to_sector_targets(
        self,
        weights: np.ndarray,
        sector_targets: np.ndarray,
        stock_to_sector: np.ndarray,
    ) -> np.ndarray:
        """Iteratively adjust stock weights to match sector targets."""
        n_sectors = len(sector_targets)

        for iteration in range(5):  # Max 5 iterations
            # Compute current sector exposure
            sector_exposure = np.zeros(n_sectors)
            for i, s in enumerate(stock_to_sector):
                if i < len(weights):
                    sector_exposure[s] += weights[i]

            # Scale stocks within each sector
            for s in range(n_sectors):
                if sector_targets[s] < 1e-8:
                    # Zero target: remove all stocks in this sector
                    for i, sec in enumerate(stock_to_sector):
                        if sec == s and i < len(weights):
                            weights[i] = 0.0
                elif sector_exposure[s] > 1e-8:
                    scale = sector_targets[s] / sector_exposure[s]
                    for i, sec in enumerate(stock_to_sector):
                        if sec == s and i < len(weights):
                            weights[i] *= scale

            # Re-enforce caps after scaling
            weights = np.clip(weights, 0.0, self.max_stock_pct)

        return weights

    # ── Universe Filtering ───────────────────────────────────────────

    def filter_stock_universe(
        self,
        stock_data: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Filter stocks based on liquidity and governance criteria.

        Args:
            stock_data: Dict with keys:
                'adtv_cr': (N,) average daily traded value in crores
                'delivery_pct': (N,) delivery percentage
                'promoter_pledge_pct': (N,) promoter pledge ratio
                'free_float_pct': (N,) free float percentage
                'listing_days': (N,) days since listing
                'is_asm': (N,) bool array for ASM stocks
                'is_gsm': (N,) bool array for GSM stocks

        Returns:
            Boolean mask (N,) where True = eligible.
        """
        n = len(stock_data.get("adtv_cr", []))
        mask = np.ones(n, dtype=bool)

        # Liquidity
        if "adtv_cr" in stock_data:
            mask &= stock_data["adtv_cr"] >= self.min_adtv

        if "delivery_pct" in stock_data:
            mask &= stock_data["delivery_pct"] >= self.min_delivery

        # Governance
        if "promoter_pledge_pct" in stock_data:
            mask &= stock_data["promoter_pledge_pct"] <= self.max_promoter_pledge

        if "free_float_pct" in stock_data:
            mask &= stock_data["free_float_pct"] >= self.min_free_float

        if "listing_days" in stock_data:
            mask &= stock_data["listing_days"] >= self.min_listing_days

        if self.exclude_asm and "is_asm" in stock_data:
            mask &= ~stock_data["is_asm"]

        if self.exclude_gsm and "is_gsm" in stock_data:
            mask &= ~stock_data["is_gsm"]

        logger.info(
            "Universe filter: %d/%d stocks eligible",
            mask.sum(), n,
        )
        return mask

    # ── Turnover Check ───────────────────────────────────────────────

    def compute_turnover(
        self,
        old_weights: np.ndarray,
        new_weights: np.ndarray,
    ) -> float:
        """Compute portfolio turnover (sum of absolute weight changes / 2)."""
        return float(np.sum(np.abs(new_weights - old_weights)) / 2.0)

    def should_rebalance(
        self,
        old_weights: np.ndarray,
        new_weights: np.ndarray,
    ) -> bool:
        """Check if rebalancing is warranted based on drift threshold."""
        max_drift = float(np.max(np.abs(new_weights - old_weights)))
        return max_drift > self.rebalance_threshold
