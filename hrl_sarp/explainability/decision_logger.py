"""
File: decision_logger.py
Module: explainability
Description: Structured logging of every agent decision for post-hoc analysis.
    Records the full decision context: input state, attention weights, action
    probabilities, risk alerts, and portfolio impact at each step.
Design Decisions: Uses JSON-lines format for streaming writes (append-friendly).
    Each log entry is self-contained for independent analysis. Provides
    natural language summaries of decisions via template-based generation.
References: Interpretable RL survey (Puiutta & Veith 2020)
Author: HRL-SARP Framework
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class DecisionLogger:
    """Log and explain agent decisions with full context."""

    SECTOR_NAMES = [
        "IT", "Financials", "Pharma", "FMCG", "Auto",
        "Energy", "Metals", "Realty", "Telecom", "Media", "Infra",
    ]

    REGIME_NAMES = ["Bull", "Bear", "Sideways"]

    def __init__(
        self,
        log_dir: str = "logs/decisions",
        max_buffer: int = 1000,
    ) -> None:
        self.log_dir = log_dir
        self.max_buffer = max_buffer
        os.makedirs(log_dir, exist_ok=True)

        self.log_file = os.path.join(log_dir, "decisions.jsonl")
        self.buffer: List[Dict[str, Any]] = []
        self.step_count: int = 0

    # ── Logging ──────────────────────────────────────────────────────

    def log_macro_decision(
        self,
        step: int,
        date: Optional[str] = None,
        macro_state: Optional[np.ndarray] = None,
        sector_weights: Optional[np.ndarray] = None,
        regime_probs: Optional[np.ndarray] = None,
        regime_selected: Optional[int] = None,
        value_estimate: Optional[float] = None,
        attention_scores: Optional[np.ndarray] = None,
        risk_report: Optional[Dict] = None,
        portfolio_value: Optional[float] = None,
        reward: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Log a Macro agent decision with full context."""
        entry = {
            "type": "macro",
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "date": date,
        }

        if macro_state is not None:
            entry["macro_state"] = macro_state.tolist()

        if sector_weights is not None:
            entry["sector_weights"] = {
                self.SECTOR_NAMES[i]: float(sector_weights[i])
                for i in range(min(len(sector_weights), len(self.SECTOR_NAMES)))
            }
            entry["top_sectors"] = self._top_k_sectors(sector_weights, k=3)

        if regime_probs is not None:
            entry["regime_probs"] = {
                self.REGIME_NAMES[i]: float(regime_probs[i])
                for i in range(min(len(regime_probs), len(self.REGIME_NAMES)))
            }

        if regime_selected is not None:
            entry["regime_selected"] = self.REGIME_NAMES[regime_selected] if regime_selected < len(self.REGIME_NAMES) else str(regime_selected)

        if value_estimate is not None:
            entry["value_estimate"] = float(value_estimate)

        if attention_scores is not None:
            entry["attention_top_sectors"] = self._top_k_sectors(attention_scores, k=3)

        if risk_report is not None:
            entry["risk_alerts"] = risk_report.get("alerts", [])
            entry["risk_halted"] = risk_report.get("is_halted", False)

        if portfolio_value is not None:
            entry["portfolio_value"] = float(portfolio_value)

        if reward is not None:
            entry["reward"] = float(reward)

        # Generate natural language explanation
        entry["explanation"] = self._explain_macro(entry)

        self._write(entry)
        return entry

    def log_micro_decision(
        self,
        step: int,
        date: Optional[str] = None,
        stock_weights: Optional[np.ndarray] = None,
        stock_names: Optional[List[str]] = None,
        goal: Optional[np.ndarray] = None,
        goal_alignment: Optional[float] = None,
        exploration_noise: Optional[float] = None,
        portfolio_value: Optional[float] = None,
        reward: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Log a Micro agent decision with full context."""
        entry = {
            "type": "micro",
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "date": date,
        }

        if stock_weights is not None:
            n_stocks = len(stock_weights)
            names = stock_names if stock_names else [f"Stock_{i}" for i in range(n_stocks)]
            top_idx = np.argsort(-stock_weights)[:5]
            entry["top_stocks"] = [
                {"name": names[i], "weight": float(stock_weights[i])}
                for i in top_idx if stock_weights[i] > 0.001
            ]
            entry["n_positions"] = int((stock_weights > 0.001).sum())

        if goal is not None:
            entry["goal_sector_weights"] = goal[:len(self.SECTOR_NAMES)].tolist() if len(goal) >= len(self.SECTOR_NAMES) else goal.tolist()

        if goal_alignment is not None:
            entry["goal_alignment"] = float(goal_alignment)

        if exploration_noise is not None:
            entry["exploration_noise"] = float(exploration_noise)

        if portfolio_value is not None:
            entry["portfolio_value"] = float(portfolio_value)

        if reward is not None:
            entry["reward"] = float(reward)

        entry["explanation"] = self._explain_micro(entry)

        self._write(entry)
        return entry

    # ── Natural Language Explanations ────────────────────────────────

    def _explain_macro(self, entry: Dict[str, Any]) -> str:
        """Generate NL explanation for a macro decision."""
        parts = []

        if "regime_selected" in entry:
            parts.append(f"Detected {entry['regime_selected']} market regime.")

        if "top_sectors" in entry:
            tops = entry["top_sectors"]
            if tops:
                sector_str = ", ".join(
                    f"{s['name']} ({s['weight']:.1%})" for s in tops[:3]
                )
                parts.append(f"Top sector allocations: {sector_str}.")

        if "risk_halted" in entry and entry["risk_halted"]:
            parts.append("⚠️ Portfolio HALTED due to risk breach — moving to cash.")

        if "risk_alerts" in entry and entry["risk_alerts"]:
            n_alerts = len(entry["risk_alerts"])
            parts.append(f"{n_alerts} risk alert(s) triggered.")

        if "reward" in entry:
            r = entry["reward"]
            if r > 0:
                parts.append(f"Positive reward: {r:.4f}.")
            else:
                parts.append(f"Negative reward: {r:.4f}.")

        return " ".join(parts) if parts else "Standard allocation step."

    def _explain_micro(self, entry: Dict[str, Any]) -> str:
        """Generate NL explanation for a micro decision."""
        parts = []

        if "n_positions" in entry:
            parts.append(f"Holding {entry['n_positions']} positions.")

        if "top_stocks" in entry:
            tops = entry["top_stocks"][:3]
            if tops:
                stock_str = ", ".join(
                    f"{s['name']} ({s['weight']:.1%})" for s in tops
                )
                parts.append(f"Largest positions: {stock_str}.")

        if "goal_alignment" in entry:
            ga = entry["goal_alignment"]
            if ga > 0.9:
                parts.append("Closely aligned with Macro goal.")
            elif ga > 0.7:
                parts.append("Moderately aligned with Macro goal.")
            else:
                parts.append("Deviating from Macro goal — exploring alternatives.")

        if "exploration_noise" in entry:
            noise = entry["exploration_noise"]
            if noise > 0.2:
                parts.append(f"High exploration noise ({noise:.3f}).")

        return " ".join(parts) if parts else "Standard stock allocation step."

    # ── Helpers ──────────────────────────────────────────────────────

    def _top_k_sectors(self, weights: np.ndarray, k: int = 3) -> List[Dict]:
        n = min(len(weights), len(self.SECTOR_NAMES))
        sorted_idx = np.argsort(-weights[:n])[:k]
        return [
            {"name": self.SECTOR_NAMES[i], "weight": float(weights[i])}
            for i in sorted_idx
        ]

    def _write(self, entry: Dict[str, Any]) -> None:
        """Write entry to buffer and flush to disk periodically."""
        self.buffer.append(entry)
        self.step_count += 1

        if len(self.buffer) >= self.max_buffer:
            self.flush()

    def flush(self) -> None:
        """Flush buffer to JSON-lines file."""
        if not self.buffer:
            return

        with open(self.log_file, "a") as f:
            for entry in self.buffer:
                f.write(json.dumps(entry, default=str) + "\n")

        logger.debug("Flushed %d decision logs", len(self.buffer))
        self.buffer.clear()

    def get_summary(self, last_n: int = 10) -> List[Dict[str, str]]:
        """Get explanations for last N decisions."""
        # Read from buffer first, then file
        entries = self.buffer[-last_n:]
        return [
            {"step": e.get("step"), "type": e.get("type"), "explanation": e.get("explanation")}
            for e in entries
        ]

    def close(self) -> None:
        """Flush remaining buffer on close."""
        self.flush()
