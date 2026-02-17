"""
File: shap_explainer.py
Module: explainability
Description: SHAP (SHapley Additive exPlanations) feature importance analysis
    for both Macro and Micro agents. Uses KernelSHAP because the policy networks
    are non-differentiable w.r.t. SHAP values (Dirichlet sampling, masking).
Design Decisions: KernelSHAP is model-agnostic and works with arbitrary PyTorch
    models. We wrap agent inference in a callable for SHAP's API. Feature groups
    enable interpretable sector-level or feature-category-level explanations.
References: Lundberg & Lee (2017) "A Unified Approach to Interpreting Model
    Predictions", SHAP documentation
Author: HRL-SARP Framework
"""

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """SHAP-based feature importance for HRL-SARP agents."""

    MACRO_FEATURE_NAMES = [
        "VIX", "Nifty_Return", "FII_Flow", "DII_Flow",
        "PCR", "INR_USD", "Crude_Oil", "US10Y",
        "Nifty_RSI", "Nifty_MACD", "Breadth", "Volatility",
        "Regime_Bull", "Regime_Bear", "Regime_Sideways",
        "Portfolio_Value", "Current_DD", "Cash_Pct",
    ]

    def __init__(
        self,
        save_dir: str = "logs/shap",
        n_background: int = 100,
        n_samples: int = 200,
    ) -> None:
        self.save_dir = save_dir
        self.n_background = n_background
        self.n_samples = n_samples
        os.makedirs(save_dir, exist_ok=True)

    # ── Macro Agent SHAP ─────────────────────────────────────────────

    @torch.no_grad()
    def explain_macro(
        self,
        macro_agent,
        macro_state: np.ndarray,
        sector_embeddings: np.ndarray,
        background_states: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Compute SHAP values for a Macro agent's sector allocation decision.

        Args:
            macro_agent: Trained MacroAgent.
            macro_state: (18,) current macro state vector.
            sector_embeddings: (11, 64) sector GNN embeddings.
            background_states: (M, 18) background dataset for KernelSHAP.

        Returns:
            Dict with 'shap_values', 'feature_names', 'base_value'.
        """
        try:
            import shap
        except ImportError:
            logger.warning("SHAP not installed; using permutation importance fallback")
            return self._permutation_importance_macro(
                macro_agent, macro_state, sector_embeddings
            )

        # Wrap agent as callable: state → sector weights
        def predict_fn(states: np.ndarray) -> np.ndarray:
            outputs = []
            for s in states:
                state_t = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
                emb_t = torch.tensor(
                    sector_embeddings, dtype=torch.float32
                ).unsqueeze(0)
                sector_w, _, _ = macro_agent.actor(
                    state_t.to(macro_agent.device),
                    emb_t.to(macro_agent.device),
                )
                outputs.append(sector_w.cpu().numpy().flatten())
            return np.array(outputs)

        # Background data
        if background_states is None:
            background_states = macro_state.reshape(1, -1) + np.random.randn(
                self.n_background, len(macro_state)
            ) * 0.1

        explainer = shap.KernelExplainer(predict_fn, background_states[:self.n_background])
        shap_values = explainer.shap_values(macro_state.reshape(1, -1), nsamples=self.n_samples)

        # shap_values is a list of arrays (one per output dimension)
        feature_names = self.MACRO_FEATURE_NAMES[:len(macro_state)]

        result = {
            "shap_values": shap_values,
            "feature_names": feature_names,
            "base_value": explainer.expected_value,
            "input_state": macro_state,
        }

        return result

    # ── Micro Agent SHAP ─────────────────────────────────────────────

    @torch.no_grad()
    def explain_micro(
        self,
        micro_agent,
        stock_features: np.ndarray,
        goal: np.ndarray,
        top_k_stocks: int = 5,
        background_features: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Compute SHAP values for Micro agent's stock selection.

        Explains which stock features drive the weight of each top-k stock.
        """
        try:
            import shap
        except ImportError:
            return self._permutation_importance_micro(
                micro_agent, stock_features, goal
            )

        # Flatten features for SHAP: (max_stocks * feat_dim,)
        flat_features = stock_features.flatten()

        def predict_fn(flat_inputs: np.ndarray) -> np.ndarray:
            outputs = []
            n_stocks, feat_dim = stock_features.shape
            for flat in flat_inputs:
                feats = flat.reshape(n_stocks, feat_dim)
                feats_t = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
                goal_t = torch.tensor(goal, dtype=torch.float32).unsqueeze(0)
                weights = micro_agent.actor(
                    feats_t.to(micro_agent.device),
                    goal_t.to(micro_agent.device),
                )
                # Return top-k stock weights
                w = weights.cpu().numpy().flatten()
                top_idx = np.argsort(-w)[:top_k_stocks]
                outputs.append(w[top_idx])
            return np.array(outputs)

        if background_features is None:
            background_features = flat_features.reshape(1, -1) + np.random.randn(
                self.n_background, len(flat_features)
            ) * 0.05

        explainer = shap.KernelExplainer(
            predict_fn, background_features[:self.n_background]
        )
        shap_values = explainer.shap_values(
            flat_features.reshape(1, -1), nsamples=self.n_samples
        )

        return {
            "shap_values": shap_values,
            "input_features": stock_features,
            "goal": goal,
            "base_value": explainer.expected_value,
        }

    # ── Permutation Importance Fallback ──────────────────────────────

    def _permutation_importance_macro(
        self,
        macro_agent,
        macro_state: np.ndarray,
        sector_embeddings: np.ndarray,
        n_repeats: int = 20,
    ) -> Dict[str, Any]:
        """Fallback: permutation-based feature importance."""
        macro_agent.actor.eval()

        state_t = torch.tensor(macro_state, dtype=torch.float32).unsqueeze(0)
        emb_t = torch.tensor(sector_embeddings, dtype=torch.float32).unsqueeze(0)

        # Baseline prediction
        baseline_w, _, _ = macro_agent.actor(
            state_t.to(macro_agent.device), emb_t.to(macro_agent.device)
        )
        baseline = baseline_w.cpu().numpy().flatten()

        importances = np.zeros(len(macro_state))

        for f in range(len(macro_state)):
            diffs = []
            for _ in range(n_repeats):
                perturbed = macro_state.copy()
                perturbed[f] = np.random.normal(
                    macro_state[f], max(abs(macro_state[f]) * 0.2, 0.01)
                )
                p_t = torch.tensor(perturbed, dtype=torch.float32).unsqueeze(0)
                p_w, _, _ = macro_agent.actor(
                    p_t.to(macro_agent.device), emb_t.to(macro_agent.device)
                )
                diff = np.abs(p_w.cpu().numpy().flatten() - baseline).mean()
                diffs.append(diff)
            importances[f] = float(np.mean(diffs))

        # Normalise
        total = importances.sum()
        if total > 1e-8:
            importances /= total

        feature_names = self.MACRO_FEATURE_NAMES[:len(macro_state)]

        macro_agent.actor.train()
        return {
            "importances": importances,
            "feature_names": feature_names,
            "method": "permutation",
        }

    def _permutation_importance_micro(
        self,
        micro_agent,
        stock_features: np.ndarray,
        goal: np.ndarray,
        n_repeats: int = 10,
    ) -> Dict[str, Any]:
        """Permutation importance for Micro agent at per-feature level."""
        micro_agent.actor.eval()

        n_stocks, feat_dim = stock_features.shape
        feats_t = torch.tensor(stock_features, dtype=torch.float32).unsqueeze(0)
        goal_t = torch.tensor(goal, dtype=torch.float32).unsqueeze(0)

        baseline = micro_agent.actor(
            feats_t.to(micro_agent.device), goal_t.to(micro_agent.device)
        ).cpu().numpy().flatten()

        importances = np.zeros(feat_dim)

        for f in range(feat_dim):
            diffs = []
            for _ in range(n_repeats):
                perturbed = stock_features.copy()
                perturbed[:, f] = np.random.permutation(perturbed[:, f])
                p_t = torch.tensor(perturbed, dtype=torch.float32).unsqueeze(0)
                p_w = micro_agent.actor(
                    p_t.to(micro_agent.device), goal_t.to(micro_agent.device)
                ).cpu().numpy().flatten()
                diff = np.abs(p_w - baseline).mean()
                diffs.append(diff)
            importances[f] = float(np.mean(diffs))

        total = importances.sum()
        if total > 1e-8:
            importances /= total

        micro_agent.actor.train()
        return {
            "importances": importances,
            "method": "permutation",
        }

    # ── Visualisation ────────────────────────────────────────────────

    def plot_macro_importance(
        self,
        result: Dict[str, Any],
        title: str = "Macro Agent Feature Importance",
        filename: str = "macro_shap.png",
    ) -> str:
        """Plot feature importance bar chart."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            if "importances" in result:
                values = result["importances"]
                names = result.get("feature_names", [f"F{i}" for i in range(len(values))])
            elif "shap_values" in result:
                # Average absolute SHAP across outputs
                sv = result["shap_values"]
                if isinstance(sv, list):
                    values = np.mean([np.abs(s).mean(axis=0) for s in sv], axis=0)
                else:
                    values = np.abs(sv).mean(axis=0)
                names = result.get("feature_names", [f"F{i}" for i in range(len(values))])
                values = values.flatten()
            else:
                return ""

            sorted_idx = np.argsort(values)[::-1]
            values = values[sorted_idx]
            names = [names[i] for i in sorted_idx]

            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(range(len(values)), values, color="#2563eb")
            ax.set_yticks(range(len(values)))
            ax.set_yticklabels(names)
            ax.set_xlabel("Importance Score")
            ax.set_title(title)
            ax.invert_yaxis()
            plt.tight_layout()

            path = os.path.join(self.save_dir, filename)
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info("Feature importance plot saved: %s", path)
            return path

        except ImportError:
            return ""
