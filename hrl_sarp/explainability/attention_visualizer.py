"""
File: attention_visualizer.py
Module: explainability
Description: Extract and visualize attention weights from Macro and Micro agent
    networks. Reveals which sectors/stocks the agent attends to when making
    allocation decisions. Supports heatmap, time-series, and network graph outputs.
Design Decisions: Hooks into PyTorch attention layers via forward hooks. Stores
    attention maps per step for temporal analysis. Can produce static (matplotlib)
    and interactive (plotly) visualisations.
References: Attention mechanism interpretability (Vaswani 2017, Wiegreffe 2019)
Author: HRL-SARP Framework
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class AttentionVisualizer:
    """Extract and visualise attention weights from HRL agent networks."""

    SECTOR_NAMES = [
        "IT", "Financials", "Pharma", "FMCG", "Auto",
        "Energy", "Metals", "Realty", "Telecom", "Media", "Infra",
    ]

    def __init__(self, save_dir: str = "logs/attention") -> None:
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Stored attention maps
        self.macro_attention_history: List[np.ndarray] = []
        self.micro_attention_history: List[np.ndarray] = []
        self._hooks: List[torch.utils.hooks.RemovableHook] = []

    # ── Hook Registration ────────────────────────────────────────────

    def register_hooks(
        self,
        macro_actor: Optional[nn.Module] = None,
        micro_actor: Optional[nn.Module] = None,
    ) -> None:
        """Register forward hooks on attention layers to capture weights."""
        self.remove_hooks()

        if macro_actor is not None:
            self._register_attention_hooks(macro_actor, "macro")

        if micro_actor is not None:
            self._register_attention_hooks(micro_actor, "micro")

        logger.info("Attention hooks registered")

    def _register_attention_hooks(
        self,
        model: nn.Module,
        agent_type: str,
    ) -> None:
        """Find and hook MultiheadAttention modules."""
        for name, module in model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                hook = module.register_forward_hook(
                    self._make_hook(agent_type, name)
                )
                self._hooks.append(hook)
                logger.debug("Hook on %s.%s", agent_type, name)

    def _make_hook(self, agent_type: str, layer_name: str):
        """Create a hook closure for capturing attention weights."""
        def hook_fn(module, input, output):
            # MultiheadAttention returns (attn_output, attn_weights)
            if isinstance(output, tuple) and len(output) >= 2:
                attn_weights = output[1]
                if attn_weights is not None:
                    weights_np = attn_weights.detach().cpu().numpy()
                    if agent_type == "macro":
                        self.macro_attention_history.append(weights_np)
                    else:
                        self.micro_attention_history.append(weights_np)
        return hook_fn

    def remove_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    # ── Attention Extraction ─────────────────────────────────────────

    @torch.no_grad()
    def extract_macro_attention(
        self,
        macro_actor: nn.Module,
        macro_state: np.ndarray,
        sector_embeddings: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Extract attention weights from macro actor for a single input.

        Returns:
            Dict with 'attention_weights', 'sector_scores' arrays.
        """
        macro_actor.eval()
        self.macro_attention_history.clear()

        # Register temporary hooks
        self.register_hooks(macro_actor=macro_actor)

        state_t = torch.tensor(macro_state, dtype=torch.float32).unsqueeze(0)
        emb_t = torch.tensor(sector_embeddings, dtype=torch.float32).unsqueeze(0)

        # Forward pass triggers hooks
        macro_actor(state_t, emb_t)

        self.remove_hooks()

        result = {}
        if self.macro_attention_history:
            # Last attention layer weights: (1, n_heads, seq_len, seq_len)
            attn = self.macro_attention_history[-1]
            result["attention_weights"] = attn

            # Average across heads → (seq_len, seq_len)
            avg_attn = attn.mean(axis=1).squeeze(0)
            result["avg_attention"] = avg_attn

            # Sector importance: sum of attention received from other sectors
            sector_scores = avg_attn.sum(axis=0)
            sector_scores /= sector_scores.sum() + 1e-8
            result["sector_scores"] = sector_scores

        macro_actor.train()
        return result

    @torch.no_grad()
    def extract_micro_attention(
        self,
        micro_actor: nn.Module,
        stock_features: np.ndarray,
        goal: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Extract attention weights from micro actor."""
        micro_actor.eval()
        self.micro_attention_history.clear()

        self.register_hooks(micro_actor=micro_actor)

        feats_t = torch.tensor(stock_features, dtype=torch.float32).unsqueeze(0)
        goal_t = torch.tensor(goal, dtype=torch.float32).unsqueeze(0)
        mask_t = None
        if mask is not None:
            mask_t = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        micro_actor(feats_t, goal_t, mask_t)

        self.remove_hooks()

        result = {}
        if self.micro_attention_history:
            attn = self.micro_attention_history[-1]
            result["attention_weights"] = attn

            avg_attn = attn.mean(axis=1).squeeze(0)
            result["avg_attention"] = avg_attn

            stock_scores = avg_attn.sum(axis=0)
            stock_scores /= stock_scores.sum() + 1e-8
            result["stock_scores"] = stock_scores

        micro_actor.train()
        return result

    # ── Visualisation ────────────────────────────────────────────────

    def plot_sector_attention_heatmap(
        self,
        attention_matrix: np.ndarray,
        title: str = "Macro Sector Attention",
        filename: str = "sector_attention.png",
    ) -> str:
        """Plot sector-to-sector attention heatmap."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            n = min(attention_matrix.shape[0], len(self.SECTOR_NAMES))
            labels = self.SECTOR_NAMES[:n]

            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(attention_matrix[:n, :n], cmap="YlOrRd", aspect="auto")

            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_yticklabels(labels)
            ax.set_title(title)
            ax.set_xlabel("Key Sectors")
            ax.set_ylabel("Query Sectors")

            # Annotate cells
            for i in range(n):
                for j in range(n):
                    ax.text(j, i, f"{attention_matrix[i, j]:.2f}",
                            ha="center", va="center", fontsize=8,
                            color="white" if attention_matrix[i, j] > 0.5 else "black")

            plt.colorbar(im, ax=ax, label="Attention Weight")
            plt.tight_layout()

            path = os.path.join(self.save_dir, filename)
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info("Attention heatmap saved: %s", path)
            return path

        except ImportError:
            logger.warning("matplotlib not available for plotting")
            return ""

    def plot_temporal_attention(
        self,
        attention_history: List[np.ndarray],
        top_k: int = 5,
        title: str = "Sector Attention Over Time",
        filename: str = "temporal_attention.png",
    ) -> str:
        """Plot how attention to sectors evolves over time."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            if not attention_history:
                return ""

            # Extract sector importance scores over time
            n_steps = len(attention_history)
            n_sectors = attention_history[0].shape[-1] if len(attention_history[0].shape) > 1 else 0
            n_sectors = min(n_sectors, len(self.SECTOR_NAMES))

            if n_sectors == 0:
                return ""

            scores = np.zeros((n_steps, n_sectors))
            for t, attn in enumerate(attention_history):
                if attn.ndim >= 2:
                    avg = attn.mean(axis=tuple(range(attn.ndim - 1)))
                    n = min(len(avg), n_sectors)
                    scores[t, :n] = avg[:n]

            fig, ax = plt.subplots(figsize=(14, 6))
            for s in range(min(top_k, n_sectors)):
                ax.plot(scores[:, s], label=self.SECTOR_NAMES[s], alpha=0.8)

            ax.set_xlabel("Time Step")
            ax.set_ylabel("Attention Score")
            ax.set_title(title)
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()

            path = os.path.join(self.save_dir, filename)
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            return path

        except ImportError:
            return ""

    def clear_history(self) -> None:
        """Clear stored attention histories."""
        self.macro_attention_history.clear()
        self.micro_attention_history.clear()
