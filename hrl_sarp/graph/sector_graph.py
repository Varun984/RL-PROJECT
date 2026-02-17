"""
File: sector_graph.py
Module: graph
Description: Builds dynamic sector correlation graphs from rolling return data.
    The graph is re-computed periodically (e.g., every 60 trading days) and captures
    time-varying co-movement between NSE sectors. Used as adjacency input for GNN.
Design Decisions: Edge weights derived from rolling Pearson correlation with a
    configurable threshold (default 0.3). Supports both binary and weighted adjacency.
    Self-loops included for GCN stability. Optional Granger causality edges for
    directed graphs.
References: Kipf & Welling (2017) GCN, Bengio sector correlation analysis
Author: HRL-SARP Framework
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SectorGraph:
    """Dynamic sector correlation graph builder.

    11 NSE Sectors mapped to graph nodes. Edges represent correlation strength
    above a configurable threshold, re-estimated every `update_freq` days.
    """

    SECTOR_NAMES = [
        "IT", "Financials", "Pharma", "FMCG", "Auto",
        "Energy", "Metals", "Realty", "Telecom", "Media", "Infra",
    ]

    def __init__(
        self,
        n_sectors: int = 11,
        correlation_window: int = 60,
        correlation_threshold: float = 0.30,
        update_freq: int = 20,
        use_abs_correlation: bool = True,
        add_self_loops: bool = True,
    ) -> None:
        self.n_sectors = n_sectors
        self.window = correlation_window
        self.threshold = correlation_threshold
        self.update_freq = update_freq
        self.use_abs = use_abs_correlation
        self.add_self_loops = add_self_loops

        # Graph state
        self.adjacency: Optional[np.ndarray] = None
        self.edge_index: Optional[np.ndarray] = None
        self.edge_weights: Optional[np.ndarray] = None
        self.correlation_matrix: Optional[np.ndarray] = None
        self.steps_since_update: int = 0

        # Node features (sector-level)
        self.node_features: Optional[np.ndarray] = None

        logger.info(
            "SectorGraph initialised | sectors=%d | window=%d | threshold=%.2f",
            n_sectors, correlation_window, correlation_threshold,
        )

    # ── Graph Construction ───────────────────────────────────────────

    def update(
        self,
        sector_returns: np.ndarray,
        sector_features: Optional[np.ndarray] = None,
        force: bool = False,
    ) -> bool:
        """Update graph if enough steps have passed.

        Args:
            sector_returns: (T, 11) historical sector return matrix.
                Must have at least `correlation_window` rows.
            sector_features: (11, F) node features for each sector.
            force: Force update regardless of timing.

        Returns:
            True if graph was updated.
        """
        self.steps_since_update += 1

        if not force and self.steps_since_update < self.update_freq:
            return False

        if len(sector_returns) < self.window:
            logger.warning(
                "Insufficient data for graph update: %d < %d",
                len(sector_returns), self.window,
            )
            # Build default fully-connected graph
            self._build_default_graph()
            return True

        # Use last `window` days
        recent = sector_returns[-self.window:]
        self.correlation_matrix = np.corrcoef(recent.T)

        # Build adjacency from correlation
        self._build_adjacency_from_correlation(self.correlation_matrix)

        # Convert to edge index format (COO)
        self._adjacency_to_edge_index()

        # Update node features
        if sector_features is not None:
            self.node_features = sector_features
        else:
            self._compute_node_features(sector_returns)

        self.steps_since_update = 0
        logger.debug("Sector graph updated | edges=%d", len(self.edge_weights))
        return True

    def _build_adjacency_from_correlation(
        self,
        corr_matrix: np.ndarray,
    ) -> None:
        """Build adjacency matrix from correlation with threshold."""
        n = self.n_sectors
        adj = np.zeros((n, n), dtype=np.float32)

        corr = np.abs(corr_matrix) if self.use_abs else corr_matrix

        for i in range(n):
            for j in range(n):
                if i == j:
                    adj[i, j] = 1.0 if self.add_self_loops else 0.0
                elif corr[i, j] >= self.threshold:
                    adj[i, j] = float(corr[i, j])

        self.adjacency = adj

    def _build_default_graph(self) -> None:
        """Build default fully-connected graph with uniform weights."""
        n = self.n_sectors
        self.adjacency = np.ones((n, n), dtype=np.float32) * 0.5
        if self.add_self_loops:
            np.fill_diagonal(self.adjacency, 1.0)
        self.correlation_matrix = np.eye(n)
        self._adjacency_to_edge_index()
        if self.node_features is None:
            self.node_features = np.zeros((n, 8), dtype=np.float32)

    def _adjacency_to_edge_index(self) -> None:
        """Convert adjacency matrix to COO edge index + weights."""
        rows, cols = np.where(self.adjacency > 0)
        self.edge_index = np.stack([rows, cols], axis=0).astype(np.int64)
        self.edge_weights = self.adjacency[rows, cols].astype(np.float32)

    def _compute_node_features(
        self,
        sector_returns: np.ndarray,
    ) -> None:
        """Compute default node features from return statistics."""
        recent = sector_returns[-self.window:] if len(sector_returns) >= self.window else sector_returns
        n = min(recent.shape[1], self.n_sectors)

        features = np.zeros((self.n_sectors, 8), dtype=np.float32)
        for i in range(n):
            ret = recent[:, i]
            features[i, 0] = float(np.mean(ret))                          # Mean return
            features[i, 1] = float(np.std(ret))                           # Volatility
            features[i, 2] = float(np.prod(1 + ret) - 1)                  # Cum return
            features[i, 3] = float(np.mean(ret) / (np.std(ret) + 1e-8))   # Sharpe (daily)
            features[i, 4] = float(np.min(ret))                           # Worst day
            features[i, 5] = float(np.max(ret))                           # Best day
            features[i, 6] = float((ret > 0).mean())                      # Win rate
            features[i, 7] = float(np.median(ret))                        # Median return

        self.node_features = features

    # ── Graph Accessors ──────────────────────────────────────────────

    def get_edge_index_tensor(self):
        """Return edge_index as PyTorch LongTensor."""
        import torch
        if self.edge_index is None:
            self._build_default_graph()
        return torch.tensor(self.edge_index, dtype=torch.long)

    def get_edge_weight_tensor(self):
        """Return edge weights as PyTorch FloatTensor."""
        import torch
        if self.edge_weights is None:
            self._build_default_graph()
        return torch.tensor(self.edge_weights, dtype=torch.float32)

    def get_node_features_tensor(self):
        """Return node features as PyTorch FloatTensor."""
        import torch
        if self.node_features is None:
            self.node_features = np.zeros((self.n_sectors, 8), dtype=np.float32)
        return torch.tensor(self.node_features, dtype=torch.float32)

    def get_adjacency_tensor(self):
        """Return adjacency matrix as PyTorch FloatTensor."""
        import torch
        if self.adjacency is None:
            self._build_default_graph()
        return torch.tensor(self.adjacency, dtype=torch.float32)

    # ── Analysis ─────────────────────────────────────────────────────

    def get_sector_clusters(self, n_clusters: int = 3) -> Dict[str, List[str]]:
        """Cluster sectors based on correlation structure."""
        if self.correlation_matrix is None:
            return {"cluster_0": self.SECTOR_NAMES}

        # Simple spectral-like clustering using correlation distance
        distance = 1.0 - np.abs(self.correlation_matrix)
        n = min(self.n_sectors, distance.shape[0])

        # Agglomerative approach: iteratively merge closest sectors
        labels = list(range(n))
        for _ in range(n - n_clusters):
            # Find closest pair
            min_dist = float("inf")
            merge_i, merge_j = 0, 1
            for i in range(n):
                for j in range(i + 1, n):
                    if labels[i] != labels[j]:
                        d = distance[i, j]
                        if d < min_dist:
                            min_dist = d
                            merge_i, merge_j = i, j
            # Merge
            old_label = labels[merge_j]
            new_label = labels[merge_i]
            for k in range(n):
                if labels[k] == old_label:
                    labels[k] = new_label

        clusters: Dict[str, List[str]] = {}
        for i, label in enumerate(labels):
            key = f"cluster_{label}"
            if key not in clusters:
                clusters[key] = []
            if i < len(self.SECTOR_NAMES):
                clusters[key].append(self.SECTOR_NAMES[i])

        return clusters

    def get_graph_stats(self) -> Dict[str, Any]:
        """Return graph statistics."""
        if self.adjacency is None:
            return {"n_nodes": self.n_sectors, "n_edges": 0, "density": 0.0}

        n_edges = int((self.adjacency > 0).sum())
        max_edges = self.n_sectors ** 2
        density = n_edges / max_edges if max_edges > 0 else 0.0

        return {
            "n_nodes": self.n_sectors,
            "n_edges": n_edges,
            "density": density,
            "avg_degree": n_edges / self.n_sectors,
            "avg_edge_weight": float(self.edge_weights.mean()) if self.edge_weights is not None else 0.0,
        }
