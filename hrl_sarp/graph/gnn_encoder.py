"""
File: gnn_encoder.py
Module: graph
Description: Graph Neural Network encoder for generating sector embeddings.
    Implements GCN and GAT variants that operate on the dynamic sector graph
    to produce 64-dimensional sector embeddings for the Macro agent.
Design Decisions: Pure PyTorch implementation (no PyG dependency) for portability.
    Supports both GCN (Kipf & Welling 2017) and GAT (Veličković 2018) layers.
    Output is (11, 64) sector embedding matrix per forward pass.
References: Kipf & Welling (2017) GCN, Veličković et al. (2018) GAT
Author: HRL-SARP Framework
"""

import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# GCN LAYER (from scratch)
# ══════════════════════════════════════════════════════════════════════


class GCNLayer(nn.Module):
    """Graph Convolutional Network layer (Kipf & Welling 2017).

    H^{(l+1)} = σ(D̃^{-1/2} Ã D̃^{-1/2} H^{(l)} W^{(l)})
    where Ã = A + I (adjacency with self-loops), D̃ = degree matrix of Ã.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (N, in_features) node features.
            adj: (N, N) adjacency matrix (with self-loops preferred).

        Returns:
            (N, out_features) updated node features.
        """
        # Normalise adjacency: D^{-1/2} A D^{-1/2}
        adj_hat = self._normalise_adj(adj)

        # (N, in) @ (in, out) = (N, out)
        support = x @ self.weight

        # (N, N) @ (N, out) = (N, out)
        out = adj_hat @ support

        if self.bias is not None:
            out = out + self.bias

        return out

    @staticmethod
    def _normalise_adj(adj: torch.Tensor) -> torch.Tensor:
        """Symmetric normalisation: D^{-1/2} A D^{-1/2}."""
        degree = adj.sum(dim=1).clamp(min=1e-8)
        d_inv_sqrt = degree.pow(-0.5)
        d_inv_sqrt[d_inv_sqrt == float("inf")] = 0.0
        # D^{-1/2} A D^{-1/2}
        return (d_inv_sqrt.unsqueeze(1) * adj) * d_inv_sqrt.unsqueeze(0)


# ══════════════════════════════════════════════════════════════════════
# GAT LAYER (from scratch)
# ══════════════════════════════════════════════════════════════════════


class GATLayer(nn.Module):
    """Graph Attention Network layer (Veličković et al. 2018).

    Computes attention over neighbours to weight their messages.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        concat: bool = True,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.out_features = out_features
        self.concat = concat

        self.W = nn.Linear(in_features, n_heads * out_features, bias=False)
        self.a_src = nn.Parameter(torch.empty(n_heads, out_features))
        self.a_dst = nn.Parameter(torch.empty(n_heads, out_features))
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (N, in_features) node features.
            adj: (N, N) adjacency (used as mask).

        Returns:
            (N, n_heads * out_features) if concat else (N, out_features)
        """
        N = x.size(0)

        # Linear projection: (N, n_heads * out)
        h = self.W(x).view(N, self.n_heads, self.out_features)

        # Attention scores
        # (N, n_heads) for source and destination
        attn_src = (h * self.a_src.unsqueeze(0)).sum(dim=-1)  # (N, n_heads)
        attn_dst = (h * self.a_dst.unsqueeze(0)).sum(dim=-1)  # (N, n_heads)

        # (N, N, n_heads) pairwise attention
        attn = attn_src.unsqueeze(1) + attn_dst.unsqueeze(0)  # (N, N, n_heads)
        attn = self.leaky_relu(attn)

        # Mask: only attend to neighbours (where adj > 0)
        mask = (adj > 0).unsqueeze(-1).expand_as(attn)
        attn = attn.masked_fill(~mask, float("-inf"))

        # Softmax over neighbours
        attn = F.softmax(attn, dim=1)
        attn = self.dropout(attn)

        # Message passing: (N, N, heads) @ (N, heads, out) → (N, heads, out)
        # Einsum: (i,j,h) * (j,h,d) -> (i,h,d)
        out = torch.einsum("ijh,jhd->ihd", attn, h)

        if self.concat:
            return out.reshape(N, self.n_heads * self.out_features)
        else:
            return out.mean(dim=1)  # (N, out_features)


# ══════════════════════════════════════════════════════════════════════
# GNN ENCODER
# ══════════════════════════════════════════════════════════════════════


class GNNEncoder(nn.Module):
    """Multi-layer GNN encoder producing sector embeddings.

    Input:  (N, node_feat_dim) node features + (N, N) adjacency
    Output: (N, embedding_dim) sector embeddings

    Typical: N=11, node_feat_dim=8, embedding_dim=64
    """

    def __init__(
        self,
        node_feat_dim: int = 8,
        hidden_dim: int = 32,
        embedding_dim: int = 64,
        n_layers: int = 2,
        gnn_type: str = "gcn",
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.gnn_type = gnn_type.lower()
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim

        layers = nn.ModuleList()
        norms = nn.ModuleList()

        if self.gnn_type == "gat":
            # GAT layers
            in_dim = node_feat_dim
            for i in range(n_layers - 1):
                layers.append(GATLayer(in_dim, hidden_dim, n_heads, dropout, concat=True))
                norms.append(nn.LayerNorm(hidden_dim * n_heads))
                in_dim = hidden_dim * n_heads
            # Final layer: average heads, no concat
            layers.append(GATLayer(in_dim, embedding_dim, n_heads, dropout, concat=False))
            norms.append(nn.LayerNorm(embedding_dim))
        else:
            # GCN layers (default)
            in_dim = node_feat_dim
            for i in range(n_layers - 1):
                layers.append(GCNLayer(in_dim, hidden_dim))
                norms.append(nn.LayerNorm(hidden_dim))
                in_dim = hidden_dim
            layers.append(GCNLayer(in_dim, embedding_dim))
            norms.append(nn.LayerNorm(embedding_dim))

        self.layers = layers
        self.norms = norms
        self.dropout = nn.Dropout(dropout)

        # Readout MLP (optional refinement)
        self.readout = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "GNNEncoder initialised | type=%s | layers=%d | embed=%d | params=%d",
            gnn_type, n_layers, embedding_dim, total_params,
        )

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        """Generate sector embeddings.

        Args:
            node_features: (B, N, F) or (N, F) node features.
            adjacency: (B, N, N) or (N, N) adjacency matrix.

        Returns:
            (B, N, embedding_dim) or (N, embedding_dim) sector embeddings.
        """
        batched = node_features.dim() == 3

        if batched:
            B, N, F = node_features.shape
            embeddings = []
            for b in range(B):
                emb = self._forward_single(node_features[b], adjacency[b])
                embeddings.append(emb)
            return torch.stack(embeddings, dim=0)
        else:
            return self._forward_single(node_features, adjacency)

    def _forward_single(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for a single graph."""
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            # Check if skip connection is feasible (matching last dim)
            has_weight = hasattr(layer, "weight")
            can_skip = has_weight and (x.shape[-1] == layer.weight.shape[-1])
            x_res = x if can_skip else None

            x = layer(x, adj)
            x = norm(x)
            if i < self.n_layers - 1:
                x = F.relu(x)
                x = self.dropout(x)
            # Skip connection where dimensions match
            if x_res is not None and x_res.shape == x.shape:
                x = x + x_res

        # Final refinement
        x = self.readout(x)

        return x

    def get_graph_embedding(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        """Get a single graph-level embedding (mean pooling over sectors).

        Returns:
            (B, embedding_dim) or (embedding_dim,) graph embedding.
        """
        node_emb = self.forward(node_features, adjacency)
        if node_emb.dim() == 3:
            return node_emb.mean(dim=1)
        return node_emb.mean(dim=0)
