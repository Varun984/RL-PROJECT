"""
File: networks.py
Module: agents
Description: All neural network architectures for HRL-SARP. Contains Macro actor/critic
    (PPO), Micro actor/twin-critic (TD3), multi-timeframe attention encoder, and
    goal-conditioned encoder. Architecture dimensions follow config specs exactly.
Design Decisions: LayerNorm + GELU throughout for training stability and smooth gradients.
    Multi-head attention captures cross-sector dependencies for Macro, cross-stock
    dependencies for Micro. Weight init uses orthogonal (PPO best practice).
References: PPO (Schulman 2017), TD3 (Fujimoto 2018), Attention (Vaswani 2017)
Author: HRL-SARP Framework
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
# UTILITY LAYERS
# ══════════════════════════════════════════════════════════════════════


def build_mlp(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    activation: str = "gelu",
    use_layer_norm: bool = True,
    output_activation: Optional[str] = None,
) -> nn.Sequential:
    """Build a multi-layer perceptron with LayerNorm + activation."""
    act_fn = {"gelu": nn.GELU, "relu": nn.ReLU, "tanh": nn.Tanh, "leaky_relu": nn.LeakyReLU}
    act_class = act_fn.get(activation, nn.GELU)

    layers: List[nn.Module] = []
    prev_dim = input_dim
    for h_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, h_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(h_dim))
        layers.append(act_class())
        prev_dim = h_dim

    layers.append(nn.Linear(prev_dim, output_dim))
    if output_activation == "tanh":
        layers.append(nn.Tanh())

    return nn.Sequential(*layers)


def orthogonal_init(module: nn.Module, gain: float = math.sqrt(2)) -> None:
    """Orthogonal weight initialisation — PPO best practice (Andrychowicz 2021)."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# ══════════════════════════════════════════════════════════════════════
# MULTI-HEAD ATTENTION ENCODER
# ══════════════════════════════════════════════════════════════════════


class MultiTimeframeAttentionEncoder(nn.Module):
    """Multi-head self-attention over a set of entity embeddings.

    Used by Macro agent over sector GNN embeddings (11 sectors × 64D)
    and by Micro agent over stock embeddings (N stocks × 64D).
    Applies scaled dot-product attention then mean-pools to a fixed-size output.
    """

    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: optional (batch, seq_len), 1 = valid, 0 = padding
        Returns:
            (batch, d_model) — mean-pooled attention output
        """
        B, S, D = x.shape

        # Project Q, K, V
        Q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, S, S)

        if mask is not None:
            # Expand mask for heads: (B, 1, 1, S)
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask_expanded == 0, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)  # (B, H, S, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, D)
        attn_output = self.out_proj(attn_output)

        # Residual + LayerNorm
        x = self.layer_norm(x + attn_output)

        # Mean pool across sequence (mask-aware)
        if mask is not None:
            mask_f = mask.unsqueeze(-1).float()
            pooled = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-8)
        else:
            pooled = x.mean(dim=1)

        return pooled  # (B, d_model)


# ══════════════════════════════════════════════════════════════════════
# GOAL-CONDITIONED ENCODER
# ══════════════════════════════════════════════════════════════════════


class GoalConditionedEncoder(nn.Module):
    """Encodes the Macro agent's output (sector weights + regime probs)
    into a dense goal embedding for the Micro agent.

    Input: 14D (11 sector weights + 3 regime probs) → Output: 64D
    """

    def __init__(
        self,
        input_dim: int = 14,
        hidden_dim: int = 64,
        output_dim: int = 64,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )
        self.apply(lambda m: orthogonal_init(m, gain=1.0))

    def forward(self, goal: torch.Tensor) -> torch.Tensor:
        return self.net(goal)


# ══════════════════════════════════════════════════════════════════════
# MACRO AGENT NETWORKS (PPO)
# ══════════════════════════════════════════════════════════════════════


class MacroActorNet(nn.Module):
    """Macro PPO actor network.

    Input: macro_state (18D) + sector GNN embeddings (11×64D)
    Pipeline:
        1. Multi-head attention over GNN embeddings → (64D)
        2. Concatenate with macro_state → (82D)
        3. MLP [256, 128, 64] with LayerNorm + GELU
        4. Actor head: sector_weights (11D softmax) + regime_logits (3D)
    """

    def __init__(
        self,
        macro_state_dim: int = 18,
        num_sectors: int = 11,
        sector_emb_dim: int = 64,
        attention_heads: int = 4,
        attention_dropout: float = 0.1,
        mlp_hidden_dims: List[int] = None,
        regime_classes: int = 3,
    ) -> None:
        super().__init__()
        if mlp_hidden_dims is None:
            mlp_hidden_dims = [256, 128, 64]

        self.num_sectors = num_sectors
        self.sector_emb_dim = sector_emb_dim

        # Attention over sector embeddings
        self.sector_attention = MultiTimeframeAttentionEncoder(
            d_model=sector_emb_dim,
            num_heads=attention_heads,
            dropout=attention_dropout,
        )

        # MLP backbone: macro_state + attention output → latent
        mlp_input_dim = macro_state_dim + sector_emb_dim
        self.backbone = build_mlp(
            input_dim=mlp_input_dim,
            hidden_dims=mlp_hidden_dims,
            output_dim=mlp_hidden_dims[-1],
            activation="gelu",
            use_layer_norm=True,
        )

        # Sector weight head
        self.sector_head = nn.Linear(mlp_hidden_dims[-1], num_sectors)

        # Regime classification head
        self.regime_head = nn.Linear(mlp_hidden_dims[-1], regime_classes)

        # Log-std for sector weight distribution (learnable)
        self.log_std = nn.Parameter(torch.zeros(num_sectors + regime_classes))

        self.apply(lambda m: orthogonal_init(m, gain=math.sqrt(2)))
        # Smaller init for output heads
        orthogonal_init(self.sector_head, gain=0.01)
        orthogonal_init(self.regime_head, gain=0.01)

    def forward(
        self,
        macro_state: torch.Tensor,
        sector_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            macro_state: (B, 18)
            sector_embeddings: (B, 11, 64)
        Returns:
            sector_weights: (B, 11) — softmax probabilities
            regime_logits: (B, 3) — raw logits
            latent: (B, 64) — shared latent for critic
        """
        # Attention-pool sector embeddings
        attn_out = self.sector_attention(sector_embeddings)  # (B, 64)

        # Concatenate with macro features
        combined = torch.cat([macro_state, attn_out], dim=-1)  # (B, 82)

        # Shared backbone
        latent = self.backbone(combined)  # (B, 64)

        # Output heads
        sector_logits = self.sector_head(latent)
        sector_weights = F.softmax(sector_logits, dim=-1)

        regime_logits = self.regime_head(latent)

        return sector_weights, regime_logits, latent

    def get_action_and_log_prob(
        self,
        macro_state: torch.Tensor,
        sector_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action from policy distribution and return log probability.

        Uses Dirichlet distribution for sector weights (natural for allocations),
        and Categorical for regime prediction.
        """
        sector_weights, regime_logits, latent = self.forward(macro_state, sector_embeddings)

        # Sector weights: use concentration parameters from softmax output
        # Add small epsilon for numerical stability
        concentration = sector_weights * 10.0 + 0.1
        dirichlet = torch.distributions.Dirichlet(concentration)
        sampled_weights = dirichlet.rsample()
        sector_log_prob = dirichlet.log_prob(sampled_weights)

        # Regime: categorical
        regime_dist = torch.distributions.Categorical(logits=regime_logits)
        sampled_regime = regime_dist.sample()
        regime_log_prob = regime_dist.log_prob(sampled_regime)

        # Total log prob
        total_log_prob = sector_log_prob + regime_log_prob

        # Entropy for PPO
        entropy = dirichlet.entropy() + regime_dist.entropy()

        # Combine action: sector weights + regime one-hot encoded as logits
        regime_onehot = F.one_hot(sampled_regime, num_classes=regime_logits.shape[-1]).float()
        action = torch.cat([sampled_weights, regime_onehot], dim=-1)

        return action, total_log_prob, entropy, latent

    def evaluate_action(
        self,
        macro_state: torch.Tensor,
        sector_embeddings: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log probability and entropy of a given action (for PPO update)."""
        sector_weights, regime_logits, latent = self.forward(macro_state, sector_embeddings)

        # Parse action back
        sampled_weights = action[:, :self.num_sectors]
        regime_idx = action[:, self.num_sectors:].argmax(dim=-1)

        # Sector distribution
        concentration = sector_weights * 10.0 + 0.1
        dirichlet = torch.distributions.Dirichlet(concentration)
        # Clamp weights for log_prob stability
        clamped = sampled_weights.clamp(1e-6, 1.0 - 1e-6)
        clamped = clamped / clamped.sum(dim=-1, keepdim=True)
        sector_log_prob = dirichlet.log_prob(clamped)

        # Regime distribution
        regime_dist = torch.distributions.Categorical(logits=regime_logits)
        regime_log_prob = regime_dist.log_prob(regime_idx)

        total_log_prob = sector_log_prob + regime_log_prob
        entropy = dirichlet.entropy() + regime_dist.entropy()

        return total_log_prob, entropy


class MacroCriticNet(nn.Module):
    """Macro PPO critic: state → V(s).

    Same input as actor but outputs a scalar value estimate.
    """

    def __init__(
        self,
        macro_state_dim: int = 18,
        num_sectors: int = 11,
        sector_emb_dim: int = 64,
        attention_heads: int = 4,
        attention_dropout: float = 0.1,
        mlp_hidden_dims: List[int] = None,
    ) -> None:
        super().__init__()
        if mlp_hidden_dims is None:
            mlp_hidden_dims = [256, 128, 64]

        self.sector_attention = MultiTimeframeAttentionEncoder(
            d_model=sector_emb_dim,
            num_heads=attention_heads,
            dropout=attention_dropout,
        )

        mlp_input_dim = macro_state_dim + sector_emb_dim
        self.backbone = build_mlp(
            input_dim=mlp_input_dim,
            hidden_dims=mlp_hidden_dims,
            output_dim=mlp_hidden_dims[-1],
            activation="gelu",
            use_layer_norm=True,
        )

        self.value_head = nn.Linear(mlp_hidden_dims[-1], 1)

        self.apply(lambda m: orthogonal_init(m, gain=math.sqrt(2)))
        orthogonal_init(self.value_head, gain=1.0)

    def forward(
        self,
        macro_state: torch.Tensor,
        sector_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        attn_out = self.sector_attention(sector_embeddings)
        combined = torch.cat([macro_state, attn_out], dim=-1)
        latent = self.backbone(combined)
        value = self.value_head(latent).squeeze(-1)
        return value


# ══════════════════════════════════════════════════════════════════════
# MICRO AGENT NETWORKS (TD3)
# ══════════════════════════════════════════════════════════════════════


class MicroActorNet(nn.Module):
    """Micro TD3 actor network — goal-conditioned stock selection.

    Input: per-stock features (N, 22) + goal embedding (64D from GoalEncoder)
    Pipeline:
        1. GoalEncoder: raw goal (14D) → 64D embedding
        2. Concatenate goal with each stock's features → (N, 86D)
        3. Per-stock MLP [128, 64] with LayerNorm + GELU
        4. Self-attention across all stocks → (N, 64)
        5. Per-stock Linear(64, 1) → Softmax = portfolio weights
    """

    def __init__(
        self,
        stock_feature_dim: int = 22,
        goal_input_dim: int = 14,
        goal_emb_dim: int = 64,
        stock_mlp_hidden: List[int] = None,
        attention_heads: int = 4,
        attention_dropout: float = 0.1,
        max_stocks: int = 50,
    ) -> None:
        super().__init__()
        if stock_mlp_hidden is None:
            stock_mlp_hidden = [128, 64]

        self.max_stocks = max_stocks
        self.stock_feature_dim = stock_feature_dim
        self.goal_emb_dim = goal_emb_dim

        # Goal encoder: 14D → 64D
        self.goal_encoder = GoalConditionedEncoder(
            input_dim=goal_input_dim,
            hidden_dim=goal_emb_dim,
            output_dim=goal_emb_dim,
        )

        # Per-stock MLP: (stock_features + goal_embed) → 64D
        stock_input_dim = stock_feature_dim + goal_emb_dim
        layers: List[nn.Module] = []
        prev = stock_input_dim
        for h in stock_mlp_hidden:
            layers.extend([nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU()])
            prev = h
        self.stock_mlp = nn.Sequential(*layers)

        # Self-attention over stocks
        self.stock_attention = MultiTimeframeAttentionEncoder(
            d_model=stock_mlp_hidden[-1],
            num_heads=attention_heads,
            dropout=attention_dropout,
        )

        # Attention produces a pooled output, but we need per-stock attention
        # So we use raw multi-head attention (no pooling)
        self.stock_self_attn = nn.MultiheadAttention(
            embed_dim=stock_mlp_hidden[-1],
            num_heads=attention_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(stock_mlp_hidden[-1])

        # Output head: per-stock weight
        self.weight_head = nn.Linear(stock_mlp_hidden[-1], 1)

        self.apply(lambda m: orthogonal_init(m, gain=math.sqrt(2)))
        orthogonal_init(self.weight_head, gain=0.01)

    def forward(
        self,
        stock_features: torch.Tensor,
        goal: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            stock_features: (B, N, 22) per-stock feature matrix
            goal: (B, 14) raw goal from Macro
            mask: (B, N) stock validity mask, 1=valid, 0=padding
        Returns:
            weights: (B, N) portfolio weights summing to 1
        """
        B, N, _ = stock_features.shape

        # Encode goal and broadcast to each stock
        goal_emb = self.goal_encoder(goal)  # (B, 64)
        goal_expanded = goal_emb.unsqueeze(1).expand(B, N, -1)  # (B, N, 64)

        # Concatenate goal with stock features
        combined = torch.cat([stock_features, goal_expanded], dim=-1)  # (B, N, 86)

        # Per-stock MLP (applied identically to each stock)
        stock_emb = self.stock_mlp(combined)  # (B, N, 64)

        # Self-attention over stocks (captures cross-stock dependencies)
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = (mask == 0)  # True = ignore
        attn_out, _ = self.stock_self_attn(
            stock_emb, stock_emb, stock_emb,
            key_padding_mask=key_padding_mask,
        )
        stock_emb = self.attn_norm(stock_emb + attn_out)  # Residual + norm

        # Per-stock weight logits
        logits = self.weight_head(stock_emb).squeeze(-1)  # (B, N)

        # Mask invalid stocks
        if mask is not None:
            logits = logits.masked_fill(mask == 0, float("-inf"))

        weights = F.softmax(logits, dim=-1)
        return weights

    def get_deterministic_action(
        self,
        stock_features: torch.Tensor,
        goal: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return deterministic portfolio weights (for TD3 — deterministic policy)."""
        return self.forward(stock_features, goal, mask)


class TwinCriticNet(nn.Module):
    """Twin Q-networks for TD3 (clipped double-Q reduces overestimation).

    Input: state (flattened stock features + goal) + action (stock weights)
    Output: two Q-value estimates Q1(s,a) and Q2(s,a)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = None,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        input_dim = state_dim + action_dim

        # Q1
        self.q1 = build_mlp(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            activation="gelu",
            use_layer_norm=True,
        )

        # Q2 (independent parameters)
        self.q2 = build_mlp(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            activation="gelu",
            use_layer_norm=True,
        )

        self.apply(lambda m: orthogonal_init(m, gain=math.sqrt(2)))

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], dim=-1)
        q1 = self.q1(sa).squeeze(-1)
        q2 = self.q2(sa).squeeze(-1)
        return q1, q2

    def q1_forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Forward through Q1 only (used for actor loss in TD3)."""
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa).squeeze(-1)
