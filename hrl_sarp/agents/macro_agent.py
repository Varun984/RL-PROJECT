"""
File: macro_agent.py
Module: agents
Description: MacroAgent implements PPO from scratch for weekly sector allocation.
    Contains actor-critic networks with attention over sector GNN embeddings,
    regime classification head, and full PPO update loop with clipped surrogate
    objective, entropy bonus, and value function loss.
Design Decisions: PPO is chosen for its stability and sample efficiency in on-policy
    settings. Dirichlet distribution for sector weights ensures valid allocations.
    Separate actor/critic networks avoid gradient interference.
References: Schulman et al., "Proximal Policy Optimization Algorithms", 2017 (eq. 7)
Author: HRL-SARP Framework
"""

import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from agents.networks import MacroActorNet, MacroCriticNet
from agents.replay_buffer import RolloutBuffer

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# MACRO AGENT (PPO)
# ══════════════════════════════════════════════════════════════════════


class MacroAgent:
    """PPO-based Macro agent for weekly sector allocation and regime prediction."""

    def __init__(
        self,
        config_path: str = "config/macro_agent_config.yaml",
        device: str = "cpu",
    ) -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device(device)
        net_cfg = self.config["network"]
        ppo_cfg = self.config["ppo"]

        # Architecture params
        self.macro_state_dim: int = net_cfg["macro_state_dim"]
        self.num_sectors: int = net_cfg["num_sectors"]
        self.sector_emb_dim: int = net_cfg["sector_embedding_dim"]
        self.regime_classes: int = net_cfg["actor"]["regime_classes"]

        # PPO params
        self.lr: float = ppo_cfg["learning_rate"]
        self.gamma: float = ppo_cfg["gamma"]
        self.gae_lambda: float = ppo_cfg["gae_lambda"]
        self.clip_eps: float = ppo_cfg["clip_epsilon"]
        self.entropy_coef: float = ppo_cfg["entropy_coef"]
        self.value_loss_coef: float = ppo_cfg["value_loss_coef"]
        self.max_grad_norm: float = ppo_cfg["max_grad_norm"]
        self.n_epochs: int = ppo_cfg["n_epochs"]
        self.batch_size: int = ppo_cfg["batch_size"]
        self.n_steps: int = ppo_cfg["n_steps"]
        self.normalize_advantages: bool = ppo_cfg["normalize_advantages"]

        # Build networks
        self.actor = MacroActorNet(
            macro_state_dim=self.macro_state_dim,
            num_sectors=self.num_sectors,
            sector_emb_dim=self.sector_emb_dim,
            attention_heads=net_cfg["attention"]["num_heads"],
            attention_dropout=net_cfg["attention"]["dropout"],
            mlp_hidden_dims=net_cfg["mlp_hidden_dims"],
            regime_classes=self.regime_classes,
        ).to(self.device)

        self.critic = MacroCriticNet(
            macro_state_dim=self.macro_state_dim,
            num_sectors=self.num_sectors,
            sector_emb_dim=self.sector_emb_dim,
            attention_heads=net_cfg["attention"]["num_heads"],
            attention_dropout=net_cfg["attention"]["dropout"],
            mlp_hidden_dims=net_cfg["mlp_hidden_dims"],
        ).to(self.device)

        # Optimiser
        opt_cfg = self.config["optimizer"]
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.lr,
            betas=tuple(opt_cfg["betas"]),
            eps=opt_cfg["eps"],
            weight_decay=opt_cfg["weight_decay"],
        )

        # LR scheduler
        lr_cfg = self.config["lr_schedule"]
        if lr_cfg["type"] == "cosine_annealing":
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=lr_cfg["T_max"],
                eta_min=lr_cfg["eta_min"],
            )
        else:
            self.scheduler = None

        # Observation: macro_state + flattened sector embeddings
        obs_dim = self.macro_state_dim + self.num_sectors * self.sector_emb_dim
        action_dim = self.num_sectors + self.regime_classes

        # Rollout buffer
        self.buffer = RolloutBuffer(
            n_steps=self.n_steps,
            state_dim=obs_dim,
            action_dim=action_dim,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            device=device,
        )

        # Training state
        self.total_steps: int = 0
        self.update_count: int = 0

        logger.info(
            "MacroAgent initialised | actor_params=%d | critic_params=%d",
            sum(p.numel() for p in self.actor.parameters()),
            sum(p.numel() for p in self.critic.parameters()),
        )

    # ── Action Selection ─────────────────────────────────────────────

    @torch.no_grad()
    def select_action(
        self,
        macro_state: np.ndarray,
        sector_embeddings: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, float, float]:
        """Select action from policy.

        Args:
            macro_state: (18,) macro features.
            sector_embeddings: (11, 64) sector GNN embeddings.
            deterministic: If True, use mean action.

        Returns:
            (action, log_prob, value)
        """
        self.actor.eval()
        self.critic.eval()

        state_t = torch.tensor(macro_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        emb_t = torch.tensor(sector_embeddings, dtype=torch.float32, device=self.device).unsqueeze(0)

        if deterministic:
            sector_weights, regime_logits, _ = self.actor(state_t, emb_t)
            regime_idx = regime_logits.argmax(dim=-1)
            regime_onehot = torch.nn.functional.one_hot(
                regime_idx, num_classes=self.regime_classes
            ).float()
            action_t = torch.cat([sector_weights, regime_onehot], dim=-1)
            log_prob = 0.0
        else:
            action_t, log_prob_t, _, _ = self.actor.get_action_and_log_prob(state_t, emb_t)
            log_prob = float(log_prob_t.item())

        value = float(self.critic(state_t, emb_t).item())
        action = action_t.squeeze(0).cpu().numpy()

        return action, log_prob, value

    # ── Rollout Storage ──────────────────────────────────────────────

    def store_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
        sector_embedding: Optional[np.ndarray] = None,
    ) -> None:
        """Store a transition in the rollout buffer."""
        self.buffer.add(obs, action, reward, value, log_prob, done, sector_embedding)
        self.total_steps += 1

    # ── PPO Update ───────────────────────────────────────────────────

    def update(
        self,
        last_value: float,
        last_done: bool,
    ) -> Dict[str, float]:
        """Perform PPO update using collected rollout data.

        PPO clipped surrogate objective (Schulman 2017, eq. 7):
        L_clip = E[min(r(θ)*A, clip(r(θ), 1-ε, 1+ε)*A)]

        Returns:
            Dict of training metrics.
        """
        self.actor.train()
        self.critic.train()

        # Compute GAE advantages
        self.buffer.compute_gae(last_value, last_done)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        total_batches = 0

        for epoch in range(self.n_epochs):
            batches = self.buffer.get_batches(
                self.batch_size, self.normalize_advantages
            )

            for batch in batches:
                states = batch["states"]
                actions = batch["actions"]
                old_log_probs = batch["log_probs"]
                advantages = batch["advantages"]
                returns = batch["returns"]
                old_values = batch["values"]

                # Split state into macro_state and sector_embeddings
                macro_state, sector_emb = self._split_observation(states)

                # Evaluate current policy on old actions
                new_log_probs, entropy = self.actor.evaluate_action(
                    macro_state, sector_emb, actions
                )
                new_values = self.critic(macro_state, sector_emb)

                # PPO ratio
                ratio = torch.exp(new_log_probs - old_log_probs)

                # Clipped surrogate loss
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # PPO value loss clipping to stabilise critic updates.
                value_pred_clipped = old_values + (new_values - old_values).clamp(
                    -self.clip_eps, self.clip_eps
                )
                value_loss_unclipped = (new_values - returns) ** 2
                value_loss_clipped = (value_pred_clipped - returns) ** 2
                value_loss = 0.5 * torch.max(
                    value_loss_unclipped, value_loss_clipped
                ).mean()

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.value_loss_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                total_batches += 1

        if self.scheduler is not None:
            self.scheduler.step()

        self.update_count += 1
        self.buffer.reset()

        metrics = {
            "policy_loss": total_policy_loss / max(total_batches, 1),
            "value_loss": total_value_loss / max(total_batches, 1),
            "entropy": -total_entropy_loss / max(total_batches, 1),
            "update_count": self.update_count,
            "lr": self.optimizer.param_groups[0]["lr"],
        }

        logger.info(
            "PPO Update %d | policy_loss=%.4f | value_loss=%.4f | entropy=%.4f",
            self.update_count, metrics["policy_loss"],
            metrics["value_loss"], metrics["entropy"],
        )

        return metrics

    def _split_observation(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split flat observation into macro_state and sector_embeddings."""
        macro_state = obs[:, :self.macro_state_dim]
        sector_flat = obs[:, self.macro_state_dim:]
        sector_emb = sector_flat.view(-1, self.num_sectors, self.sector_emb_dim)
        return macro_state, sector_emb

    # ── Goal Output ──────────────────────────────────────────────────

    def get_goal_embedding(self, action: np.ndarray) -> np.ndarray:
        """Extract goal embedding (14D) from the agent's action output."""
        sector_weights = action[:self.num_sectors]
        regime_onehot = action[self.num_sectors:]
        return np.concatenate([sector_weights, regime_onehot])

    # ── Persistence ──────────────────────────────────────────────────

    def save(self, path: str) -> None:
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "update_count": self.update_count,
        }, path)
        logger.info("MacroAgent saved to %s", path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.total_steps = ckpt.get("total_steps", 0)
        self.update_count = ckpt.get("update_count", 0)
        logger.info("MacroAgent loaded from %s", path)

    def freeze(self) -> None:
        """Freeze all parameters (for hierarchical training phases)."""
        for p in self.actor.parameters():
            p.requires_grad = False
        for p in self.critic.parameters():
            p.requires_grad = False
        logger.info("MacroAgent frozen")

    def unfreeze(self) -> None:
        for p in self.actor.parameters():
            p.requires_grad = True
        for p in self.critic.parameters():
            p.requires_grad = True
        logger.info("MacroAgent unfrozen")
