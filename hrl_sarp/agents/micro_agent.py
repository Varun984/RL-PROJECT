"""
File: micro_agent.py
Module: agents
Description: MicroAgent implements TD3 (Twin Delayed DDPG) with HER for daily stock
    selection conditioned on the Macro agent's goal. Features delayed policy updates,
    target policy smoothing, twin critics, and hindsight experience replay.
Design Decisions: TD3 is chosen for continuous action spaces (portfolio weights) where
    deterministic policy gradients outperform stochastic PPO. HER is critical because
    the Micro agent rarely achieves the exact sector allocation goal from Macro,
    so relabelling with achieved goals dramatically improves sample efficiency.
References:
    - Fujimoto et al., "Addressing Function Approximation Error in Actor-Critic", 2018
    - Andrychowicz et al., "Hindsight Experience Replay", 2017
Author: HRL-SARP Framework
"""

import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml

from agents.networks import MicroActorNet, TwinCriticNet
from agents.replay_buffer import HERReplayBuffer, ReplayBuffer

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# MICRO AGENT (TD3 + HER)
# ══════════════════════════════════════════════════════════════════════


class MicroAgent:
    """TD3 + HER agent for daily goal-conditioned stock selection."""

    def __init__(
        self,
        config_path: str = "config/micro_agent_config.yaml",
        device: str = "cpu",
    ) -> None:
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device(device)
        net_cfg = self.config["network"]
        td3_cfg = self.config["td3"]
        her_cfg = self.config["her"]
        buf_cfg = self.config["replay_buffer"]

        # Architecture params
        self.stock_feature_dim: int = net_cfg["stock_feature_dim"]
        self.goal_input_dim: int = net_cfg["goal_encoder"]["input_dim"]
        self.goal_emb_dim: int = net_cfg["goal_encoder"]["output_dim"]
        self.max_stocks: int = net_cfg["stock_attention"]["max_stocks"]

        # TD3 params
        self.lr_actor: float = td3_cfg["lr_actor"]
        self.lr_critic: float = td3_cfg["lr_critic"]
        self.gamma: float = td3_cfg["gamma"]
        self.tau: float = td3_cfg["tau"]
        self.policy_noise: float = td3_cfg["policy_noise"]
        self.noise_clip: float = td3_cfg["noise_clip"]
        self.policy_delay: int = td3_cfg["policy_delay"]
        self.max_grad_norm: float = td3_cfg["max_grad_norm"]

        # Exploration noise (linearly decayed)
        exp_cfg = td3_cfg["exploration_noise"]
        self.expl_noise_initial: float = exp_cfg["initial_std"]
        self.expl_noise_final: float = exp_cfg["final_std"]
        self.expl_decay_steps: int = exp_cfg["decay_steps"]

        # Build actor network
        self.actor = MicroActorNet(
            stock_feature_dim=self.stock_feature_dim,
            goal_input_dim=self.goal_input_dim,
            goal_emb_dim=self.goal_emb_dim,
            stock_mlp_hidden=net_cfg["stock_mlp"]["hidden_dims"],
            attention_heads=net_cfg["stock_attention"]["num_heads"],
            attention_dropout=net_cfg["stock_attention"]["dropout"],
            max_stocks=self.max_stocks,
        ).to(self.device)

        # Target actor (Polyak-averaged copy)
        self.actor_target = deepcopy(self.actor).to(self.device)
        self.actor_target.requires_grad_(False)

        # Build twin critic
        # State dim = flattened stock features + goal
        state_dim = self.max_stocks * self.stock_feature_dim + self.goal_input_dim
        action_dim = self.max_stocks

        self.critic = TwinCriticNet(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=net_cfg["critic"]["hidden_dims"],
        ).to(self.device)

        # Target critic
        self.critic_target = deepcopy(self.critic).to(self.device)
        self.critic_target.requires_grad_(False)

        # Optimisers
        actor_opt_cfg = self.config["optimizer"]["actor"]
        critic_opt_cfg = self.config["optimizer"]["critic"]

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=self.lr_actor,
            betas=tuple(actor_opt_cfg["betas"]),
            eps=actor_opt_cfg["eps"],
            weight_decay=actor_opt_cfg["weight_decay"],
        )

        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=self.lr_critic,
            betas=tuple(critic_opt_cfg["betas"]),
            eps=critic_opt_cfg["eps"],
            weight_decay=critic_opt_cfg["weight_decay"],
        )

        # LR schedulers
        lr_cfg = self.config["lr_schedule"]
        self.actor_scheduler = optim.lr_scheduler.StepLR(
            self.actor_optimizer,
            step_size=lr_cfg["actor"]["step_size"],
            gamma=lr_cfg["actor"]["gamma_decay"],
        )
        self.critic_scheduler = optim.lr_scheduler.StepLR(
            self.critic_optimizer,
            step_size=lr_cfg["critic"]["step_size"],
            gamma=lr_cfg["critic"]["gamma_decay"],
        )

        # Replay buffer (HER or standard)
        goal_dim = self.goal_input_dim
        if her_cfg["enabled"]:
            self.buffer = HERReplayBuffer(
                capacity=int(buf_cfg["capacity"]),
                state_dim=state_dim,
                action_dim=action_dim,
                goal_dim=goal_dim,
                her_k=her_cfg["k"],
                strategy=her_cfg["strategy"],
                goal_tolerance=her_cfg["goal_tolerance"],
                device=device,
            )
            self.use_her = True
        else:
            self.buffer = ReplayBuffer(
                capacity=int(buf_cfg["capacity"]),
                state_dim=state_dim,
                action_dim=action_dim,
                device=device,
            )
            self.use_her = False

        self.batch_size: int = buf_cfg["batch_size"]

        # Training state
        self.total_steps: int = 0
        self.update_count: int = 0
        self.actor_update_count: int = 0

        logger.info(
            "MicroAgent initialised | actor_params=%d | critic_params=%d | HER=%s",
            sum(p.numel() for p in self.actor.parameters()),
            sum(p.numel() for p in self.critic.parameters()),
            self.use_her,
        )

    # ── Action Selection ─────────────────────────────────────────────

    @torch.no_grad()
    def select_action(
        self,
        stock_features: np.ndarray,
        goal: np.ndarray,
        mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> np.ndarray:
        """Select portfolio weights from policy.

        Args:
            stock_features: (N, 22) or (max_stocks, 22) per-stock features.
            goal: (14,) raw goal from Macro agent.
            mask: (max_stocks,) validity mask.
            deterministic: If True, no exploration noise.

        Returns:
            (max_stocks,) portfolio weight vector.
        """
        self.actor.eval()

        # Pad features to max_stocks if needed
        n_stocks = stock_features.shape[0]
        padded = np.zeros((self.max_stocks, self.stock_feature_dim), dtype=np.float32)
        padded[:n_stocks] = stock_features[:self.max_stocks]

        if mask is None:
            mask = np.zeros(self.max_stocks, dtype=np.float32)
            mask[:n_stocks] = 1.0

        feat_t = torch.tensor(padded, dtype=torch.float32, device=self.device).unsqueeze(0)
        goal_t = torch.tensor(goal, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)

        weights = self.actor.get_deterministic_action(feat_t, goal_t, mask_t)
        weights = weights.squeeze(0).cpu().numpy()

        # Add exploration noise (only during training)
        if not deterministic:
            noise_std = self._get_exploration_noise()
            noise = np.random.normal(0, noise_std, size=weights.shape).astype(np.float32)
            weights = weights + noise * mask  # Only add noise to valid stocks

            # Re-normalise to valid portfolio weights
            weights = np.clip(weights, 0.0, 1.0)
            valid_sum = weights[:n_stocks].sum()
            if valid_sum > 1e-8:
                weights[:n_stocks] /= valid_sum
            else:
                weights[:n_stocks] = mask[:n_stocks] / max(mask[:n_stocks].sum(), 1e-8)
            weights[n_stocks:] = 0.0

        return weights

    def _get_exploration_noise(self) -> float:
        """Linearly decay exploration noise."""
        progress = min(self.total_steps / max(self.expl_decay_steps, 1), 1.0)
        return self.expl_noise_initial + progress * (self.expl_noise_final - self.expl_noise_initial)

    # ── Transition Storage ───────────────────────────────────────────

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        desired_goal: Optional[np.ndarray] = None,
        achieved_goal: Optional[np.ndarray] = None,
    ) -> None:
        """Store a transition in the replay buffer."""
        if self.use_her and desired_goal is not None and achieved_goal is not None:
            self.buffer.add(state, action, reward, next_state, done, desired_goal, achieved_goal)
        elif isinstance(self.buffer, ReplayBuffer):
            self.buffer.add(state, action, reward, next_state, done)
        self.total_steps += 1

    # ── TD3 Update ───────────────────────────────────────────────────

    def update(self) -> Dict[str, float]:
        """Perform one TD3 update step.

        TD3 key innovations:
        1. Twin critics (take min Q to reduce overestimation)
        2. Delayed policy updates (actor updates every policy_delay critic updates)
        3. Target policy smoothing (add noise to target action)

        Returns:
            Dict of training metrics.
        """
        if not self.buffer.is_ready(self.batch_size):
            return {}

        self.actor.train()
        self.critic.train()

        batch = self.buffer.sample(self.batch_size)

        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]

        # If HER buffer, goals are included in the batch
        # Goals are already embedded in state for the flat critic

        # ── 1. Critic Update ──

        with torch.no_grad():
            # Target policy smoothing: add clipped noise to target action
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )

            # Get target actor's action from next_state
            # For flat critic, we need to reconstruct stock features and goal
            next_stock_feats, next_goal, next_mask = self._parse_flat_state(next_states)
            next_action = self.actor_target(next_stock_feats, next_goal, next_mask)

            # Flatten next_action for critic
            next_action_flat = next_action.view(next_action.size(0), -1)
            next_action_flat = next_action_flat + noise
            next_action_flat = next_action_flat.clamp(0.0, 1.0)

            # Re-normalise
            row_sums = next_action_flat.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            next_action_flat = next_action_flat / row_sums

            # Twin target Q-values
            target_q1, target_q2 = self.critic_target(next_states, next_action_flat)
            target_q = torch.min(target_q1, target_q2)  # Clipped double-Q
            target_value = rewards + (1.0 - dones) * self.gamma * target_q

        # Current Q estimates
        current_q1, current_q2 = self.critic(states, actions)

        # Critic loss
        critic_loss = F.mse_loss(current_q1, target_value) + F.mse_loss(current_q2, target_value)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        self.update_count += 1

        metrics = {
            "critic_loss": critic_loss.item(),
            "q1_mean": current_q1.mean().item(),
            "q2_mean": current_q2.mean().item(),
        }

        # ── 2. Delayed Actor Update ──

        if self.update_count % self.policy_delay == 0:
            # Actor loss = -Q1(s, actor(s))
            stock_feats, goal, stock_mask = self._parse_flat_state(states)
            actor_action = self.actor(stock_feats, goal, stock_mask)
            actor_action_flat = actor_action.view(actor_action.size(0), -1)

            actor_loss = -self.critic.q1_forward(states, actor_action_flat).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            # Polyak update targets
            self._soft_update(self.actor, self.actor_target, self.tau)
            self._soft_update(self.critic, self.critic_target, self.tau)

            self.actor_update_count += 1
            metrics["actor_loss"] = actor_loss.item()

        # Step schedulers
        self.actor_scheduler.step()
        self.critic_scheduler.step()

        if self.update_count % 1000 == 0:
            logger.info(
                "TD3 Update %d | critic_loss=%.4f | q1=%.4f | noise=%.4f",
                self.update_count, metrics["critic_loss"],
                metrics["q1_mean"], self._get_exploration_noise(),
            )

        return metrics

    def _parse_flat_state(
        self, flat_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Parse flat state vector back into stock features + goal + mask.

        State layout: [stock_features_flat (max_stocks*22), goal (14)]
        """
        B = flat_state.size(0)
        stock_flat_dim = self.max_stocks * self.stock_feature_dim

        stock_flat = flat_state[:, :stock_flat_dim]
        goal = flat_state[:, stock_flat_dim:stock_flat_dim + self.goal_input_dim]

        stock_feats = stock_flat.view(B, self.max_stocks, self.stock_feature_dim)

        # Infer mask from features (non-zero rows are valid stocks)
        mask = (stock_feats.abs().sum(dim=-1) > 1e-6).float()

        return stock_feats, goal, mask

    @staticmethod
    def _soft_update(
        source: nn.Module,
        target: nn.Module,
        tau: float,
    ) -> None:
        """Polyak averaging: target = tau*source + (1-tau)*target."""
        for src_param, tgt_param in zip(source.parameters(), target.parameters()):
            tgt_param.data.copy_(tau * src_param.data + (1.0 - tau) * tgt_param.data)

    # ── Persistence ──────────────────────────────────────────────────

    def save(self, path: str) -> None:
        torch.save({
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "total_steps": self.total_steps,
            "update_count": self.update_count,
            "actor_update_count": self.actor_update_count,
        }, path)
        logger.info("MicroAgent saved to %s", path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
        self.total_steps = ckpt.get("total_steps", 0)
        self.update_count = ckpt.get("update_count", 0)
        self.actor_update_count = ckpt.get("actor_update_count", 0)
        logger.info("MicroAgent loaded from %s", path)

    def freeze(self) -> None:
        """Freeze all parameters (for hierarchical training phases)."""
        for p in self.actor.parameters():
            p.requires_grad = False
        for p in self.critic.parameters():
            p.requires_grad = False
        logger.info("MicroAgent frozen")

    def unfreeze(self) -> None:
        for p in self.actor.parameters():
            p.requires_grad = True
        for p in self.critic.parameters():
            p.requires_grad = True
        logger.info("MicroAgent unfrozen")
