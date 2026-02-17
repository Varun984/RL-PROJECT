"""
File: replay_buffer.py
Module: agents
Description: Standard uniform replay buffer and HER (Hindsight Experience Replay) buffer
    with goal relabelling strategies (final, future, episode). HER dramatically improves
    sample efficiency when the Micro agent rarely achieves the exact Macro goal.
Design Decisions: Numpy-backed circular buffer for memory efficiency. HER relabelling
    happens on-the-fly during sampling to avoid storing redundant transitions.
    Episode boundaries tracked for correct goal sampling.
References: Andrychowicz et al., "Hindsight Experience Replay", NeurIPS 2017
Author: HRL-SARP Framework
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# STANDARD REPLAY BUFFER
# ══════════════════════════════════════════════════════════════════════


class ReplayBuffer:
    """Circular replay buffer with uniform sampling."""

    def __init__(
        self,
        capacity: int = 1_000_000,
        state_dim: int = 1,
        action_dim: int = 1,
        device: str = "cpu",
    ) -> None:
        self.capacity = capacity
        self.device = torch.device(device)
        self.ptr = 0
        self.size = 0

        # Pre-allocate numpy arrays
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        logger.info(
            "ReplayBuffer initialised | capacity=%d | state=%d | action=%d",
            capacity, state_dim, action_dim,
        )

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "states": torch.tensor(self.states[idx], device=self.device),
            "actions": torch.tensor(self.actions[idx], device=self.device),
            "rewards": torch.tensor(self.rewards[idx], device=self.device),
            "next_states": torch.tensor(self.next_states[idx], device=self.device),
            "dones": torch.tensor(self.dones[idx], device=self.device),
        }

    def __len__(self) -> int:
        return self.size

    def is_ready(self, batch_size: int) -> bool:
        return self.size >= batch_size


# ══════════════════════════════════════════════════════════════════════
# PPO ROLLOUT BUFFER
# ══════════════════════════════════════════════════════════════════════


class RolloutBuffer:
    """On-policy rollout buffer for PPO. Stores full trajectories and computes GAE."""

    def __init__(
        self,
        n_steps: int = 128,
        state_dim: int = 1,
        action_dim: int = 1,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: str = "cpu",
    ) -> None:
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = torch.device(device)

        self.states = np.zeros((n_steps, state_dim), dtype=np.float32)
        self.actions = np.zeros((n_steps, action_dim), dtype=np.float32)
        self.rewards = np.zeros(n_steps, dtype=np.float32)
        self.values = np.zeros(n_steps, dtype=np.float32)
        self.log_probs = np.zeros(n_steps, dtype=np.float32)
        self.dones = np.zeros(n_steps, dtype=np.float32)

        # Computed after rollout
        self.advantages = np.zeros(n_steps, dtype=np.float32)
        self.returns = np.zeros(n_steps, dtype=np.float32)

        # Additional: sector embeddings for Macro agent
        self.sector_embeddings: Optional[np.ndarray] = None

        self.ptr = 0
        self.full = False

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
        sector_embedding: Optional[np.ndarray] = None,
    ) -> None:
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = float(done)

        if sector_embedding is not None:
            if self.sector_embeddings is None:
                self.sector_embeddings = np.zeros(
                    (self.n_steps,) + sector_embedding.shape, dtype=np.float32
                )
            self.sector_embeddings[self.ptr] = sector_embedding

        self.ptr += 1
        if self.ptr >= self.n_steps:
            self.full = True

    def compute_gae(self, last_value: float, last_done: bool) -> None:
        """Compute Generalised Advantage Estimation (Schulman 2016).

        GAE(lambda) smoothly interpolates between MC and TD estimates.
        """
        last_gae = 0.0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]

            # TD residual
            delta = self.rewards[t] + self.gamma * next_value * next_non_terminal - self.values[t]
            # GAE recursive
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values

    def get_batches(
        self, batch_size: int, normalize_advantages: bool = True
    ) -> List[Dict[str, torch.Tensor]]:
        """Yield mini-batches for PPO epochs."""
        if normalize_advantages:
            adv_mean = self.advantages.mean()
            adv_std = self.advantages.std() + 1e-8
            self.advantages = (self.advantages - adv_mean) / adv_std

        indices = np.arange(self.n_steps)
        np.random.shuffle(indices)

        batches = []
        for start in range(0, self.n_steps, batch_size):
            end = min(start + batch_size, self.n_steps)
            idx = indices[start:end]

            batch = {
                "states": torch.tensor(self.states[idx], device=self.device),
                "actions": torch.tensor(self.actions[idx], device=self.device),
                "log_probs": torch.tensor(self.log_probs[idx], device=self.device),
                "advantages": torch.tensor(self.advantages[idx], device=self.device),
                "returns": torch.tensor(self.returns[idx], device=self.device),
                "values": torch.tensor(self.values[idx], device=self.device),
            }

            if self.sector_embeddings is not None:
                batch["sector_embeddings"] = torch.tensor(
                    self.sector_embeddings[idx], device=self.device
                )

            batches.append(batch)

        return batches

    def reset(self) -> None:
        self.ptr = 0
        self.full = False


# ══════════════════════════════════════════════════════════════════════
# HER REPLAY BUFFER
# ══════════════════════════════════════════════════════════════════════


class HERReplayBuffer:
    """Hindsight Experience Replay buffer with goal relabelling.

    Stores transitions with goals and relabels them during sampling using
    the "future" strategy — for each transition (s, a, r, s', g), create
    k additional transitions substituting g with goals achieved at future
    timesteps in the same episode.
    """

    def __init__(
        self,
        capacity: int = 1_000_000,
        state_dim: int = 1,
        action_dim: int = 1,
        goal_dim: int = 14,
        her_k: int = 4,
        strategy: str = "future",
        goal_tolerance: float = 0.1,
        device: str = "cpu",
    ) -> None:
        self.capacity = capacity
        self.her_k = her_k
        self.strategy = strategy
        self.goal_tolerance = goal_tolerance
        self.device = torch.device(device)
        self.goal_dim = goal_dim

        self.ptr = 0
        self.size = 0

        # Main storage
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        # Goal storage
        self.desired_goals = np.zeros((capacity, goal_dim), dtype=np.float32)
        self.achieved_goals = np.zeros((capacity, goal_dim), dtype=np.float32)

        # Episode tracking for HER relabelling
        self.episode_starts: List[int] = []
        self.episode_lengths: List[int] = []
        self._current_episode_start: int = 0
        self._current_episode_len: int = 0

        # Map buffer index → (episode_idx, step_within_episode)
        self.idx_to_episode: np.ndarray = np.zeros((capacity, 2), dtype=np.int64)

        logger.info(
            "HERReplayBuffer initialised | capacity=%d | k=%d | strategy=%s",
            capacity, her_k, strategy,
        )

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        desired_goal: np.ndarray,
        achieved_goal: np.ndarray,
    ) -> None:
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)
        self.desired_goals[self.ptr] = desired_goal
        self.achieved_goals[self.ptr] = achieved_goal

        # Track episode
        ep_idx = len(self.episode_starts) - 1 if self.episode_starts else 0
        self.idx_to_episode[self.ptr] = [ep_idx, self._current_episode_len]
        self._current_episode_len += 1

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

        if done:
            self.episode_starts.append(self._current_episode_start)
            self.episode_lengths.append(self._current_episode_len)
            self._current_episode_start = self.ptr
            self._current_episode_len = 0

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch with HER goal relabelling applied."""
        # Number of real transitions vs HER-augmented
        n_real = batch_size // (1 + self.her_k)
        n_her = batch_size - n_real

        # Sample real transitions
        real_idx = np.random.randint(0, self.size, size=n_real)

        real_batch = {
            "states": self.states[real_idx].copy(),
            "actions": self.actions[real_idx].copy(),
            "rewards": self.rewards[real_idx].copy(),
            "next_states": self.next_states[real_idx].copy(),
            "dones": self.dones[real_idx].copy(),
            "goals": self.desired_goals[real_idx].copy(),
        }

        # HER relabelled transitions
        her_batch = self._relabel_goals(n_her)

        # Concatenate
        combined = {
            "states": np.concatenate([real_batch["states"], her_batch["states"]]),
            "actions": np.concatenate([real_batch["actions"], her_batch["actions"]]),
            "rewards": np.concatenate([real_batch["rewards"], her_batch["rewards"]]),
            "next_states": np.concatenate([real_batch["next_states"], her_batch["next_states"]]),
            "dones": np.concatenate([real_batch["dones"], her_batch["dones"]]),
            "goals": np.concatenate([real_batch["goals"], her_batch["goals"]]),
        }

        # Shuffle
        perm = np.random.permutation(batch_size)
        return {
            k: torch.tensor(v[perm], device=self.device)
            for k, v in combined.items()
        }

    def _relabel_goals(self, n_samples: int) -> Dict[str, np.ndarray]:
        """Apply HER goal relabelling to create augmented transitions."""
        result = {
            "states": np.zeros((n_samples, self.states.shape[1]), dtype=np.float32),
            "actions": np.zeros((n_samples, self.actions.shape[1]), dtype=np.float32),
            "rewards": np.zeros(n_samples, dtype=np.float32),
            "next_states": np.zeros((n_samples, self.states.shape[1]), dtype=np.float32),
            "dones": np.zeros(n_samples, dtype=np.float32),
            "goals": np.zeros((n_samples, self.goal_dim), dtype=np.float32),
        }

        if len(self.episode_starts) < 1:
            return result

        for i in range(n_samples):
            # Pick a random transition
            idx = np.random.randint(0, self.size)
            ep_idx, step_in_ep = self.idx_to_episode[idx]

            # Ensure episode bounds are valid
            if ep_idx >= len(self.episode_starts):
                ep_idx = len(self.episode_starts) - 1

            ep_start = self.episode_starts[ep_idx]
            ep_len = self.episode_lengths[ep_idx] if ep_idx < len(self.episode_lengths) else 1

            # Copy transition
            result["states"][i] = self.states[idx]
            result["actions"][i] = self.actions[idx]
            result["next_states"][i] = self.next_states[idx]

            # Relabel goal based on strategy
            new_goal = self._sample_alternative_goal(
                ep_start, ep_len, step_in_ep
            )
            result["goals"][i] = new_goal

            # Recompute reward with new goal
            result["rewards"][i] = self._compute_her_reward(
                self.achieved_goals[idx], new_goal
            )
            result["dones"][i] = self.dones[idx]

        return result

    def _sample_alternative_goal(
        self,
        ep_start: int,
        ep_len: int,
        current_step: int,
    ) -> np.ndarray:
        """Sample an alternative goal based on the HER strategy."""
        if self.strategy == "final":
            # Use the achieved goal at the end of the episode
            final_idx = (ep_start + ep_len - 1) % self.capacity
            return self.achieved_goals[final_idx].copy()

        elif self.strategy == "future":
            # Sample from future steps in the same episode
            remaining = ep_len - current_step - 1
            if remaining <= 0:
                final_idx = (ep_start + ep_len - 1) % self.capacity
                return self.achieved_goals[final_idx].copy()
            future_offset = np.random.randint(1, remaining + 1)
            future_idx = (ep_start + current_step + future_offset) % self.capacity
            return self.achieved_goals[future_idx].copy()

        elif self.strategy == "episode":
            # Sample from any step in the episode
            random_step = np.random.randint(0, ep_len)
            random_idx = (ep_start + random_step) % self.capacity
            return self.achieved_goals[random_idx].copy()

        else:
            raise ValueError(f"Unknown HER strategy: {self.strategy}")

    def _compute_her_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
    ) -> float:
        """Compute reward for HER relabelled transition.

        Uses cosine similarity between achieved and desired sector allocations.
        """
        dot = np.dot(achieved_goal, desired_goal)
        norm_a = np.linalg.norm(achieved_goal)
        norm_b = np.linalg.norm(desired_goal)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        cos_sim = dot / (norm_a * norm_b)
        # Sparse reward: +1 if close enough, 0 otherwise
        if cos_sim >= (1.0 - self.goal_tolerance):
            return 1.0
        return cos_sim - 1.0  # Dense negative reward proportional to distance

    def __len__(self) -> int:
        return self.size

    def is_ready(self, batch_size: int) -> bool:
        return self.size >= batch_size
