"""
File: market_simulator.py
Module: environment
Description: GAN-based synthetic market generator for curriculum training. Generates
    Indian-market-specific stress scenarios (circuit breakers, expiry squeezes,
    flash crashes, sector rotations) to augment historical data.
Design Decisions: Uses a conditional WGAN-GP to generate realistic multi-asset return
    distributions conditioned on regime labels. Outputs can be injected directly
    into MacroEnv/MicroEnv as synthetic episodes, enabling curriculum learning
    from easy (bull) to hard (crisis) scenarios.
References: WGAN-GP (Gulrajani 2017), TimeGAN (Yoon 2019), Market simulation (Wiese 2020)
Author: HRL-SARP Framework
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# GAN COMPONENTS
# ══════════════════════════════════════════════════════════════════════


class Generator(nn.Module):
    """Conditional generator: noise + regime label → synthetic return sequence."""

    def __init__(
        self,
        noise_dim: int = 64,
        condition_dim: int = 3,
        output_dim: int = 11,
        seq_length: int = 5,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.seq_length = seq_length
        self.output_dim = output_dim

        input_dim = noise_dim + condition_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, seq_length * output_dim),
            nn.Tanh(),  # Returns bounded to [-1, 1], scaled later
        )

    def forward(self, noise: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        x = torch.cat([noise, condition], dim=-1)
        out = self.net(x)
        return out.view(-1, self.seq_length, self.output_dim)


class Critic(nn.Module):
    """WGAN-GP critic (no sigmoid): real/fake return sequences + condition → score."""

    def __init__(
        self,
        input_dim: int = 11,
        condition_dim: int = 3,
        seq_length: int = 5,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        flat_dim = seq_length * input_dim + condition_dim
        self.net = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, sequence: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        flat = sequence.view(sequence.size(0), -1)
        x = torch.cat([flat, condition], dim=-1)
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════
# MARKET SIMULATOR
# ══════════════════════════════════════════════════════════════════════


class MarketSimulator:
    """GAN-based synthetic market data generator for curriculum training.

    Generates regime-conditioned multi-sector return sequences that preserve
    the statistical properties of Indian equity markets.
    """

    def __init__(
        self,
        num_sectors: int = 11,
        seq_length: int = 5,
        noise_dim: int = 64,
        hidden_dim: int = 128,
        device: str = "cpu",
    ) -> None:
        self.num_sectors = num_sectors
        self.seq_length = seq_length
        self.noise_dim = noise_dim
        self.device = torch.device(device)
        self.condition_dim = 3  # Bull / Bear / Sideways

        self.generator = Generator(
            noise_dim=noise_dim,
            condition_dim=self.condition_dim,
            output_dim=num_sectors,
            seq_length=seq_length,
            hidden_dim=hidden_dim,
        ).to(self.device)

        self.critic = Critic(
            input_dim=num_sectors,
            condition_dim=self.condition_dim,
            seq_length=seq_length,
            hidden_dim=hidden_dim,
        ).to(self.device)

        # Return scaling factors (Indian market daily return statistics)
        self.return_scale: Dict[int, float] = {
            0: 0.015,  # Bull: avg daily sector move ~1.5%
            1: 0.025,  # Bear: higher volatility ~2.5%
            2: 0.008,  # Sideways: low volatility ~0.8%
        }
        self.return_bias: Dict[int, float] = {
            0: 0.002,   # Bull: positive drift
            1: -0.003,  # Bear: negative drift
            2: 0.0,     # Sideways: no drift
        }

        # Trained flag
        self.is_trained = False

        logger.info(
            "MarketSimulator initialised | sectors=%d | seq=%d | noise=%d",
            num_sectors, seq_length, noise_dim,
        )

    # ── Training ─────────────────────────────────────────────────────

    def train(
        self,
        real_returns: np.ndarray,
        regime_labels: np.ndarray,
        n_epochs: int = 200,
        batch_size: int = 64,
        lr: float = 1e-4,
        n_critic: int = 5,
        gp_lambda: float = 10.0,
    ) -> Dict[str, List[float]]:
        """Train WGAN-GP on historical sector returns conditioned on regime.

        Args:
            real_returns: (N_weeks, seq_length, num_sectors) real return sequences.
            regime_labels: (N_weeks,) regime label per sequence (0=Bull, 1=Bear, 2=Sideways).
            n_epochs: Number of training epochs.
            batch_size: Mini-batch size.
            lr: Learning rate for both generator and critic.
            n_critic: Critic updates per generator update.
            gp_lambda: Gradient penalty coefficient.

        Returns:
            Training history dict with g_loss and c_loss per epoch.
        """
        # Convert to tensors
        real_data = torch.tensor(real_returns, dtype=torch.float32, device=self.device)
        labels = torch.tensor(regime_labels, dtype=torch.long, device=self.device)
        n_samples = len(real_data)

        opt_g = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.0, 0.9))
        opt_c = optim.Adam(self.critic.parameters(), lr=lr, betas=(0.0, 0.9))

        history = {"g_loss": [], "c_loss": []}

        for epoch in range(n_epochs):
            perm = torch.randperm(n_samples, device=self.device)
            epoch_g_loss = 0.0
            epoch_c_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples - batch_size, batch_size):
                idx = perm[start : start + batch_size]
                real_batch = real_data[idx]
                label_batch = labels[idx]
                cond = torch.nn.functional.one_hot(label_batch, self.condition_dim).float()

                # -- Critic update --
                for _ in range(n_critic):
                    noise = torch.randn(batch_size, self.noise_dim, device=self.device)
                    fake_batch = self.generator(noise, cond).detach()

                    c_real = self.critic(real_batch, cond).mean()
                    c_fake = self.critic(fake_batch, cond).mean()

                    # Gradient penalty
                    gp = self._gradient_penalty(real_batch, fake_batch, cond)

                    c_loss = c_fake - c_real + gp_lambda * gp
                    opt_c.zero_grad()
                    c_loss.backward()
                    opt_c.step()

                # -- Generator update --
                noise = torch.randn(batch_size, self.noise_dim, device=self.device)
                fake_batch = self.generator(noise, cond)
                g_loss = -self.critic(fake_batch, cond).mean()

                opt_g.zero_grad()
                g_loss.backward()
                opt_g.step()

                epoch_g_loss += g_loss.item()
                epoch_c_loss += c_loss.item()
                n_batches += 1

            if n_batches > 0:
                history["g_loss"].append(epoch_g_loss / n_batches)
                history["c_loss"].append(epoch_c_loss / n_batches)

            if (epoch + 1) % 50 == 0:
                logger.info(
                    "Epoch %d/%d | G_loss=%.4f | C_loss=%.4f",
                    epoch + 1, n_epochs,
                    history["g_loss"][-1], history["c_loss"][-1],
                )

        self.is_trained = True
        logger.info("MarketSimulator training complete")
        return history

    def _gradient_penalty(
        self,
        real: torch.Tensor,
        fake: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """WGAN-GP gradient penalty on interpolated samples."""
        alpha = torch.rand(real.size(0), 1, 1, device=self.device)
        interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)

        c_interp = self.critic(interpolated, condition)

        gradients = torch.autograd.grad(
            outputs=c_interp,
            inputs=interpolated,
            grad_outputs=torch.ones_like(c_interp),
            create_graph=True,
            retain_graph=True,
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        gp = ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()
        return gp

    # ── Generation ───────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        regime: int,
        n_sequences: int = 1,
    ) -> np.ndarray:
        """Generate synthetic return sequences for a given regime.

        Args:
            regime: 0=Bull, 1=Bear, 2=Sideways.
            n_sequences: Number of sequences to generate.

        Returns:
            Array of shape (n_sequences, seq_length, num_sectors).
        """
        self.generator.eval()
        cond = torch.zeros(n_sequences, self.condition_dim, device=self.device)
        cond[:, regime] = 1.0
        noise = torch.randn(n_sequences, self.noise_dim, device=self.device)

        if self.is_trained:
            raw = self.generator(noise, cond).cpu().numpy()
        else:
            # Untrained fallback: use statistical model
            raw = noise[:, :self.seq_length * self.num_sectors].cpu().numpy()
            raw = raw.reshape(n_sequences, self.seq_length, self.num_sectors)

        # Scale to realistic return magnitudes
        scale = self.return_scale[regime]
        bias = self.return_bias[regime]
        returns = raw * scale + bias

        return returns.astype(np.float32)

    # ── Predefined Stress Scenarios ──────────────────────────────────

    def generate_circuit_breaker_scenario(self, n_sequences: int = 1) -> np.ndarray:
        """Simulate an NSE circuit breaker event: sharp drop followed by halt.

        Level 1: -10% index decline → 45 min halt
        Level 2: -15% → 1h 45min halt
        Level 3: -20% → trading halted for day
        """
        sequences = []
        for _ in range(n_sequences):
            seq = np.zeros((self.seq_length, self.num_sectors), dtype=np.float32)
            # Day 1-2: gradual decline
            seq[0] = np.random.uniform(-0.03, -0.01, self.num_sectors)
            seq[1] = np.random.uniform(-0.05, -0.02, self.num_sectors)
            # Day 3: circuit breaker hit
            seq[2] = np.random.uniform(-0.10, -0.07, self.num_sectors)
            # Day 4-5: dead cat bounce or continued fall
            seq[3] = np.random.uniform(-0.02, 0.03, self.num_sectors)
            seq[4] = np.random.uniform(-0.03, 0.02, self.num_sectors)
            sequences.append(seq)
        return np.array(sequences, dtype=np.float32)

    def generate_expiry_squeeze_scenario(self, n_sequences: int = 1) -> np.ndarray:
        """Simulate F&O expiry week volatility spike.

        Expiry weeks see elevated volatility due to rollover and short covering.
        """
        sequences = []
        for _ in range(n_sequences):
            # Higher variance, mean-reverting pattern
            base = np.random.normal(0, 0.02, (self.seq_length, self.num_sectors))
            # Spike on expiry day (last day)
            base[-1] *= 2.5
            # Add sector-specific effects (Banking sector more affected)
            base[:, 1] *= 1.5  # Financials
            sequences.append(base.astype(np.float32))
        return np.array(sequences, dtype=np.float32)

    def generate_flash_crash_scenario(self, n_sequences: int = 1) -> np.ndarray:
        """Simulate a flash crash: massive intraday drop with partial recovery."""
        sequences = []
        for _ in range(n_sequences):
            seq = np.zeros((self.seq_length, self.num_sectors), dtype=np.float32)
            # Normal days
            seq[0] = np.random.normal(0.001, 0.01, self.num_sectors)
            seq[1] = np.random.normal(0, 0.008, self.num_sectors)
            # Flash crash day
            seq[2] = np.random.uniform(-0.08, -0.04, self.num_sectors)
            # Partial recovery
            seq[3] = np.random.uniform(0.01, 0.05, self.num_sectors)
            seq[4] = np.random.uniform(0.005, 0.02, self.num_sectors)
            sequences.append(seq)
        return np.array(sequences, dtype=np.float32)

    def generate_sector_rotation_scenario(self, n_sequences: int = 1) -> np.ndarray:
        """Simulate sector rotation: leaders become laggards and vice versa."""
        sequences = []
        for _ in range(n_sequences):
            seq = np.zeros((self.seq_length, self.num_sectors), dtype=np.float32)
            # First half: some sectors outperform
            leaders = np.random.choice(self.num_sectors, size=4, replace=False)
            for d in range(2):
                seq[d, leaders] = np.random.uniform(0.01, 0.03, len(leaders))
                mask = np.ones(self.num_sectors, dtype=bool)
                mask[leaders] = False
                seq[d, mask] = np.random.uniform(-0.02, 0.0, mask.sum())
            # Rotation: laggards catch up, leaders fall back
            for d in range(2, self.seq_length):
                seq[d, leaders] = np.random.uniform(-0.02, 0.0, len(leaders))
                mask = np.ones(self.num_sectors, dtype=bool)
                mask[leaders] = False
                seq[d, mask] = np.random.uniform(0.01, 0.03, mask.sum())
            sequences.append(seq)
        return np.array(sequences, dtype=np.float32)

    def generate_covid_like_crash(self, n_sequences: int = 1) -> np.ndarray:
        """Simulate a pandemic-style crash: multi-week sustained decline."""
        sequences = []
        for _ in range(n_sequences):
            # Multi-week extreme decline
            seq = np.random.normal(-0.04, 0.02, (self.seq_length, self.num_sectors))
            # Pharma outperforms during pandemic
            seq[:, 3] = np.random.normal(0.01, 0.015, self.seq_length)
            # IT resilient (WFH theme)
            seq[:, 0] = np.random.normal(-0.01, 0.015, self.seq_length)
            sequences.append(seq.astype(np.float32))
        return np.array(sequences, dtype=np.float32)

    # ── Curriculum Data Builder ──────────────────────────────────────

    def build_curriculum_data(
        self,
        n_easy: int = 100,
        n_medium: int = 100,
        n_hard: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build curriculum training data from easy to hard.

        Easy: Bull market sequences
        Medium: Sideways + sector rotation
        Hard: Bear + circuit breakers + flash crashes

        Returns:
            (returns_data, difficulty_labels) where difficulty is 0=easy, 1=medium, 2=hard
        """
        easy_data = self.generate(regime=0, n_sequences=n_easy)
        easy_labels = np.zeros(n_easy, dtype=np.int64)

        medium_sw = self.generate(regime=2, n_sequences=n_medium // 2)
        medium_rot = self.generate_sector_rotation_scenario(n_medium - n_medium // 2)
        medium_data = np.concatenate([medium_sw, medium_rot], axis=0)
        medium_labels = np.ones(len(medium_data), dtype=np.int64)

        n_cb = n_hard // 3
        n_fc = n_hard // 3
        n_bear = n_hard - n_cb - n_fc
        hard_bear = self.generate(regime=1, n_sequences=n_bear)
        hard_cb = self.generate_circuit_breaker_scenario(n_cb)
        hard_fc = self.generate_flash_crash_scenario(n_fc)
        hard_data = np.concatenate([hard_bear, hard_cb, hard_fc], axis=0)
        hard_labels = np.full(len(hard_data), 2, dtype=np.int64)

        all_data = np.concatenate([easy_data, medium_data, hard_data], axis=0)
        all_labels = np.concatenate([easy_labels, medium_labels, hard_labels])

        logger.info(
            "Curriculum data built | easy=%d | medium=%d | hard=%d",
            n_easy, len(medium_data), len(hard_data),
        )
        return all_data, all_labels

    # ── Persistence ──────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save trained GAN weights."""
        torch.save({
            "generator": self.generator.state_dict(),
            "critic": self.critic.state_dict(),
            "is_trained": self.is_trained,
        }, path)
        logger.info("MarketSimulator saved to %s", path)

    def load(self, path: str) -> None:
        """Load trained GAN weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(checkpoint["generator"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.is_trained = checkpoint.get("is_trained", True)
        logger.info("MarketSimulator loaded from %s", path)
