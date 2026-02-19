"""
File: regime_detector.py
Module: agents
Description: Hidden Markov Model for market regime classification trained on
    India VIX + Nifty return features. Identifies Bull/Bear/Sideways regimes
    for supervised labelling and Macro agent regime prediction targets.
Design Decisions: HMM is well-suited for regime detection because market states
    are latent (unobserved) and exhibit temporal persistence. Using VIX + returns
    as observables captures both volatility and directional regimes.
References: Hamilton (1989) regime-switching models, hmmlearn library
Author: HRL-SARP Framework
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import yaml

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# REGIME DETECTOR
# ══════════════════════════════════════════════════════════════════════


class RegimeDetector:
    """HMM-based market regime classifier for Indian equity markets.

    Fits a Gaussian HMM with 3 states on [VIX_level, VIX_change, Nifty_return,
    Nifty_volatility] features. After fitting, labels the states as
    Bull (0), Bear (1), or Sideways (2) based on emission means.
    """

    def __init__(
        self,
        n_regimes: int = 3,
        n_features: int = 4,
        covariance_type: str = "full",
        n_iter: int = 100,
        random_state: int = 42,
        config_path: str = "config/macro_agent_config.yaml",
    ) -> None:
        self.n_regimes = n_regimes
        self.n_features = n_features
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.regime_cfg = self.config["regime"]
        self.bull_threshold: float = self.regime_cfg["bull_threshold"]
        self.bear_threshold: float = self.regime_cfg["bear_threshold"]

        self.model = None
        self.state_mapping: Dict[int, int] = {}
        self.is_fitted: bool = False

        # Feature statistics for normalisation
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_std: Optional[np.ndarray] = None

        logger.info(
            "RegimeDetector initialised | n_regimes=%d | features=%d",
            n_regimes, n_features,
        )

    def _init_model(self) -> None:
        """Lazy import and initialise HMM (hmmlearn may not be installed)."""
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            logger.warning("hmmlearn not installed; using threshold-based fallback")
            self.model = None
            return

        self.model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )

    def fit(
        self,
        features: np.ndarray,
        lengths: Optional[list] = None,
    ) -> Dict[str, any]:
        """Fit the HMM on historical feature data.

        Args:
            features: (T, n_features) array of [VIX, VIX_change, nifty_return, nifty_vol].
            lengths: Optional list of sequence lengths if multiple episodes.

        Returns:
            Dict with training info (convergence, state means, transition matrix).
        """
        self._init_model()

        # Normalise features
        self.feature_mean = features.mean(axis=0)
        self.feature_std = features.std(axis=0) + 1e-8
        features_norm = (features - self.feature_mean) / self.feature_std

        if self.model is not None:
            self.model.fit(features_norm, lengths=lengths)
            self.is_fitted = True

            # Decode most likely state sequence
            hidden_states = self.model.predict(features_norm)

            # Map HMM states to semantic labels based on emission means
            self._assign_state_labels(features, hidden_states)

            info = {
                "converged": self.model.monitor_.converged,
                "n_iter": self.model.monitor_.iter,
                "log_likelihood": float(self.model.score(features_norm)),
                "transition_matrix": self.model.transmat_.tolist(),
                "state_mapping": self.state_mapping,
            }
        else:
            # Fallback: threshold-based classification
            self.is_fitted = True
            self.state_mapping = {0: 0, 1: 1, 2: 2}
            info = {"method": "threshold_fallback"}

        logger.info("RegimeDetector fitted | %s", info)
        return info

    def _assign_state_labels(
        self, raw_features: np.ndarray, hidden_states: np.ndarray
    ) -> None:
        """Map HMM latent states to Bull/Bear/Sideways using feature statistics.

        Logic: compute mean Nifty return per state. Highest return → Bull,
        lowest → Bear, middle → Sideways.
        """
        return_idx = 2  # nifty_return is 3rd feature
        state_mean_returns = {}

        for s in range(self.n_regimes):
            mask = hidden_states == s
            if mask.sum() > 0:
                state_mean_returns[s] = raw_features[mask, return_idx].mean()
            else:
                state_mean_returns[s] = 0.0

        # Sort by mean return: highest → Bull(0), lowest → Bear(1), middle → Sideways(2)
        sorted_states = sorted(state_mean_returns.items(), key=lambda x: x[1], reverse=True)

        self.state_mapping = {
            sorted_states[0][0]: 0,  # Bull
            sorted_states[1][0]: 2,  # Sideways
            sorted_states[2][0]: 1,  # Bear
        }

        logger.info(
            "State mapping: %s | Mean returns: %s",
            self.state_mapping,
            {self.state_mapping[s]: f"{r:.4f}" for s, r in state_mean_returns.items()},
        )

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict regime labels for a sequence of observations.

        Args:
            features: (T, n_features)

        Returns:
            Array of regime labels (T,) with values in {0=Bull, 1=Bear, 2=Sideways}.
        """
        if not self.is_fitted:
            raise RuntimeError("RegimeDetector not fitted. Call fit() first.")

        if self.model is not None:
            features_norm = (features - self.feature_mean) / self.feature_std
            raw_states = self.model.predict(features_norm)
            # Map HMM states to semantic labels
            mapped = np.array([self.state_mapping.get(s, 2) for s in raw_states])
            return mapped
        else:
            return self._threshold_predict(features)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict regime probabilities for each timestep.

        Returns:
            (T, 3) array of [P(Bull), P(Bear), P(Sideways)].
        """
        if not self.is_fitted:
            raise RuntimeError("RegimeDetector not fitted. Call fit() first.")

        if self.model is not None:
            features_norm = (features - self.feature_mean) / self.feature_std
            raw_proba = self.model.predict_proba(features_norm)  # (T, n_regimes)

            # Remap columns to semantic order
            mapped_proba = np.zeros_like(raw_proba)
            for hmm_state, semantic_label in self.state_mapping.items():
                mapped_proba[:, semantic_label] = raw_proba[:, hmm_state]

            return mapped_proba
        else:
            labels = self._threshold_predict(features)
            proba = np.zeros((len(labels), 3), dtype=np.float32)
            for i, l in enumerate(labels):
                proba[i, l] = 0.8
                # Spread remaining probability
                others = [j for j in range(3) if j != l]
                proba[i, others[0]] = 0.1
                proba[i, others[1]] = 0.1
            return proba

    def predict_single(self, features: np.ndarray) -> Tuple[int, np.ndarray]:
        """Predict regime for a single observation.

        Args:
            features: (n_features,) single observation.

        Returns:
            (regime_label, probabilities)
        """
        features_2d = features.reshape(1, -1)
        label = self.predict(features_2d)[0]
        proba = self.predict_proba(features_2d)[0]
        return int(label), proba

    def _threshold_predict(self, features: np.ndarray) -> np.ndarray:
        """Simple threshold-based fallback when hmmlearn is unavailable.

        Uses Nifty return (feature index 2) with config thresholds.
        """
        returns = features[:, 2] if features.ndim > 1 else features
        labels = np.full(len(returns), 2, dtype=np.int64)  # Default: Sideways
        labels[returns > self.bull_threshold] = 0  # Bull
        labels[returns < self.bear_threshold] = 1  # Bear
        return labels

    def get_regime_label(self, regime_id: int) -> str:
        """Convert numeric regime to string label."""
        labels = {0: "Bull", 1: "Bear", 2: "Sideways"}
        return labels.get(regime_id, "Unknown")

    def generate_regime_labels(
        self, features: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate regime labels and probabilities for a full dataset.

        Returns:
            (labels, probabilities) — labels shape (T,), probabilities shape (T, 3)
        """
        labels = self.predict(features)
        proba = self.predict_proba(features)
        return labels, proba

    # ── Persistence ──────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save fitted model to disk."""
        import pickle
        state = {
            "model": self.model,
            "state_mapping": self.state_mapping,
            "feature_mean": self.feature_mean,
            "feature_std": self.feature_std,
            "is_fitted": self.is_fitted,
            "n_regimes": self.n_regimes,
            "n_features": self.n_features,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info("RegimeDetector saved to %s", path)

    def load(self, path: str) -> None:
        """Load fitted model from disk."""
        import pickle
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.model = state["model"]
        self.state_mapping = state["state_mapping"]
        self.feature_mean = state["feature_mean"]
        self.feature_std = state["feature_std"]
        self.is_fitted = state["is_fitted"]
        logger.info("RegimeDetector loaded from %s", path)
