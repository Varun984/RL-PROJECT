"""
File: training_data_loader.py
Module: data
Description: Loads collected Parquet data and prepares it for training.
    Converts raw market/macro/fundamental/news data into training-ready format
    for both Macro and Micro agents.
Author: HRL-SARP Framework
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class TrainingDataLoader:
    """Loads and prepares training data from collected Parquet files."""

    def __init__(self, config_path: str = "config/data_config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        
        # Resolve data_dir relative to config (handles different run locations)
        if os.path.isabs(config_path):
            project_root = os.path.dirname(os.path.dirname(config_path))
            self.data_dir = os.path.join(project_root, "data", "raw")
        else:
            self.data_dir = "data/raw"
        self.sectors = list(self.config["sectors"]["mapping"].keys())
        self.num_sectors = len(self.sectors)
        
        logger.info("TrainingDataLoader initialized | sectors=%d", self.num_sectors)

    def load_macro_training_data(
        self,
        start_date: str,
        end_date: str,
    ) -> Dict[str, np.ndarray]:
        """Load data for Macro agent pre-training.
        
        Returns:
            Dict with:
                - macro_states: (N, 18) - Macro features
                - sector_embeddings: (N, 11, 64) - Sector embeddings (dummy for now)
                - sector_returns: (N, 11) - Next-period sector returns
                - regime_labels: (N,) - Regime labels (0=Bear, 1=Bull, 2=Sideways)
        """
        logger.info("Loading macro training data: %s to %s", start_date, end_date)
        
        # Load sector index data
        sector_returns_list = []
        for sector in self.sectors:
            sector_file = self._get_sector_file(sector)
            if os.path.exists(sector_file):
                df = pd.read_parquet(sector_file)
                # Set date as index if it's a column
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                elif 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df = df.set_index('Date')
                # Convert index to datetime if needed
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                # Filter by date range
                df = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]
                if not df.empty:
                    # Use lowercase 'close' column
                    close_col = 'close' if 'close' in df.columns else 'Close'
                    returns = df[close_col].pct_change().fillna(0).values
                    sector_returns_list.append(returns)
        
        if not sector_returns_list:
            raise ValueError("No sector data found")
        
        # Align all sectors to same length (use shortest)
        min_len = min(len(r) for r in sector_returns_list)
        sector_returns = np.array([r[:min_len] for r in sector_returns_list]).T  # (N, 11)
        
        N = len(sector_returns)
        logger.info("Loaded sector returns: shape=%s", sector_returns.shape)
        
        # Load macro indicators
        macro_file = os.path.join(self.data_dir, "macro", "macro_indicators.parquet")
        if os.path.exists(macro_file):
            macro_df = pd.read_parquet(macro_file)
            # Set date as index if it's a column
            if 'date' in macro_df.columns:
                macro_df['date'] = pd.to_datetime(macro_df['date'])
                macro_df = macro_df.set_index('date')
            elif 'Date' in macro_df.columns:
                macro_df['Date'] = pd.to_datetime(macro_df['Date'])
                macro_df = macro_df.set_index('Date')
            # Convert index to datetime if needed
            if not isinstance(macro_df.index, pd.DatetimeIndex):
                macro_df.index = pd.to_datetime(macro_df.index)
            macro_df = macro_df[(macro_df.index >= pd.to_datetime(start_date)) & (macro_df.index <= pd.to_datetime(end_date))]
            
            # Extract macro features (18D)
            macro_states = self._extract_macro_features(macro_df, N)
        else:
            logger.warning("No macro data found, using dummy features")
            macro_states = np.random.randn(N, 18).astype(np.float32)
        
        # Build deterministic sector embeddings from return dynamics.
        # Using random embeddings here injects pure noise into Macro state.
        sector_embeddings = self._build_sector_embeddings(sector_returns)
        
        # Generate regime labels based on market returns
        regime_labels = self._generate_regime_labels(sector_returns)
        
        return {
            "macro_states": macro_states.astype(np.float32),
            "sector_embeddings": sector_embeddings.astype(np.float32),
            "sector_returns": sector_returns.astype(np.float32),
            "regime_labels": regime_labels.astype(np.int64),
        }

    def load_micro_training_data(
        self,
        start_date: str,
        end_date: str,
        max_stocks: int = 50,
    ) -> Dict[str, np.ndarray]:
        """Load data for Micro agent pre-training.
        
        Returns:
            Dict with:
                - stock_returns: (N, max_stocks) - Stock returns
                - stock_features: (N, max_stocks, 22) - Stock features
                - stock_to_sector: (max_stocks,) - Sector index for each stock
                - stock_masks: (N, max_stocks) - Valid stock mask
        """
        logger.info("Loading micro training data: %s to %s", start_date, end_date)
        
        # Collect all stocks from all sectors
        all_stocks = []
        stock_to_sector = []
        
        for sector_idx, sector in enumerate(self.sectors):
            sector_file = self._get_sector_file(sector)
            if os.path.exists(sector_file):
                df = pd.read_parquet(sector_file)
                # Set date as index if it's a column
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                elif 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df = df.set_index('Date')
                # Convert index to datetime if needed
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                df = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]
                
                # Check for symbol column (case insensitive)
                symbol_col = 'symbol' if 'symbol' in df.columns else 'Symbol' if 'Symbol' in df.columns else None
                if not df.empty and symbol_col:
                    symbols = df[symbol_col].unique()
                    for symbol in symbols[:10]:  # Limit per sector
                        all_stocks.append((symbol, sector_idx, sector))
        
        logger.info("Found %d stocks across %d sectors", len(all_stocks), self.num_sectors)
        
        # Limit to max_stocks
        all_stocks = all_stocks[:max_stocks]
        actual_num_stocks = len(all_stocks)
        
        # Load stock data
        stock_returns_list = []
        stock_features_list = []
        stock_to_sector_list = []
        
        for symbol, sector_idx, sector in all_stocks:
            sector_file = self._get_sector_file(sector)
            df = pd.read_parquet(sector_file)
            # Set date as index if it's a column
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
            elif 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date')
            # Convert index to datetime if needed
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            df = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]
            
            # Check for symbol column (case insensitive)
            symbol_col = 'symbol' if 'symbol' in df.columns else 'Symbol' if 'Symbol' in df.columns else None
            if symbol_col:
                stock_df = df[df[symbol_col] == symbol]
            else:
                stock_df = df
            
            if not stock_df.empty:
                # Use lowercase column names
                close_col = 'close' if 'close' in stock_df.columns else 'Close'
                returns = stock_df[close_col].pct_change().fillna(0).values
                features = self._extract_stock_features(stock_df)
                
                stock_returns_list.append(returns)
                stock_features_list.append(features)
                stock_to_sector_list.append(sector_idx)
        
        if not stock_returns_list:
            raise ValueError("No stock data found")
        
        # Align to same length
        min_len = min(len(r) for r in stock_returns_list)
        N = min_len
        
        # Pad to max_stocks
        stock_returns = np.zeros((N, max_stocks), dtype=np.float32)
        stock_features = np.zeros((N, max_stocks, 22), dtype=np.float32)
        stock_masks = np.zeros((N, max_stocks), dtype=np.float32)
        stock_to_sector = np.zeros(max_stocks, dtype=np.int64)
        
        for i, (returns, features, sector_idx) in enumerate(zip(
            stock_returns_list, stock_features_list, stock_to_sector_list
        )):
            stock_returns[:, i] = returns[:N]
            stock_features[:, i, :] = features[:N, :]
            stock_masks[:, i] = 1.0
            stock_to_sector[i] = sector_idx
        
        logger.info("Loaded stock data: returns=%s, features=%s", 
                   stock_returns.shape, stock_features.shape)
        
        return {
            "stock_returns": stock_returns,
            "stock_features": stock_features,
            "stock_to_sector": stock_to_sector,
            "stock_masks": stock_masks,
        }

    def _get_sector_file(self, sector: str) -> str:
        """Get the parquet file path for a sector."""
        sector_map = {
            "IT": "it_ohlcv.parquet",
            "Financials": "financials_ohlcv.parquet",
            "Auto": "auto_ohlcv.parquet",
            "Pharma": "pharma_ohlcv.parquet",
            "FMCG": "fmcg_ohlcv.parquet",
            "Energy": "energy_ohlcv.parquet",
            "Metals": "metals_ohlcv.parquet",
            "Realty": "realty_ohlcv.parquet",
            "Media": "media_ohlcv.parquet",
            "Telecom": "telecom_ohlcv.parquet",
            "Infra": "infra_ohlcv.parquet",
        }
        filename = sector_map.get(sector, f"{sector.lower()}_ohlcv.parquet")
        return os.path.join(self.data_dir, "market", filename)

    def _extract_macro_features(self, macro_df: pd.DataFrame, target_len: int) -> np.ndarray:
        """Extract 18D macro feature vector."""
        # For now, use available columns and pad/repeat to match target length
        features = []
        
        for col in macro_df.columns:
            if col != "Date":
                values = macro_df[col].fillna(0).values
                # Interpolate to target length
                if len(values) < target_len:
                    # Repeat last value
                    values = np.pad(values, (0, target_len - len(values)), mode='edge')
                elif len(values) > target_len:
                    # Sample evenly
                    indices = np.linspace(0, len(values) - 1, target_len, dtype=int)
                    values = values[indices]
                features.append(values)
        
        # Pad to 18 features if needed
        while len(features) < 18:
            features.append(np.zeros(target_len))
        
        features = features[:18]  # Take first 18
        return np.array(features).T  # (N, 18)

    def _extract_stock_features(self, stock_df: pd.DataFrame) -> np.ndarray:
        """Extract 22D stock feature vector."""
        N = len(stock_df)
        features = np.zeros((N, 22), dtype=np.float32)
        
        # Use lowercase column names
        close_col = 'close' if 'close' in stock_df.columns else 'Close'
        high_col = 'high' if 'high' in stock_df.columns else 'High'
        low_col = 'low' if 'low' in stock_df.columns else 'Low'
        volume_col = 'volume' if 'volume' in stock_df.columns else 'Volume'
        
        # Price-based features
        close = stock_df[close_col].values
        high = stock_df[high_col].values
        low = stock_df[low_col].values
        volume = stock_df[volume_col].values
        
        # Returns (features 0-2)
        features[:, 0] = np.concatenate([[0], np.diff(close) / (close[:-1] + 1e-8)])  # 1d return
        features[:, 1] = self._rolling_return(close, 5)  # 5d return
        features[:, 2] = self._rolling_return(close, 20)  # 20d return
        
        # Volatility (feature 3)
        features[:, 3] = self._rolling_std(close, 20)
        
        # Technical indicators (features 4-7)
        features[:, 4] = self._compute_rsi(close, 14)
        features[:, 5] = np.random.randn(N) * 0.1  # MACD placeholder
        features[:, 6] = (close - low) / (high - low + 1e-8)  # BB position proxy
        features[:, 7] = volume / (self._rolling_mean(volume, 20) + 1e-8)  # Volume ratio
        
        # Fundamental features (features 8-16) - would come from fundamentals data
        features[:, 8:17] = np.random.randn(N, 9) * 0.1  # Placeholder
        
        # Market cap and other (features 17-21)
        features[:, 17:22] = np.random.randn(N, 5) * 0.1  # Placeholder
        
        return features

    def _generate_regime_labels(self, sector_returns: np.ndarray) -> np.ndarray:
        """Generate regime labels based on market returns.
        
        Simple heuristic:
        - Bull: avg return > 0.5%
        - Bear: avg return < -0.5%
        - Sideways: otherwise
        """
        avg_returns = sector_returns.mean(axis=1)
        labels = np.ones(len(avg_returns), dtype=np.int64) * 2  # Default: Sideways
        labels[avg_returns > 0.005] = 1  # Bull
        labels[avg_returns < -0.005] = 0  # Bear
        return labels

    def _build_sector_embeddings(self, sector_returns: np.ndarray) -> np.ndarray:
        """Construct simple deterministic 64D embeddings per sector and timestep.

        This is a placeholder until GNN embeddings are integrated. It encodes each
        sector's local return dynamics using rolling moments, then tiles to 64 dims.
        """
        N, S = sector_returns.shape
        out = np.zeros((N, S, 64), dtype=np.float32)

        for s in range(S):
            r = sector_returns[:, s].astype(np.float32)
            roll5_mean = self._rolling_mean(r, 5).astype(np.float32)
            roll20_mean = self._rolling_mean(r, 20).astype(np.float32)
            roll20_std = self._rolling_std(r, 20).astype(np.float32)

            base = np.stack([r, roll5_mean, roll20_mean, roll20_std], axis=1)  # (N, 4)
            tiled = np.tile(base, (1, 16))  # (N, 64)
            out[:, s, :] = tiled

        return out

    @staticmethod
    def _rolling_return(prices: np.ndarray, window: int) -> np.ndarray:
        """Compute rolling return."""
        result = np.zeros_like(prices)
        for i in range(window, len(prices)):
            result[i] = (prices[i] - prices[i - window]) / (prices[i - window] + 1e-8)
        return result

    @staticmethod
    def _rolling_std(prices: np.ndarray, window: int) -> np.ndarray:
        """Compute rolling standard deviation."""
        result = np.zeros_like(prices)
        for i in range(window, len(prices)):
            result[i] = np.std(prices[i - window:i])
        return result

    @staticmethod
    def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
        """Compute rolling mean."""
        result = np.zeros_like(values)
        for i in range(window, len(values)):
            result[i] = np.mean(values[i - window:i])
        result[:window] = values[:window]  # Fill initial values
        return result

    @staticmethod
    def _compute_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Compute RSI indicator."""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.zeros(len(prices))
        avg_losses = np.zeros(len(prices))
        
        for i in range(period, len(prices)):
            avg_gains[i] = np.mean(gains[i - period:i])
            avg_losses[i] = np.mean(losses[i - period:i])
        
        rs = avg_gains / (avg_losses + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi
