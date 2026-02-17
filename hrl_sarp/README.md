# HRL-SARP: Hierarchical Reinforcement Learning for Sector-Aware Risk-Adaptive Portfolio Management

## Indian Equity Markets Edition

---

## Abstract

**HRL-SARP** is a two-agent hierarchical reinforcement learning framework designed for
intelligent portfolio management in Indian equity markets. The system decomposes the
complex portfolio optimization problem into two temporally distinct decision layers:

1. **Macro Agent** (weekly): Processes macroeconomic signals, news sentiment, FII/DII
   capital flows, sector momentum, and F&O derivatives data to output optimal sector
   allocation weights and a volatility regime classification (Bull / Bear / Sideways).

2. **Micro Agent** (daily): Receives the Macro agent's goal embedding (sector weights +
   regime label) and selects fundamentally undervalued stocks within the favoured sectors,
   sizing positions via a Conditional Value-at-Risk (CVaR) constrained risk budget.

The framework incorporates India-specific financial constraints (STT, SEBI regulations,
NSE/BSE trading calendars, circuit breaker rules) and employs advanced techniques including
Graph Neural Networks for sector correlation modelling, FinBERT-India for financial
sentiment analysis, and Hindsight Experience Replay (HER) for sample-efficient goal-
conditioned learning.

---

## Motivation

Traditional portfolio management approaches in Indian markets suffer from:

- **Static allocation**: Mean-variance optimization ignores regime changes (e.g., 2020 COVID crash, 2022 rate hike cycle).
- **Flat RL agents**: Single-agent approaches struggle with the multi-timescale nature of sector rotation (weekly) vs. stock selection (daily).
- **Western-centric models**: Most RL-for-finance research targets US/EU markets, ignoring India-specific factors like FII/DII flows, STT impact, and NSE sectoral indices.
- **Lack of explainability**: Black-box RL models face regulatory scrutiny under SEBI's evolving AI governance framework.

HRL-SARP addresses all four gaps through hierarchical temporal abstraction, India-specific
feature engineering, and SHAP-based post-hoc explainability.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          HRL-SARP SYSTEM ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌──────────────────────── DATA LAYER ─────────────────────────┐               │
│  │  Market Data │ Fundamentals │ News/Sentiment │ Macro Signals │               │
│  │  (Zerodha/   │ (Screener.in │ (ET, MCcontrol │ (RBI, VIX,   │               │
│  │   yfinance)  │  Trendlyne)  │  BSE filings)  │  FII/DII)    │               │
│  └──────┬───────┴──────┬───────┴───────┬────────┴──────┬───────┘               │
│         │              │               │               │                        │
│         ▼              ▼               ▼               ▼                        │
│  ┌──────────────────── FEATURE ENGINEERING ─────────────────────┐               │
│  │  Technical  │ Fundamental │ Sentiment    │ Macro     │ Sector │              │
│  │  (RSI,MACD, │ (P/E z-score│ (FinBERT-    │ (VIX,PCR, │ (Rel.  │              │
│  │   BB,Vol%)  │  ROE,D/E)   │  India)      │  FII norm) │ Str.)  │              │
│  └──────┬──────┴──────┬──────┴──────┬───────┴──────┬────┴───┬───┘              │
│         │             │             │              │        │                    │
│         ▼             ▼             ▼              ▼        ▼                    │
│  ┌────────────────────────────────────────────────────────────────┐             │
│  │                     FEATURE STORE (SQLite/TimescaleDB)         │             │
│  │              Timestamp-gated reads (no lookahead bias)         │             │
│  └──────────────────────────┬────────────────────────────────────┘             │
│                             │                                                   │
│         ┌───────────────────┴───────────────────┐                              │
│         ▼                                       ▼                              │
│  ┌──────────────────┐                ┌────────────────────────┐                │
│  │   MACRO AGENT    │                │     SECTOR GNN         │                │
│  │   (PPO-based)    │◄───────────────│   (GCN/GAT Encoder)    │                │
│  │                  │  sector node   │   Dynamic correlation  │                │
│  │  ┌────────────┐  │  embeddings    │   graph (60D rolling)  │                │
│  │  │ Actor:     │  │                └────────────────────────┘                │
│  │  │ Sector Wts │  │                                                          │
│  │  │ + Regime   │  │  Weekly cadence                                          │
│  │  ├────────────┤  │  ┌──────────────────────────────┐                        │
│  │  │ Critic:    │  │  │ Goal Embedding (64D):        │                        │
│  │  │ V(s)       │  │  │ sector_weights ⊕ regime_vec  │                        │
│  │  └────────────┘  │  └──────────────┬───────────────┘                        │
│  └──────────────────┘                 │                                         │
│                                       ▼                                         │
│  ┌──────────────────────────────────────────────────────────┐                  │
│  │                     MICRO AGENT (TD3 + HER)              │                  │
│  │                                                          │                  │
│  │  Input: stock_features (22D) ⊕ goal_embedding (64D)     │                  │
│  │                                                          │                  │
│  │  ┌─────────────┐  ┌──────────────────┐                  │                  │
│  │  │ Actor:      │  │ Twin Critics:    │  Daily cadence   │                  │
│  │  │ Stock Wts   │  │ Q1(s,a), Q2(s,a)│                  │                  │
│  │  └─────────────┘  └──────────────────┘                  │                  │
│  └──────────────────────────┬───────────────────────────────┘                  │
│                             │                                                   │
│                             ▼                                                   │
│  ┌──────────────────── RISK MANAGEMENT ────────────────────────┐               │
│  │  Max Drawdown Gate │ Sector Cap │ Liquidity Filter │ CVaR   │               │
│  │  Pledge Filter     │ Event Risk │ Transaction Costs (STT)   │               │
│  └──────────────────────────┬──────────────────────────────────┘               │
│                             │                                                   │
│                             ▼                                                   │
│  ┌──────────────────── EXECUTION LAYER ────────────────────────┐               │
│  │  Portfolio Rebalancing │ Walk-Forward Backtest │ Dashboard   │               │
│  └─────────────────────────────────────────────────────────────┘               │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## MDP Formulation

### Macro Agent MDP (Weekly)

| Component   | Definition |
|-------------|-----------|
| **State** `s_M` | `[India_VIX, FII_flow_norm, DII_flow_norm, PCR_idx, USD_INR_mom, Crude_regime, RBI_rate, Credit_spread, Yield_curve_slope, Nifty_return_4w, Nifty_vol_4w, Breadth_adv_dec, FnO_OI_change, sector_gnn_embeddings(11×64D), event_risk_flag, days_to_event, regime_label_prev, portfolio_drawdown]` |
| **Action** `a_M` | `w ∈ Δ^{10}` (simplex over 11 NSE sectors via softmax) + regime prediction `r ∈ {Bull, Bear, Sideways}` |
| **Reward** `R_M` | `0.4·R_macro + 0.3·R_portfolio + 0.2·R_regime + 0.1·R_value_bonus` |
| **Transition** | Weekly market dynamics; state observed every Friday close |
| **Discount** `γ` | 0.99 |

### Micro Agent MDP (Daily)

| Component   | Definition |
|-------------|-----------|
| **State** `s_μ` | `[stock_features(22D per stock)] ⊕ goal_embedding(64D)` |
| **Goal** `g` | `goal_embedding = f(sector_weights, regime_vector)` from Macro agent |
| **Action** `a_μ` | `w_stock ∈ Δ^{N}` (simplex over N stocks in sector-filtered universe) |
| **Reward** `R_μ` | `0.5·R_micro + 0.3·R_portfolio + 0.2·R_value_bonus` |
| **Transition** | Daily market dynamics; goal fixed within each weekly Macro cycle |
| **Discount** `γ` | 0.99 |

### Reward Components

```
R_macro     = sector_alpha_vs_nifty - 0.1·HHI(weights) - 0.001·turnover_cost
R_micro     = 0.5·Sharpe_weekly + 0.3·cos_sim(achieved_alloc, goal_alloc) - 0.2·max(0, DD-0.05)²
R_value     = return_week · 𝟙(PE_z < -0.5) · 𝟙(return > 0)
R_portfolio = Calmar_monthly · (1 - CVaR_95%) - STT_cost_week
R_regime    = +0.3 if predicted == realised, else -0.1
```

---

## Training Procedure

The training follows a **5-phase curriculum**:

| Phase | Description | Frozen | Trainable |
|-------|------------|--------|-----------|
| **1** | Supervised pre-training of Macro agent on labelled sector-winner episodes | — | Macro |
| **2** | Supervised pre-training of Micro agent using analyst consensus as weak labels | — | Micro |
| **3** | RL training of Micro agent with Macro frozen as oracle | Macro | Micro |
| **4** | RL training of Macro agent with Micro frozen | Micro | Macro |
| **5** | Alternating-gradient joint fine-tuning of both agents | — | Both (alternating) |

Curriculum difficulty progression: Bull market episodes → Mixed volatility → Stress scenarios (COVID, IL&FS, rate hikes).

---

## Setup Instructions

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended for GNN + FinBERT training)
- SQLite3 (bundled with Python) or TimescaleDB (optional, for production)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/hrl-sarp.git
cd hrl-sarp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# OR
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Configure API keys (edit data_config.yaml)
cp config/data_config.yaml config/data_config.local.yaml
# Edit config/data_config.local.yaml with your API keys
```

### Configuration

All hyperparameters and settings are in `config/`:
- `macro_agent_config.yaml` — Macro PPO hyperparameters
- `micro_agent_config.yaml` — Micro TD3 + HER hyperparameters
- `data_config.yaml` — Data sources, API keys, feature lists
- `risk_config.yaml` — Risk limits, SEBI constraints, event calendars

### Running

```bash
# Full training pipeline (all 5 phases)
bash scripts/run_full_training.sh

# Walk-forward backtest with report generation
bash scripts/run_backtest.sh

# Launch monitoring dashboard
bash scripts/run_dashboard.sh
```

---

## Key Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Macro RL Agent | PPO (from scratch) | Sector allocation under regime uncertainty |
| Micro RL Agent | TD3 + HER (from scratch) | Goal-conditioned stock selection |
| Sector Graph | PyTorch Geometric (GCN/GAT) | Dynamic sector correlation modelling |
| Sentiment | FinBERT fine-tuned on Indian corpus | Financial news sentiment scoring |
| Risk Mgmt | CVaR optimization | SEBI-aware constraint enforcement |
| Explainability | SHAP (DeepExplainer) | Per-trade feature attribution |
| Dashboard | Streamlit | Real-time portfolio monitoring |
| Logging | MLflow | Experiment tracking & reproducibility |

---

## NSE Sector Classification (Nifty 11 Sectors)

1. **Information Technology** (Nifty IT)
2. **Financial Services** (Nifty Financial Services)
3. **Automobile & Auto Components** (Nifty Auto)
4. **Healthcare / Pharma** (Nifty Pharma)
5. **Fast Moving Consumer Goods** (Nifty FMCG)
6. **Oil, Gas & Energy** (Nifty Energy)
7. **Metals & Mining** (Nifty Metal)
8. **Realty** (Nifty Realty)
9. **Media, Entertainment & Publication** (Nifty Media)
10. **Telecommunication** (Nifty Telecom / IT-Telecom)
11. **Infrastructure / Construction** (Nifty Infrastructure)

---

## Results (Placeholder)

| Metric | HRL-SARP | Nifty 50 B&H | Nifty QLV30 | Flat RL | Random |
|--------|----------|-------------|-------------|---------|--------|
| Annualised Return | — | — | — | — | — |
| Sharpe Ratio | — | — | — | — | — |
| Calmar Ratio | — | — | — | — | — |
| Max Drawdown | — | — | — | — | — |
| CVaR (95%) | — | — | — | — | — |
| Hit Rate | — | — | — | — | — |
| Turnover (annual) | — | — | — | — | — |

*Results to be populated after walk-forward backtest on 2018–2025 data.*

---

## Citation

```bibtex
@software{hrl_sarp_2025,
  title     = {HRL-SARP: Hierarchical Reinforcement Learning for Sector-Aware
               Risk-Adaptive Portfolio Management in Indian Equity Markets},
  author    = {HRL-SARP Framework Contributors},
  year      = {2025},
  url       = {https://github.com/your-org/hrl-sarp},
  note      = {Two-agent hierarchical RL system combining PPO (macro) and
               TD3+HER (micro) with GNN sector encoding and FinBERT-India
               sentiment analysis for NSE/BSE portfolio optimization}
}
```

---

## References

1. Schulman, J. et al. (2017). "Proximal Policy Optimization Algorithms." *arXiv:1707.06347*
2. Fujimoto, S. et al. (2018). "Addressing Function Approximation Error in Actor-Critic Methods." (TD3) *ICML 2018*
3. Andrychowicz, M. et al. (2017). "Hindsight Experience Replay." *NeurIPS 2017*
4. Kipf, T. N. & Welling, M. (2017). "Semi-Supervised Classification with Graph Convolutional Networks." *ICLR 2017*
5. Araci, D. (2019). "FinBERT: Financial Sentiment Analysis with Pre-Trained Language Models." *arXiv:1908.10063*
6. Lundberg, S. M. & Lee, S.-I. (2017). "A Unified Approach to Interpreting Model Predictions." (SHAP) *NeurIPS 2017*
7. Nachum, O. et al. (2018). "Data-Efficient Hierarchical Reinforcement Learning." *NeurIPS 2018*
8. Yang, H. et al. (2020). "Deep Reinforcement Learning for Automated Stock Trading." *ACM ICAIF 2020*

---

## License

MIT License. See `LICENSE` for details.

---

## Disclaimer

This software is for **research and educational purposes only**. It does not constitute
financial advice. Past performance does not guarantee future results. Trading in Indian
equity markets involves significant risk. Always consult with a SEBI-registered investment
advisor before making investment decisions.
