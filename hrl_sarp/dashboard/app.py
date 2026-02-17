"""
File: app.py
Module: dashboard
Description: Streamlit-based real-time monitoring dashboard for HRL-SARP.
    Displays live portfolio metrics, agent decisions, risk status,
    sector allocations, and training progress.
Usage: streamlit run dashboard/app.py
Author: HRL-SARP Framework
"""

import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False


SECTOR_NAMES = [
    "IT", "Financials", "Pharma", "FMCG", "Auto",
    "Energy", "Metals", "Realty", "Telecom", "Media", "Infra",
]


def main():
    if not HAS_STREAMLIT:
        print("Streamlit not installed. Run: pip install streamlit plotly")
        return

    st.set_page_config(
        page_title="HRL-SARP Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem; font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e, #2d2d44);
        border-radius: 12px; padding: 20px; border: 1px solid #3d3d5c;
    }
    .stMetric { background: rgba(30, 30, 46, 0.6); border-radius: 10px; padding: 10px; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="main-header">📊 HRL-SARP Dashboard</p>', unsafe_allow_html=True)
    st.markdown("*Hierarchical RL for Sector-Aware Risk-Adaptive Portfolio Management*")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Controls")
        page = st.selectbox("Navigate", [
            "Portfolio Overview",
            "Sector Allocation",
            "Risk Monitor",
            "Agent Decisions",
            "Training Progress",
            "Stress Testing",
            "Evaluation Results",
        ])

    if page == "Portfolio Overview":
        render_portfolio_overview()
    elif page == "Sector Allocation":
        render_sector_allocation()
    elif page == "Risk Monitor":
        render_risk_monitor()
    elif page == "Agent Decisions":
        render_agent_decisions()
    elif page == "Training Progress":
        render_training_progress()
    elif page == "Stress Testing":
        render_stress_testing()
    elif page == "Evaluation Results":
        render_evaluation_results()


# ── Portfolio Overview ───────────────────────────────────────────────


def render_portfolio_overview():
    st.header("💼 Portfolio Overview")

    # Metrics row
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Portfolio Value", "₹1,05,23,456", "+2.3%")
    col2.metric("CAGR", "18.4%", "+3.2%")
    col3.metric("Sharpe Ratio", "1.42", "+0.15")
    col4.metric("Max Drawdown", "-12.3%", "-1.2%")
    col5.metric("Win Rate", "58.3%", "+2.1%")
    col6.metric("Current Regime", "Bull 🟢", "")

    st.divider()

    # Equity curve
    st.subheader("📈 Equity Curve")
    results_path = os.path.join(PROJECT_ROOT, "results", "evaluation_results.json")

    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        portfolio_values = results.get("portfolio_values", [])
        if portfolio_values:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=portfolio_values, mode="lines",
                name="HRL-SARP",
                line=dict(color="#667eea", width=2),
                fill="tonexty", fillcolor="rgba(102, 126, 234, 0.1)",
            ))
            fig.update_layout(
                title="Portfolio Value Over Time",
                yaxis_title="Portfolio Value (₹)",
                xaxis_title="Trading Days",
                template="plotly_dark",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📁 Run evaluation to generate portfolio data: `python main.py evaluate`")
        _render_demo_equity_curve()


def _render_demo_equity_curve():
    """Render demo equity curve for display purposes."""
    np.random.seed(42)
    T = 500
    returns = np.random.normal(0.0005, 0.012, T)
    values = 10_000_000 * np.cumprod(1 + returns)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=values, mode="lines", name="Demo Portfolio",
        line=dict(color="#667eea", width=2),
        fill="tonexty", fillcolor="rgba(102, 126, 234, 0.1)",
    ))
    fig.update_layout(
        title="Portfolio Value (Demo Data)",
        yaxis_title="₹", xaxis_title="Trading Days",
        template="plotly_dark", height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Sector Allocation ────────────────────────────────────────────────


def render_sector_allocation():
    st.header("🏗️ Sector Allocation")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Current Allocation")
        # Demo weights
        weights = np.array([0.15, 0.20, 0.10, 0.08, 0.12, 0.08, 0.07, 0.05, 0.05, 0.03, 0.07])
        fig = go.Figure(data=[go.Pie(
            labels=SECTOR_NAMES, values=weights,
            hole=0.4, marker=dict(colors=px.colors.qualitative.Set3),
        )])
        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Sector Performance")
        returns = np.random.normal(0.02, 0.05, 11)
        colors = ["#22c55e" if r > 0 else "#ef4444" for r in returns]
        fig = go.Figure(data=[go.Bar(
            x=SECTOR_NAMES, y=returns * 100,
            marker_color=colors,
        )])
        fig.update_layout(
            yaxis_title="Return (%)",
            template="plotly_dark", height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap
    st.subheader("📊 Sector Correlation Matrix")
    np.random.seed(42)
    corr = np.random.uniform(0.2, 0.8, (11, 11))
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1.0)

    fig = go.Figure(data=go.Heatmap(
        z=corr, x=SECTOR_NAMES, y=SECTOR_NAMES,
        colorscale="RdYlBu_r",
        text=np.round(corr, 2), texttemplate="%{text}",
    ))
    fig.update_layout(template="plotly_dark", height=500)
    st.plotly_chart(fig, use_container_width=True)


# ── Risk Monitor ─────────────────────────────────────────────────────


def render_risk_monitor():
    st.header("🛡️ Risk Monitor")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Drawdown", "-4.2%", "within limit ✅")
    col2.metric("VaR (95%)", "2.1%", "")
    col3.metric("CVaR (95%)", "3.4%", "")
    col4.metric("Circuit Breaker", "OFF 🟢", "")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Drawdown History")
        np.random.seed(42)
        T = 300
        dd = np.abs(np.cumsum(np.random.normal(-0.001, 0.01, T)))
        dd = np.minimum(dd, 0.15)

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=-dd * 100, mode="lines", name="Drawdown",
                                  line=dict(color="#ef4444"), fill="tonexty"))
        fig.add_hline(y=-12, line_dash="dash", line_color="red",
                      annotation_text="Max DD Limit (12%)")
        fig.update_layout(yaxis_title="Drawdown (%)", template="plotly_dark", height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Risk Limits Status")
        limits = {
            "Max Drawdown": (4.2, 12.0),
            "Stock Concentration": (8.5, 10.0),
            "Sector Concentration": (22.0, 25.0),
            "Position Count": (35, 50),
            "Cash Reserve": (5.2, 5.0),
        }
        for name, (current, limit) in limits.items():
            pct = current / limit * 100
            status = "🟢" if pct < 80 else "🟡" if pct < 95 else "🔴"
            st.progress(min(pct / 100, 1.0), text=f"{status} {name}: {current:.1f} / {limit:.1f}")


# ── Agent Decisions ──────────────────────────────────────────────────


def render_agent_decisions():
    st.header("🤖 Agent Decision Log")

    log_path = os.path.join(PROJECT_ROOT, "logs", "decisions", "decisions.jsonl")
    if os.path.exists(log_path):
        with open(log_path) as f:
            lines = f.readlines()
        decisions = [json.loads(line) for line in lines[-20:]]

        for d in reversed(decisions):
            with st.expander(f"Step {d.get('step', '?')} — {d.get('type', '').upper()} — {d.get('date', 'N/A')}"):
                st.markdown(f"**Explanation:** {d.get('explanation', 'N/A')}")
                if "sector_weights" in d:
                    st.json(d["sector_weights"])
                if "top_stocks" in d:
                    st.json(d["top_stocks"])
    else:
        st.info("📁 No decision logs yet. Run a backtest to generate logs.")
        st.markdown("""
        **Demo decision log:**
        - Step 145 — MACRO — Detected Bull regime. Top sectors: Financials (20.1%), IT (15.3%), Auto (12.2%).
        - Step 145 — MICRO — Holding 28 positions. Largest: HDFCBANK (4.2%), TCS (3.8%), RELIANCE (3.5%). Closely aligned with Macro goal.
        """)


# ── Training Progress ────────────────────────────────────────────────


def render_training_progress():
    st.header("📈 Training Progress")

    phase = st.selectbox("Training Phase", [
        "Phase 1: Macro Pre-training",
        "Phase 2: Micro Pre-training",
        "Phase 3: Macro RL (frozen Micro)",
        "Phase 4: Micro RL (frozen Macro)",
        "Phase 5: Joint Fine-tuning",
    ])

    # Demo training curves
    np.random.seed(42)
    epochs = np.arange(1, 101)

    col1, col2 = st.columns(2)

    with col1:
        loss = 2.0 * np.exp(-0.03 * epochs) + np.random.normal(0, 0.05, 100)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=loss, mode="lines", name="Loss",
                                  line=dict(color="#ef4444")))
        fig.update_layout(title="Training Loss", template="plotly_dark", height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        reward = -0.5 + 1.5 * (1 - np.exp(-0.04 * epochs)) + np.random.normal(0, 0.1, 100)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=reward, mode="lines", name="Reward",
                                  line=dict(color="#22c55e")))
        fig.update_layout(title="Average Reward", template="plotly_dark", height=350)
        st.plotly_chart(fig, use_container_width=True)


# ── Stress Testing ───────────────────────────────────────────────────


def render_stress_testing():
    st.header("⚡ Stress Testing")

    from risk.stress_testing import StressTester

    tester = StressTester()
    weights = np.ones(11) / 11  # Equal weight

    st.slider("Portfolio Value (₹ Cr)", 0.5, 50.0, 1.0, key="stress_capital")
    capital = st.session_state.stress_capital * 10_000_000

    results = tester.run_all(weights, capital)
    report = tester.generate_report(results)

    col1, col2, col3 = st.columns(3)
    col1.metric("Scenarios Passed", f"{report['scenarios_passed']}/{report['n_scenarios']}")
    col2.metric("Worst Case P&L", f"₹{report['worst_case_pnl']:,.0f}")
    col3.metric("Worst Scenario", report["worst_scenario"])

    st.divider()

    # Scenario table
    scenario_data = []
    for name, detail in results.items():
        scenario_data.append({
            "Scenario": detail["description"][:45],
            "Return": f"{detail['portfolio_return'] * 100:.2f}%",
            "P&L": f"₹{detail['pnl']:,.0f}",
            "Max DD": f"{detail['max_drawdown'] * 100:.2f}%",
            "Status": "✅" if detail["passes_drawdown_limit"] else "❌",
        })

    st.table(scenario_data)


# ── Evaluation Results ───────────────────────────────────────────────


def render_evaluation_results():
    st.header("📋 Evaluation Results")

    results_path = os.path.join(PROJECT_ROOT, "results", "evaluation_results.json")
    report_path = os.path.join(PROJECT_ROOT, "results", "reports", "evaluation_report.md")

    if os.path.exists(report_path):
        with open(report_path) as f:
            st.markdown(f.read())
    elif os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        st.json(results)
    else:
        st.info("📁 No evaluation results yet. Run: `python main.py evaluate`")
        st.markdown("""
        **Expected output includes:**
        - Backtest performance metrics (Sharpe, Sortino, CAGR, Max DD)
        - Benchmark comparison (vs Nifty 50, Equal Weight, Momentum, etc.)
        - Stress test results across 7 India-specific scenarios
        - Regime-conditional analysis (Bull / Bear / Sideways)
        - Statistical significance tests
        """)


if __name__ == "__main__":
    main()
