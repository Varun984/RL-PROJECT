"""
Streamlit monitoring dashboard for HRL-SARP.
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
from datetime import datetime
from html import escape
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

try:
    import streamlit as st
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import yaml

    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False


SECTOR_NAMES = [
    "IT",
    "Financials",
    "Pharma",
    "FMCG",
    "Auto",
    "Energy",
    "Metals",
    "Realty",
    "Telecom",
    "Media",
    "Infra",
]

EPISODE_PATTERN = re.compile(
    r"Episode\s+(\d+)\s+\|\s+return=([-+0-9.eE]+)\s+\|\s+sharpe=([-+0-9.eE]+)\s+\|\s+steps=(\d+)\s+\|\s+stage=([A-Za-z0-9_-]+)"
)
PPO_PATTERN = re.compile(
    r"PPO Update\s+(\d+)\s+\|\s+policy_loss=([-+0-9.eE]+)\s+\|\s+value_loss=([-+0-9.eE]+)\s+\|\s+entropy=([-+0-9.eE]+)"
)
TOTAL_STEPS_PATTERN = re.compile(r"total_steps=(\d+)")
RUN_TIMESTAMP_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+\|")


def main() -> None:
    if not HAS_STREAMLIT:
        print("Streamlit not installed. Run: pip install streamlit plotly")
        return

    st.set_page_config(
        page_title="HRL-SARP Dashboard",
        page_icon="HS",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _inject_styles()
    page, show_paths = _render_sidebar()
    bundle = _load_data_bundle()
    _render_header(bundle)

    if page == "Overview":
        render_overview(bundle, show_paths)
    elif page == "Training":
        render_training_progress(bundle, show_paths)
    elif page == "Sector Allocation":
        render_sector_allocation(bundle)
    elif page == "Risk Monitor":
        render_risk_monitor(bundle)
    elif page == "Agent Decisions":
        render_agent_decisions(bundle)
    elif page == "Stress Testing":
        render_stress_testing(bundle)
    elif page == "Evaluation":
        render_evaluation_results(bundle)


def _render_sidebar() -> Tuple[str, bool]:
    with st.sidebar:
        st.markdown("### Control Panel")
        if st.button("Refresh data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        page = st.radio(
            "View",
            [
                "Overview",
                "Training",
                "Sector Allocation",
                "Risk Monitor",
                "Agent Decisions",
                "Stress Testing",
                "Evaluation",
            ],
            index=1,
        )
        show_paths = st.checkbox("Show file sources", value=False)
        st.caption("Tip: use Refresh data while Phase 3 is running.")
    return page, show_paths


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
        :root {
            --bg-soft: #f6fbff;
            --bg-accent: #d7ecff;
            --card: #ffffff;
            --text: #12243a;
            --muted: #5d7089;
            --border: #d8e3f0;
            --ok: #0f9d76;
            --warn: #c77d00;
            --danger: #d64545;
            --brand: #1f77ff;
        }
        .stApp {
            background:
                radial-gradient(60rem 25rem at 105% -10%, var(--bg-accent) 0%, transparent 65%),
                radial-gradient(40rem 20rem at -5% 0%, #e6f5ff 0%, transparent 65%),
                var(--bg-soft);
            color: var(--text);
            font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
        }
        h1, h2, h3 {
            font-family: "Space Grotesk", "Segoe UI", sans-serif;
            letter-spacing: 0.01em;
        }
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 0.75rem 0.8rem;
            box-shadow: 0 8px 24px rgba(16, 42, 67, 0.05);
        }
        .hero {
            background: linear-gradient(145deg, #ffffff 0%, #f3faff 100%);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            margin: 0 0 1rem 0;
            box-shadow: 0 12px 36px rgba(16, 42, 67, 0.08);
        }
        .hero-title {
            font-size: 1.55rem;
            font-weight: 700;
            color: var(--text);
            margin: 0;
        }
        .hero-sub {
            color: var(--muted);
            margin-top: 0.2rem;
            font-size: 0.95rem;
        }
        .kpi-card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 0.85rem 0.9rem;
            min-height: 104px;
            box-shadow: 0 10px 30px rgba(16, 42, 67, 0.05);
        }
        .kpi-label {
            color: var(--muted);
            font-size: 0.82rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .kpi-value {
            margin-top: 0.2rem;
            color: var(--text);
            font-size: 1.35rem;
            font-weight: 700;
        }
        .kpi-sub {
            margin-top: 0.2rem;
            color: var(--muted);
            font-size: 0.82rem;
        }
        .kpi-ok .kpi-value { color: var(--ok); }
        .kpi-warn .kpi-value { color: var(--warn); }
        .kpi-danger .kpi-value { color: var(--danger); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_kpi_card(label: str, value: str, subtitle: str = "", tone: str = "ok") -> None:
    safe_label = escape(label)
    safe_value = escape(value)
    safe_subtitle = escape(subtitle)
    st.markdown(
        f"""
        <div class="kpi-card kpi-{tone}">
            <div class="kpi-label">{safe_label}</div>
            <div class="kpi-value">{safe_value}</div>
            <div class="kpi-sub">{safe_subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_header(bundle: Dict[str, Any]) -> None:
    training = bundle["training_log"]
    status = training.get("status", "Idle")
    status_updated = training.get("last_timestamp")
    updated_text = (
        status_updated.strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(status_updated, datetime)
        else "N/A"
    )
    st.markdown(
        f"""
        <div class="hero">
            <p class="hero-title">HRL-SARP Command Dashboard</p>
            <p class="hero-sub">
                Live monitoring for training, allocations, risk posture, and evaluation outputs.
                Current Phase 3 status: <strong>{escape(status)}</strong> | Last update: {escape(updated_text)}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=15, show_spinner=False)
def _load_data_bundle() -> Dict[str, Any]:
    eval_result, eval_path = _load_evaluation_results()
    phase3_metrics, phase3_metrics_path = _load_phase3_metrics()
    decisions, decisions_path = _load_decisions()
    risk_cfg, risk_cfg_path = _load_risk_config()
    train_log_path = _latest_train_log_path()
    training_log = _parse_training_log(train_log_path) if train_log_path else {}

    return {
        "evaluation": eval_result,
        "evaluation_path": eval_path,
        "phase3_metrics": phase3_metrics,
        "phase3_metrics_path": phase3_metrics_path,
        "decisions": decisions,
        "decisions_path": decisions_path,
        "risk_config": risk_cfg,
        "risk_config_path": risk_cfg_path,
        "training_log": training_log,
        "train_log_path": train_log_path,
    }


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _load_evaluation_results() -> Tuple[Dict[str, Any], Optional[str]]:
    candidates = [
        os.path.join(PROJECT_ROOT, "results", "evaluation_results.json"),
        os.path.join(PROJECT_ROOT, "results", "results", "evaluation_results.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            data = _read_json(path)
            if data:
                return data, path
    return {}, None


def _load_phase3_metrics() -> Tuple[Dict[str, Any], Optional[str]]:
    candidates = [os.path.join(PROJECT_ROOT, "logs", "phase3_macro", "phase3_macro_metrics.json")]
    archive_paths = glob.glob(
        os.path.join(PROJECT_ROOT, "logs", "archive", "phase3_*", "phase3_macro_metrics.json")
    )
    archive_paths.sort(key=os.path.getmtime, reverse=True)
    candidates.extend(archive_paths)

    for path in candidates:
        if os.path.exists(path):
            data = _read_json(path)
            if data and isinstance(data.get("episode/return"), list):
                return data, path
    return {}, None


def _load_decisions() -> Tuple[List[Dict[str, Any]], Optional[str]]:
    path = os.path.join(PROJECT_ROOT, "logs", "decisions", "decisions.jsonl")
    if not os.path.exists(path):
        return [], None

    decisions: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    decisions.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return [], path
    return decisions, path


def _load_risk_config() -> Tuple[Dict[str, Any], Optional[str]]:
    path = os.path.join(PROJECT_ROOT, "config", "risk_config.yaml")
    if not os.path.exists(path):
        return {}, None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}, path
    except OSError:
        return {}, path


def _latest_train_log_path() -> Optional[str]:
    paths = glob.glob(os.path.join(PROJECT_ROOT, "logs", "train_*.log"))
    if not paths:
        return None
    return max(paths, key=os.path.getmtime)


def _parse_training_log(path: str) -> Dict[str, Any]:
    episodes: List[Dict[str, Any]] = []
    ppo_updates: List[Dict[str, Any]] = []
    total_steps = 0
    completed = False
    start_timestamp: Optional[datetime] = None
    last_timestamp: Optional[datetime] = None

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except OSError:
        return {}

    for line in lines:
        ts_match = RUN_TIMESTAMP_PATTERN.search(line)
        if ts_match:
            try:
                ts = datetime.strptime(ts_match.group(1), "%Y-%m-%d %H:%M:%S")
                if start_timestamp is None:
                    start_timestamp = ts
                last_timestamp = ts
            except ValueError:
                pass

        if "Phase 3: Macro RL training" in line:
            step_match = TOTAL_STEPS_PATTERN.search(line)
            if step_match:
                total_steps = int(step_match.group(1))

        if "Phase 3 complete" in line:
            completed = True

        episode_match = EPISODE_PATTERN.search(line)
        if episode_match:
            episodes.append(
                {
                    "episode": int(episode_match.group(1)),
                    "return": float(episode_match.group(2)),
                    "sharpe": float(episode_match.group(3)),
                    "steps": int(episode_match.group(4)),
                    "stage": episode_match.group(5),
                }
            )

        ppo_match = PPO_PATTERN.search(line)
        if ppo_match:
            ppo_updates.append(
                {
                    "update": int(ppo_match.group(1)),
                    "policy_loss": float(ppo_match.group(2)),
                    "value_loss": float(ppo_match.group(3)),
                    "entropy": float(ppo_match.group(4)),
                }
            )

    tail_lines = [ln.rstrip("\n") for ln in lines[-60:]]

    latest_step = episodes[-1]["steps"] if episodes else 0
    latest_mtime = datetime.fromtimestamp(os.path.getmtime(path))
    active = (not completed) and ((datetime.now() - latest_mtime).total_seconds() < 180)
    status = "Running" if active else ("Completed" if completed else "Idle")

    return {
        "path": path,
        "episodes": episodes,
        "ppo_updates": ppo_updates,
        "total_steps": total_steps,
        "latest_step": latest_step,
        "start_timestamp": start_timestamp,
        "last_timestamp": last_timestamp,
        "status": status,
        "tail_lines": tail_lines,
    }


def _extract_macro_weights(decisions: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    macro_decisions = [d for d in decisions if d.get("type") == "macro" and "sector_weights" in d]
    if not macro_decisions:
        return None
    raw = macro_decisions[-1].get("sector_weights", {})
    return {sector: float(raw.get(sector, 0.0)) for sector in SECTOR_NAMES}


def _get_portfolio_values(evaluation: Dict[str, Any]) -> np.ndarray:
    values = evaluation.get("portfolio_values", [])
    if not values:
        return np.array([], dtype=float)
    arr = np.array(values, dtype=float)
    return arr[arr > 0]


def _series_to_drawdown(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.array([], dtype=float)
    peak = np.maximum.accumulate(values)
    return (values / peak) - 1.0


def _format_inr(value: float) -> str:
    return f"INR {value:,.0f}"


def _format_pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def _source_frame(bundle: Dict[str, Any]) -> pd.DataFrame:
    rows = [
        ("Evaluation results", bundle.get("evaluation_path")),
        ("Phase 3 metrics", bundle.get("phase3_metrics_path")),
        ("Training log", bundle.get("train_log_path")),
        ("Decision log", bundle.get("decisions_path")),
        ("Risk config", bundle.get("risk_config_path")),
    ]
    out = []
    for name, path in rows:
        if path and os.path.exists(path):
            out.append(
                {
                    "Source": name,
                    "Path": path,
                    "Last updated": datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
        else:
            out.append({"Source": name, "Path": "Not found", "Last updated": "N/A"})
    return pd.DataFrame(out)


def render_overview(bundle: Dict[str, Any], show_paths: bool) -> None:
    st.subheader("Portfolio Overview")

    evaluation = bundle["evaluation"]
    training = bundle["training_log"]
    values = _get_portfolio_values(evaluation)
    perf = evaluation.get("performance_metrics", {})

    if values.size > 1:
        total_return = (values[-1] / values[0]) - 1.0
        drawdown = _series_to_drawdown(values)
        current_dd = drawdown[-1]
    else:
        total_return = 0.0
        current_dd = 0.0

    sharpe = float(perf.get("sharpe_ratio", np.nan))
    cagr = float(perf.get("cagr", np.nan))
    status = training.get("status", "Idle")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _render_kpi_card(
            "Portfolio Value",
            _format_inr(values[-1]) if values.size else "N/A",
            "Latest mark-to-market value",
            tone="ok",
        )
    with c2:
        _render_kpi_card("Total Return", _format_pct(total_return), "From evaluation period", tone="ok")
    with c3:
        sharpe_text = f"{sharpe:.2f}" if np.isfinite(sharpe) else "N/A"
        cagr_text = _format_pct(cagr) if np.isfinite(cagr) else "CAGR unavailable"
        _render_kpi_card("Sharpe Ratio", sharpe_text, cagr_text, tone="ok")
    with c4:
        dd_text = _format_pct(current_dd)
        _render_kpi_card("Run Status", status, f"Current drawdown: {dd_text}", tone="warn" if status == "Idle" else "ok")

    st.markdown("")
    left, right = st.columns([2, 1])
    with left:
        if values.size > 1:
            dd = _series_to_drawdown(values) * 100.0
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.1)
            fig.add_trace(
                go.Scatter(
                    y=values,
                    mode="lines",
                    line=dict(width=2.5, color="#1f77ff"),
                    fill="tozeroy",
                    fillcolor="rgba(31,119,255,0.12)",
                    name="Portfolio value",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    y=dd,
                    mode="lines",
                    line=dict(width=2, color="#d64545"),
                    fill="tozeroy",
                    fillcolor="rgba(214,69,69,0.16)",
                    name="Drawdown %",
                ),
                row=2,
                col=1,
            )
            fig.update_layout(
                template="plotly_white",
                height=520,
                margin=dict(l=10, r=10, t=25, b=10),
                title="Equity Curve and Drawdown",
                legend=dict(orientation="h", y=1.05, x=0.0),
            )
            fig.update_yaxes(title_text="Portfolio Value", row=1, col=1)
            fig.update_yaxes(title_text="Drawdown %", row=2, col=1)
            fig.update_xaxes(title_text="Observation Index", row=2, col=1)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No evaluation portfolio series found yet.")

    with right:
        phase3_metrics = bundle["phase3_metrics"]
        episode_returns = phase3_metrics.get("episode/return", [])
        eval_sharpe = phase3_metrics.get("eval_sharpe_mean", [])
        summary_rows = [
            ("Phase 3 episodes recorded", len(episode_returns)),
            ("Eval checkpoints", len(eval_sharpe)),
            ("Decision log entries", len(bundle["decisions"])),
            ("Latest log file", os.path.basename(bundle["train_log_path"] or "N/A")),
        ]
        summary_df = pd.DataFrame(summary_rows, columns=["Metric", "Value"])
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    if show_paths:
        st.caption("Data Source Files")
        st.dataframe(_source_frame(bundle), use_container_width=True, hide_index=True)


def render_training_progress(bundle: Dict[str, Any], show_paths: bool) -> None:
    st.subheader("Phase 3 Training Monitor")

    training = bundle["training_log"]
    phase3_metrics = bundle["phase3_metrics"]
    episodes = training.get("episodes", [])
    updates = training.get("ppo_updates", [])
    total_steps = int(training.get("total_steps", 0))
    latest_step = int(training.get("latest_step", 0))
    status = training.get("status", "Idle")

    latest_ep = episodes[-1]["episode"] if episodes else 0
    latest_ret = episodes[-1]["return"] if episodes else 0.0
    latest_sharpe = episodes[-1]["sharpe"] if episodes else 0.0
    pct_done = (latest_step / total_steps) if total_steps > 0 else 0.0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _render_kpi_card("Run Status", status, "Derived from recent train log activity", tone="ok" if status == "Running" else "warn")
    with c2:
        _render_kpi_card("Latest Episode", f"{latest_ep}", "From current train log", tone="ok")
    with c3:
        _render_kpi_card("Latest Sharpe", f"{latest_sharpe:.3f}", f"Episode return: {latest_ret:.3f}", tone="ok")
    with c4:
        total_text = f"{latest_step:,}/{total_steps:,}" if total_steps > 0 else f"{latest_step:,}"
        _render_kpi_card("Step Progress", total_text, f"{pct_done * 100:.1f}% complete" if total_steps else "Target not detected", tone="ok")

    if total_steps > 0:
        st.progress(min(max(pct_done, 0.0), 1.0), text=f"Phase 3 step progress: {pct_done * 100:.1f}%")

    left, right = st.columns(2)
    with left:
        if episodes:
            ep_df = pd.DataFrame(episodes)
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Scatter(
                    x=ep_df["episode"],
                    y=ep_df["return"],
                    mode="lines+markers",
                    line=dict(color="#1f77ff", width=2),
                    marker=dict(size=5),
                    name="Episode return",
                ),
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(
                    x=ep_df["episode"],
                    y=ep_df["sharpe"],
                    mode="lines",
                    line=dict(color="#0f9d76", width=2),
                    name="Episode sharpe",
                ),
                secondary_y=True,
            )
            fig.update_layout(
                template="plotly_white",
                height=380,
                margin=dict(l=10, r=10, t=25, b=10),
                title="Live Episode Metrics",
            )
            fig.update_xaxes(title_text="Episode")
            fig.update_yaxes(title_text="Return", secondary_y=False)
            fig.update_yaxes(title_text="Sharpe", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Episode lines not found yet in the latest train log.")

    with right:
        if updates:
            upd_df = pd.DataFrame(updates)
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=upd_df["update"],
                    y=upd_df["policy_loss"],
                    mode="lines",
                    name="Policy loss",
                    line=dict(color="#ff7f0e", width=2),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=upd_df["update"],
                    y=upd_df["value_loss"],
                    mode="lines",
                    name="Value loss",
                    line=dict(color="#d64545", width=2),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=upd_df["update"],
                    y=upd_df["entropy"],
                    mode="lines",
                    name="Entropy",
                    line=dict(color="#9b51e0", width=2),
                )
            )
            fig.update_layout(
                template="plotly_white",
                height=380,
                margin=dict(l=10, r=10, t=25, b=10),
                title="PPO Update Signals",
                xaxis_title="Update #",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("PPO update lines not found yet in the latest train log.")

    eval_returns = phase3_metrics.get("eval_return_mean", [])
    eval_sharpes = phase3_metrics.get("eval_sharpe_mean", [])
    if eval_returns and eval_sharpes:
        eval_idx = np.arange(1, len(eval_returns) + 1)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(
                x=eval_idx,
                y=eval_returns,
                mode="lines+markers",
                line=dict(color="#1f77ff", width=2),
                name="Eval return mean",
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=eval_idx,
                y=eval_sharpes,
                mode="lines+markers",
                line=dict(color="#0f9d76", width=2),
                name="Eval sharpe mean",
            ),
            secondary_y=True,
        )
        fig.update_layout(
            template="plotly_white",
            height=320,
            margin=dict(l=10, r=10, t=25, b=10),
            title="Validation Trend (saved Phase 3 metrics)",
            xaxis_title="Evaluation checkpoint",
        )
        fig.update_yaxes(title_text="Eval return mean", secondary_y=False)
        fig.update_yaxes(title_text="Eval sharpe mean", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("`phase3_macro_metrics.json` not available yet, this is expected before run completion.")

    with st.expander("Latest train log tail", expanded=False):
        tail = "\n".join(training.get("tail_lines", []))
        st.code(tail or "No log lines available.", language="text")

    if show_paths:
        st.caption("Data Source Files")
        st.dataframe(_source_frame(bundle), use_container_width=True, hide_index=True)


def render_sector_allocation(bundle: Dict[str, Any]) -> None:
    st.subheader("Sector Allocation and Rotation")
    decisions = bundle["decisions"]
    evaluation = bundle["evaluation"]

    latest_weights = _extract_macro_weights(decisions)
    if latest_weights is None:
        latest_weights = {sector: 1.0 / len(SECTOR_NAMES) for sector in SECTOR_NAMES}
        st.info("No macro decision weights found, showing equal-weight fallback.")

    weights = np.array([latest_weights[s] for s in SECTOR_NAMES], dtype=float)

    left, right = st.columns(2)
    with left:
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=SECTOR_NAMES,
                    values=weights,
                    hole=0.52,
                    sort=False,
                    marker=dict(colors=px.colors.qualitative.Safe),
                )
            ]
        )
        fig.update_layout(
            template="plotly_white",
            title="Current Sector Mix",
            height=430,
            margin=dict(l=10, r=10, t=45, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        sector_perf = evaluation.get("sector_performance", {})
        perf = np.array([float(sector_perf.get(s, 0.0)) for s in SECTOR_NAMES], dtype=float)
        colors = ["#0f9d76" if x >= 0 else "#d64545" for x in perf]
        fig = go.Figure(
            data=[
                go.Bar(
                    x=SECTOR_NAMES,
                    y=perf * 100,
                    marker_color=colors,
                    text=[f"{v * 100:.2f}%" for v in perf],
                    textposition="outside",
                )
            ]
        )
        fig.update_layout(
            template="plotly_white",
            title="Sector Performance (Evaluation File)",
            height=430,
            margin=dict(l=10, r=10, t=45, b=10),
            yaxis_title="Return %",
        )
        st.plotly_chart(fig, use_container_width=True)

    macro_hist = [d for d in decisions if d.get("type") == "macro" and "sector_weights" in d]
    if len(macro_hist) >= 2:
        hist_rows = []
        for item in macro_hist[-50:]:
            row = {"step": int(item.get("step", 0))}
            sector_weights = item.get("sector_weights", {})
            for sector in SECTOR_NAMES:
                row[sector] = float(sector_weights.get(sector, 0.0))
            hist_rows.append(row)
        hist_df = pd.DataFrame(hist_rows).sort_values("step")

        fig = go.Figure()
        for sector in SECTOR_NAMES:
            fig.add_trace(
                go.Scatter(
                    x=hist_df["step"],
                    y=hist_df[sector] * 100.0,
                    stackgroup="alloc",
                    mode="lines",
                    line=dict(width=0.7),
                    name=sector,
                )
            )
        fig.update_layout(
            template="plotly_white",
            height=420,
            margin=dict(l=10, r=10, t=40, b=10),
            title="Allocation Drift Across Macro Decisions",
            yaxis_title="Weight %",
            xaxis_title="Decision step",
        )
        st.plotly_chart(fig, use_container_width=True)

        corr = hist_df[SECTOR_NAMES].corr()
        heatmap = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.index,
                colorscale="RdBu",
                zmid=0,
            )
        )
        heatmap.update_layout(
            template="plotly_white",
            height=470,
            margin=dict(l=10, r=10, t=45, b=10),
            title="Correlation of Allocation Changes",
        )
        st.plotly_chart(heatmap, use_container_width=True)
    else:
        st.info("Need at least two macro decisions to show allocation history and correlation.")


def render_risk_monitor(bundle: Dict[str, Any]) -> None:
    st.subheader("Risk Monitor")
    evaluation = bundle["evaluation"]
    risk_cfg = bundle["risk_config"]
    decisions = bundle["decisions"]

    values = _get_portfolio_values(evaluation)
    returns = np.diff(values) / values[:-1] if values.size > 1 else np.array([], dtype=float)
    drawdown = _series_to_drawdown(values)

    current_dd = float(drawdown[-1]) if drawdown.size else 0.0
    max_dd = float(np.min(drawdown)) if drawdown.size else 0.0
    if returns.size:
        var95 = float(np.percentile(returns, 5))
        tail = returns[returns <= var95]
        cvar95 = float(tail.mean()) if tail.size else var95
    else:
        var95 = 0.0
        cvar95 = 0.0

    latest_weights = _extract_macro_weights(decisions) or {s: 1.0 / len(SECTOR_NAMES) for s in SECTOR_NAMES}
    max_sector_weight = max(latest_weights.values())

    micro_decisions = [d for d in decisions if d.get("type") == "micro" and "top_stocks" in d]
    top_stock_weight = 0.0
    if micro_decisions:
        top_stock_weight = max(float(v) for v in micro_decisions[-1]["top_stocks"].values())

    portfolio_cfg = risk_cfg.get("portfolio", {})
    sector_cfg = risk_cfg.get("sector", {})
    stock_cfg = risk_cfg.get("stock", {})

    max_dd_limit = float(portfolio_cfg.get("max_drawdown_pct", 0.07))
    max_sector_limit = float(sector_cfg.get("max_single_sector_pct", 0.35))
    max_stock_limit = float(stock_cfg.get("max_single_stock_pct", 0.10))
    min_cash_limit = float(portfolio_cfg.get("min_cash_pct", 0.02))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current drawdown", _format_pct(current_dd))
    c2.metric("Max drawdown (run)", _format_pct(max_dd))
    c3.metric("VaR 95%", _format_pct(var95))
    c4.metric("CVaR 95%", _format_pct(cvar95))

    st.markdown("")
    limit_rows = [
        ("Drawdown limit", abs(current_dd), max_dd_limit, "lower_better"),
        ("Max sector concentration", max_sector_weight, max_sector_limit, "lower_better"),
        ("Max stock concentration", top_stock_weight, max_stock_limit, "lower_better"),
        ("Min cash reserve", min_cash_limit, min_cash_limit, "meets_target"),
    ]
    for label, current, limit, mode in limit_rows:
        if mode == "meets_target":
            ratio = 1.0
            color = "ok"
            text = f"{label}: target {limit * 100:.2f}% configured"
        else:
            ratio = current / limit if limit > 0 else 0.0
            color = "ok" if ratio <= 0.8 else ("warn" if ratio <= 1.0 else "danger")
            text = f"{label}: {current * 100:.2f}% / limit {limit * 100:.2f}%"
        st.progress(min(ratio, 1.0), text=text)
        if color == "danger":
            st.error(f"Breach: {label}")

    left, right = st.columns(2)
    with left:
        if drawdown.size:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    y=drawdown * 100.0,
                    mode="lines",
                    line=dict(color="#d64545", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(214,69,69,0.15)",
                    name="Drawdown %",
                )
            )
            fig.add_hline(
                y=-max_dd_limit * 100.0,
                line_color="#c77d00",
                line_dash="dash",
                annotation_text=f"Configured limit ({max_dd_limit * 100:.1f}%)",
            )
            fig.update_layout(
                template="plotly_white",
                height=360,
                margin=dict(l=10, r=10, t=35, b=10),
                title="Drawdown Profile",
                yaxis_title="Drawdown %",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Drawdown chart needs evaluation portfolio values.")

    with right:
        if returns.size:
            fig = go.Figure()
            fig.add_trace(
                go.Histogram(
                    x=returns * 100.0,
                    nbinsx=40,
                    marker_color="#1f77ff",
                    opacity=0.85,
                )
            )
            fig.update_layout(
                template="plotly_white",
                height=360,
                margin=dict(l=10, r=10, t=35, b=10),
                title="Return Distribution",
                xaxis_title="Return %",
                yaxis_title="Count",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Return distribution needs at least two portfolio value points.")


def render_agent_decisions(bundle: Dict[str, Any]) -> None:
    st.subheader("Agent Decision Logs")
    decisions = bundle["decisions"]
    if not decisions:
        st.info("No decision log found yet. Expected file: logs/decisions/decisions.jsonl")
        return

    decision_type = st.selectbox("Filter", ["all", "macro", "micro"], index=0)
    filtered = [d for d in decisions if decision_type == "all" or d.get("type") == decision_type]

    if not filtered:
        st.warning("No records for the selected filter.")
        return

    rows = []
    for d in filtered[-120:]:
        rows.append(
            {
                "step": d.get("step", ""),
                "date": d.get("date", ""),
                "type": d.get("type", ""),
                "regime": d.get("regime", ""),
                "explanation": str(d.get("explanation", ""))[:120],
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    latest_micro = next((d for d in reversed(filtered) if d.get("type") == "micro" and "top_stocks" in d), None)
    if latest_micro:
        stocks = latest_micro["top_stocks"]
        stock_df = pd.DataFrame(
            [{"symbol": k, "weight": float(v)} for k, v in stocks.items()]
        ).sort_values("weight", ascending=False)
        fig = go.Figure(
            data=[
                go.Bar(
                    x=stock_df["symbol"],
                    y=stock_df["weight"] * 100.0,
                    marker_color="#1f77ff",
                    text=[f"{v * 100:.2f}%" for v in stock_df["weight"]],
                    textposition="outside",
                )
            ]
        )
        fig.update_layout(
            template="plotly_white",
            height=330,
            margin=dict(l=10, r=10, t=35, b=10),
            title="Latest Micro Agent Top Holdings",
            yaxis_title="Weight %",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Recent Decisions")
    for d in reversed(filtered[-20:]):
        step = d.get("step", "?")
        d_type = str(d.get("type", "")).upper()
        date = d.get("date", "N/A")
        with st.expander(f"Step {step} | {d_type} | {date}", expanded=False):
            st.write(d.get("explanation", "No explanation provided."))
            if "sector_weights" in d:
                st.json(d["sector_weights"])
            if "top_stocks" in d:
                st.json(d["top_stocks"])


def render_stress_testing(bundle: Dict[str, Any]) -> None:
    st.subheader("Stress Testing")
    try:
        from risk.stress_testing import StressTester
    except ImportError as exc:
        st.error(f"Stress testing module import failed: {exc}")
        return

    decisions = bundle["decisions"]
    latest_weights = _extract_macro_weights(decisions)
    if latest_weights is None:
        weights = np.ones(len(SECTOR_NAMES), dtype=float) / len(SECTOR_NAMES)
    else:
        weights = np.array([latest_weights[s] for s in SECTOR_NAMES], dtype=float)

    capital_cr = st.slider("Portfolio value (INR crore)", 0.5, 100.0, 1.0, step=0.5)
    capital = capital_cr * 10_000_000.0

    tester = StressTester(config_path=os.path.join(PROJECT_ROOT, "config", "risk_config.yaml"))
    results = tester.run_all(weights, capital)
    report = tester.generate_report(results)

    c1, c2, c3 = st.columns(3)
    c1.metric("Scenarios passed", f"{report['scenarios_passed']}/{report['n_scenarios']}")
    c2.metric("Worst case P&L", _format_inr(report["worst_case_pnl"]))
    c3.metric("Worst scenario", report["worst_scenario"])

    rows = []
    for scenario, detail in results.items():
        rows.append(
            {
                "Scenario": scenario,
                "Description": detail["description"],
                "Return %": detail["portfolio_return"] * 100.0,
                "P&L": detail["pnl"],
                "Max DD %": detail["max_drawdown"] * 100.0,
                "Pass": "Yes" if detail["passes_drawdown_limit"] else "No",
            }
        )
    df = pd.DataFrame(rows).sort_values("P&L", ascending=True)

    fig = go.Figure(
        data=[
            go.Bar(
                x=df["Scenario"],
                y=df["P&L"],
                marker_color=["#d64545" if x < 0 else "#0f9d76" for x in df["P&L"]],
                text=[_format_inr(v) for v in df["P&L"]],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        template="plotly_white",
        height=360,
        margin=dict(l=10, r=10, t=35, b=10),
        title="Scenario-wise P&L",
        yaxis_title="P&L (INR)",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_evaluation_results(bundle: Dict[str, Any]) -> None:
    st.subheader("Evaluation Results")
    evaluation = bundle["evaluation"]
    if not evaluation:
        st.info("No evaluation results found. Run: python main.py evaluate")
        return

    perf = evaluation.get("performance_metrics", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total return", _format_pct(float(perf.get("total_return", 0.0))))
    c2.metric("CAGR", _format_pct(float(perf.get("cagr", 0.0))))
    c3.metric("Sharpe", f"{float(perf.get('sharpe_ratio', 0.0)):.2f}")
    c4.metric("Max drawdown", _format_pct(float(perf.get("max_drawdown", 0.0))))

    benchmark = evaluation.get("benchmark_comparison", {})
    if benchmark:
        rows = []
        for name, stats in benchmark.items():
            rows.append(
                {
                    "benchmark": name,
                    "total_return": float(stats.get("total_return", 0.0)),
                    "sharpe_ratio": float(stats.get("sharpe_ratio", 0.0)),
                    "max_drawdown": float(stats.get("max_drawdown", 0.0)),
                }
            )
        bench_df = pd.DataFrame(rows)
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=bench_df["benchmark"],
                y=bench_df["total_return"] * 100.0,
                name="Total return %",
                marker_color="#1f77ff",
            )
        )
        fig.add_trace(
            go.Bar(
                x=bench_df["benchmark"],
                y=bench_df["sharpe_ratio"],
                name="Sharpe ratio",
                marker_color="#0f9d76",
            )
        )
        fig.update_layout(
            template="plotly_white",
            barmode="group",
            height=360,
            margin=dict(l=10, r=10, t=35, b=10),
            title="Benchmark Comparison",
        )
        st.plotly_chart(fig, use_container_width=True)

    regimes = evaluation.get("regime_performance", {})
    if regimes:
        regime_rows = []
        for regime, stats in regimes.items():
            regime_rows.append(
                {
                    "regime": regime,
                    "episodes": int(stats.get("episodes", 0)),
                    "avg_return": float(stats.get("avg_return", 0.0)),
                    "sharpe_ratio": float(stats.get("sharpe_ratio", 0.0)),
                    "win_rate": float(stats.get("win_rate", 0.0)),
                }
            )
        regime_df = pd.DataFrame(regime_rows)
        st.dataframe(regime_df, use_container_width=True, hide_index=True)

    with st.expander("Raw evaluation JSON", expanded=False):
        st.json(evaluation)


if __name__ == "__main__":
    main()
