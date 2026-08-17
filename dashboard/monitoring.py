"""Monitoring tab - prediction distribution, segment drift, LLM costs, latency.

Run standalone:  streamlit run dashboard/monitoring.py
Or import render_monitoring_tab() into the main dashboard.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from dashboard.theme import ACCENT_2, ACCENT_DEEP, MUTED, TEXT, apply_plotly_theme, inject_theme
except ImportError:
    def apply_plotly_theme(fig):
        return fig
    def inject_theme():
        pass
    ACCENT_2 = "#10b981"
    ACCENT_DEEP = "#6366f1"
    MUTED = "#64748b"
    TEXT = "#f1f5f9"

API_BASE = "http://localhost:8000"


def _fetch_metrics() -> dict | None:
    """Fetch from running API; return None if unavailable."""
    try:
        import requests
        r = requests.get(f"{API_BASE}/metrics", timeout=3)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def render_monitoring_tab():
    """Render the monitoring section (embeddable in the main dashboard)."""
    st.header("System Monitoring")

    data = _fetch_metrics()
    if data is None or data.get("total_predictions", 0) == 0:
        st.info("No prediction data available yet. Make some API calls to populate metrics.")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Predictions", data["total_predictions"])
    col2.metric("Avg Churn Prob", f"{data['avg_churn_probability']:.2%}" if data["avg_churn_probability"] else "N/A")
    col3.metric("Latency P50", f"{data['latency_p50_ms']}ms" if data["latency_p50_ms"] else "N/A")
    col4.metric("Latency P95", f"{data['latency_p95_ms']}ms" if data["latency_p95_ms"] else "N/A")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Segment Distribution")
        seg_dist = data.get("segment_distribution", {})
        if seg_dist:
            fig = px.pie(
                names=list(seg_dist.keys()),
                values=list(seg_dist.values()),
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color=TEXT,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No segment data yet.")

    with right:
        st.subheader("Risk Tier Distribution")
        risk_dist = data.get("risk_distribution", {})
        if risk_dist:
            fig = px.bar(
                x=list(risk_dist.keys()),
                y=list(risk_dist.values()),
                labels={"x": "Risk Tier", "y": "Count"},
                color=list(risk_dist.keys()),
                color_discrete_map={"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"},
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color=TEXT,
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No risk data yet.")

    st.divider()

    st.subheader("LLM Token Usage")
    tokens = data.get("llm_tokens_total", {})
    tcol1, tcol2, tcol3 = st.columns(3)
    input_t = tokens.get("input", 0)
    output_t = tokens.get("output", 0)
    cost_est = (input_t * 0.003 + output_t * 0.015) / 1000
    tcol1.metric("Input Tokens", f"{input_t:,}")
    tcol2.metric("Output Tokens", f"{output_t:,}")
    tcol3.metric("Est. Cost", f"${cost_est:.4f}")


if __name__ == "__main__":
    st.set_page_config(page_title="Monitoring - INSIGHTFORGE", layout="wide")
    inject_theme()
    render_monitoring_tab()
