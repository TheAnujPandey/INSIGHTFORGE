"""In-memory prediction metrics collector for monitoring.

Stores the last N predictions in a deque and exposes aggregation helpers
for the /metrics endpoint and the monitoring dashboard.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import List

import numpy as np


@dataclass
class PredictionLog:
    timestamp: str
    customer_id: str
    churn_probability: float
    segment: str
    risk_tier: str
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    latency_ms: int = 0
    endpoint: str = ""


_MAX_HISTORY = 1000
_history: deque[PredictionLog] = deque(maxlen=_MAX_HISTORY)


def log_prediction(entry: PredictionLog) -> None:
    _history.append(entry)


def get_history() -> List[dict]:
    return [asdict(e) for e in _history]


def get_metrics_summary() -> dict:
    """Compute aggregate metrics from the prediction history."""
    if not _history:
        return {
            "total_predictions": 0,
            "avg_churn_probability": None,
            "segment_distribution": {},
            "risk_distribution": {},
            "llm_tokens_total": {"input": 0, "output": 0},
            "latency_p50_ms": None,
            "latency_p95_ms": None,
        }

    entries = list(_history)
    probas = [e.churn_probability for e in entries]
    latencies = [e.latency_ms for e in entries if e.latency_ms > 0]

    seg_counts: dict[str, int] = {}
    risk_counts: dict[str, int] = {}
    total_input_tokens = 0
    total_output_tokens = 0

    for e in entries:
        seg_counts[e.segment] = seg_counts.get(e.segment, 0) + 1
        risk_counts[e.risk_tier] = risk_counts.get(e.risk_tier, 0) + 1
        total_input_tokens += e.llm_input_tokens
        total_output_tokens += e.llm_output_tokens

    return {
        "total_predictions": len(entries),
        "avg_churn_probability": round(float(np.mean(probas)), 4),
        "churn_prob_std": round(float(np.std(probas)), 4),
        "segment_distribution": seg_counts,
        "risk_distribution": risk_counts,
        "llm_tokens_total": {"input": total_input_tokens, "output": total_output_tokens},
        "latency_p50_ms": int(np.percentile(latencies, 50)) if latencies else None,
        "latency_p95_ms": int(np.percentile(latencies, 95)) if latencies else None,
        "latest_prediction_at": entries[-1].timestamp if entries else None,
    }
