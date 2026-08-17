"""Retrieval quality evaluation harness.

Measures precision@k, recall@k, and MRR against a hand-curated ground-truth
set of (query, expected_source_docs) pairs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from src.config import settings
from src.rag.retriever import Retriever, get_retriever
from src.utils.logger import get_logger

log = get_logger(__name__)

_DEFAULT_EVAL_PATH = Path(__file__).resolve().parents[2] / "tests" / "eval_retrieval.json"


def load_eval_set(path: Path | None = None) -> List[dict]:
    path = path or _DEFAULT_EVAL_PATH
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_retrieval(
    retriever: Retriever | None = None,
    eval_path: Path | None = None,
    k: int = 4,
    min_score: float = 0.0,
) -> dict:
    """Run the evaluation and return metrics.

    Returns:
        dict with per-query results, aggregate precision@k, recall@k, and MRR.
    """
    retriever = retriever or get_retriever()
    eval_set = load_eval_set(eval_path)

    per_query: List[dict] = []
    total_precision = 0.0
    total_recall = 0.0
    total_rr = 0.0

    for item in eval_set:
        query = item["query"]
        expected = set(item["expected_sources"])
        results = retriever.search(query, k=k, min_score=min_score)
        retrieved_sources = [r.source for r in results]
        retrieved_set = set(retrieved_sources)

        hits = retrieved_set & expected
        precision = len(hits) / len(retrieved_sources) if retrieved_sources else 0.0
        recall = len(hits) / len(expected) if expected else 0.0

        # MRR: reciprocal rank of first relevant result
        rr = 0.0
        for rank, src in enumerate(retrieved_sources, 1):
            if src in expected:
                rr = 1.0 / rank
                break

        per_query.append({
            "query": query[:80],
            "expected": sorted(expected),
            "retrieved": retrieved_sources,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "reciprocal_rank": round(rr, 3),
        })
        total_precision += precision
        total_recall += recall
        total_rr += rr

    n = len(eval_set)
    summary = {
        "n_queries": n,
        "k": k,
        "min_score": min_score,
        "mean_precision_at_k": round(total_precision / n, 4) if n else 0.0,
        "mean_recall_at_k": round(total_recall / n, 4) if n else 0.0,
        "mrr": round(total_rr / n, 4) if n else 0.0,
        "per_query": per_query,
    }

    out_path = settings.report_dir / "retrieval_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    log.info(
        "Retrieval eval: P@%d=%.3f, R@%d=%.3f, MRR=%.3f (%d queries) → %s",
        k, summary["mean_precision_at_k"],
        k, summary["mean_recall_at_k"],
        summary["mrr"], n, out_path,
    )
    return summary


if __name__ == "__main__":
    evaluate_retrieval()
