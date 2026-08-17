"""Agent 4 - Retention Strategy Agent.

The most important node. Retrieves relevant playbook chunks from FAISS, asks
the LLM for a structured retention plan via tool-use, extracting the offer_key
directly from the structured response instead of fragile regex matching.
"""
from __future__ import annotations

import time

from src.agents.state import InsightForgeState
from src.llm.client import get_async_llm, get_llm
from src.llm.prompts import RETENTION_SYSTEM, RETENTION_TOOL, build_retention_user_prompt
from src.models.roi_estimator import OFFER_CATALOG
from src.rag.retriever import get_retriever
from src.utils.logger import get_logger

log = get_logger(__name__)


def _fallback_offer_key(segment: str | None) -> str:
    """Segment-based fallback when tool-use is unavailable (DummyLLM / parse failure)."""
    if segment == "High Value + High Risk":
        return "loyalty_discount_15pct_12mo"
    if segment == "Low Value + High Risk":
        return "email_only_discount_5pct"
    return "free_tech_support_6mo"


def _retrieval_query(profile_summary: dict, drivers: list[dict]) -> str:
    parts = [f"segment customer with contract {profile_summary.get('Contract')}"]
    parts.append(f"internet {profile_summary.get('InternetService')}")
    parts.append(f"tenure {profile_summary.get('tenure')} months")
    parts.append(f"sentiment {profile_summary.get('sentiment')}")
    if drivers:
        parts.append("drivers: " + ", ".join(d["pretty"] for d in drivers[:3]))
    return " | ".join(str(p) for p in parts)


def _build_recommendation_markdown(tool_result: dict) -> str:
    """Format the structured tool output as readable markdown."""
    parts = []
    parts.append(f"### Risk Analysis\n{tool_result.get('risk_analysis', '')}\n")
    parts.append(f"### Recommended Action Plan\n1. {tool_result.get('primary_action', '')}")
    for i, action in enumerate(tool_result.get("supporting_actions", []), 2):
        parts.append(f"{i}. {action}")
    parts.append(f"\n### Expected Impact\n- Estimated retention: {tool_result.get('expected_retention_lift', 'unknown')}")
    parts.append(f"\n### Talk Track\n{tool_result.get('talk_track', '')}")
    return "\n".join(parts)


def run(state: InsightForgeState) -> InsightForgeState:
    t0 = time.time()
    trace = list(state.get("trace", []))
    errors = list(state.get("errors", []))
    usage = dict(state.get("llm_usage", {"input_tokens": 0, "output_tokens": 0}))

    profile_summary = state.get("profile_summary") or {}
    drivers = state.get("shap_drivers") or []
    segment = state.get("segment")
    proba = state.get("churn_probability")

    if not profile_summary or proba is None:
        errors.append("retention_agent: missing profile_summary or churn_probability")
        trace.append({"agent": "retention", "ok": False, "ms": int((time.time() - t0) * 1000)})
        return {**state, "errors": errors, "trace": trace}

    try:
        retriever = get_retriever()
        query = _retrieval_query(profile_summary, drivers)
        docs = retriever.search(query, k=4)
        retrieved = [{"source": d.source, "score": d.score, "text": d.text} for d in docs]

        customer_summary = {**profile_summary, "segment": segment, "churn_probability": f"{proba:.0%}"}
        driver_strs = [f"{d['pretty']} (impact {d['contribution']:+.2f})" for d in drivers]
        user_prompt = build_retention_user_prompt(
            customer_summary=customer_summary,
            shap_drivers=driver_strs,
            retrieved_context=[d["text"] for d in retrieved],
        )

        llm = get_llm()

        # Attempt structured tool-use; fall back to plain completion + segment heuristic.
        offer_key = None
        recommendation_md = ""

        if hasattr(llm, "complete_with_tools"):
            resp = llm.complete_with_tools(
                system=RETENTION_SYSTEM,
                user=user_prompt,
                tools=[RETENTION_TOOL],
                max_tokens=1024,
            )
            usage["input_tokens"] += resp.input_tokens
            usage["output_tokens"] += resp.output_tokens

            if resp.tool_result and resp.tool_result.get("offer_key") in OFFER_CATALOG:
                offer_key = resp.tool_result["offer_key"]
                recommendation_md = _build_recommendation_markdown(resp.tool_result)
            elif resp.text:
                recommendation_md = resp.text

        if not offer_key:
            # Fallback: plain completion (DummyLLM or tool-use parse failure)
            if not recommendation_md:
                resp = llm.complete(system=RETENTION_SYSTEM, user=user_prompt, max_tokens=900)
                usage["input_tokens"] += resp.input_tokens
                usage["output_tokens"] += resp.output_tokens
                recommendation_md = resp.text
            offer_key = _fallback_offer_key(segment)

        log.info("RetentionAgent → offer=%s, %d retrieved docs", offer_key, len(retrieved))
        trace.append({"agent": "retention", "ok": True, "ms": int((time.time() - t0) * 1000)})
        return {
            **state,
            "retrieved_docs": retrieved,
            "recommendation_markdown": recommendation_md,
            "primary_offer_key": offer_key,
            "llm_usage": usage,
            "trace": trace,
            "errors": errors,
        }
    except Exception as e:
        log.exception("RetentionAgent failed: %s", e)
        errors.append(f"retention_agent: {e}")
        trace.append({"agent": "retention", "ok": False, "ms": int((time.time() - t0) * 1000)})
        return {**state, "errors": errors, "trace": trace}


async def run_async(state: InsightForgeState) -> InsightForgeState:
    """Async variant for the API path — awaits the LLM call."""
    t0 = time.time()
    trace = list(state.get("trace", []))
    errors = list(state.get("errors", []))
    usage = dict(state.get("llm_usage", {"input_tokens": 0, "output_tokens": 0}))

    profile_summary = state.get("profile_summary") or {}
    drivers = state.get("shap_drivers") or []
    segment = state.get("segment")
    proba = state.get("churn_probability")

    if not profile_summary or proba is None:
        errors.append("retention_agent: missing profile_summary or churn_probability")
        trace.append({"agent": "retention", "ok": False, "ms": int((time.time() - t0) * 1000)})
        return {**state, "errors": errors, "trace": trace}

    try:
        retriever = get_retriever()
        query = _retrieval_query(profile_summary, drivers)
        docs = retriever.search(query, k=4)
        retrieved = [{"source": d.source, "score": d.score, "text": d.text} for d in docs]

        customer_summary = {**profile_summary, "segment": segment, "churn_probability": f"{proba:.0%}"}
        driver_strs = [f"{d['pretty']} (impact {d['contribution']:+.2f})" for d in drivers]
        user_prompt = build_retention_user_prompt(
            customer_summary=customer_summary,
            shap_drivers=driver_strs,
            retrieved_context=[d["text"] for d in retrieved],
        )

        llm = get_async_llm()
        offer_key = None
        recommendation_md = ""

        if hasattr(llm, "complete_with_tools"):
            resp = await llm.complete_with_tools(
                system=RETENTION_SYSTEM,
                user=user_prompt,
                tools=[RETENTION_TOOL],
                max_tokens=1024,
            )
            usage["input_tokens"] += resp.input_tokens
            usage["output_tokens"] += resp.output_tokens

            if resp.tool_result and resp.tool_result.get("offer_key") in OFFER_CATALOG:
                offer_key = resp.tool_result["offer_key"]
                recommendation_md = _build_recommendation_markdown(resp.tool_result)
            elif resp.text:
                recommendation_md = resp.text

        if not offer_key:
            if not recommendation_md:
                resp = await llm.complete(system=RETENTION_SYSTEM, user=user_prompt, max_tokens=900)
                usage["input_tokens"] += resp.input_tokens
                usage["output_tokens"] += resp.output_tokens
                recommendation_md = resp.text
            offer_key = _fallback_offer_key(segment)

        log.info("RetentionAgent async → offer=%s, %d retrieved docs", offer_key, len(retrieved))
        trace.append({"agent": "retention", "ok": True, "ms": int((time.time() - t0) * 1000)})
        return {
            **state,
            "retrieved_docs": retrieved,
            "recommendation_markdown": recommendation_md,
            "primary_offer_key": offer_key,
            "llm_usage": usage,
            "trace": trace,
            "errors": errors,
        }
    except Exception as e:
        log.exception("RetentionAgent async failed: %s", e)
        errors.append(f"retention_agent: {e}")
        trace.append({"agent": "retention", "ok": False, "ms": int((time.time() - t0) * 1000)})
        return {**state, "errors": errors, "trace": trace}
