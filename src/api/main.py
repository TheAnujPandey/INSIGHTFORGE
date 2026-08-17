"""FastAPI surface.

Endpoints map 1:1 to the project spec:
- POST /predict_churn       - churn probability + segment
- POST /customer_analysis   - SHAP-driven explanation
- POST /generate_strategy   - LLM + RAG recommendation
- POST /customer_roi        - ROI table for actions
- POST /insightforge/run         - full multi-agent pipeline

Plus GET /health and GET /metrics.
"""
from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.agents.orchestrator import run as run_insightforge, run_async as run_insightforge_async
from src.api.metrics import PredictionLog, get_metrics_summary, log_prediction
from src.api.schemas import (
    ChurnResponse,
    InsightForgeResponse,
    CustomerAnalysisResponse,
    CustomerRequest,
    ROIRequest,
    ROIResponse,
    ROIResponseItem,
    ShapDriver,
    StrategyRequest,
    StrategyResponse,
)
from src.data.loader import get_customer
from src.data.preprocessor import load_preprocessor, transform_one
from src.explainability.shap_explainer import load_explainer_cache
from src.features.engineering import risk_tier, segment_label, value_tier
from src.models.churn_predictor import load_production
from src.models.roi_estimator import estimate, rank_offers, to_dict
from src.models.segmentation import BEHAVIOURAL_COLS, load_bundle
from src.utils.logger import get_logger

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy artifacts once at startup; store on app.state."""
    log.info("Lifespan: loading model + preprocessor + explainer...")
    app.state.model_bundle = load_production()
    app.state.preprocessor = load_preprocessor()
    try:
        app.state.explainer = load_explainer_cache()
    except Exception as e:
        log.warning("Lifespan: explainer not pre-loaded: %s", e)
        app.state.explainer = None
    yield


app = FastAPI(
    title="INSIGHTFORGE",
    version="0.1.0",
    description="INSIGHTFORGE: churn prediction + SHAP + LLM/RAG + LangGraph multi-agent retention intelligence.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _profile_or_404(customer_id: str) -> dict:
    p = get_customer(customer_id)
    if p is None:
        raise HTTPException(status_code=404, detail=f"customer_id {customer_id} not found")
    return p


def _persona_for(profile: dict) -> str:
    """Look up the KMeans persona for a single customer. Returns "Unknown" if the
    segmentation bundle is unavailable or any feature is missing - never raises,
    so it can't break /predict_churn."""
    try:
        import numpy as np

        bundle = load_bundle()
        row = np.array([[profile[c] for c in BEHAVIOURAL_COLS]], dtype=float)
        cluster = int(bundle.kmeans.predict(bundle.scaler.transform(row))[0])
        return bundle.cluster_personas.get(cluster, "Unknown")
    except Exception as e:
        log.warning("persona lookup failed: %s", e)
        return "Unknown"


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> dict:
    """Aggregated prediction metrics for monitoring dashboards."""
    return get_metrics_summary()


@app.post("/predict_churn", response_model=ChurnResponse)
async def predict_churn(req: CustomerRequest) -> ChurnResponse:
    t0 = time.time()
    profile = _profile_or_404(req.customer_id)
    bundle = app.state.model_bundle or load_production()
    pre = app.state.preprocessor or load_preprocessor()
    X = await asyncio.to_thread(transform_one, profile, pre)
    proba = float(bundle["model"].predict_proba(X)[0, 1])
    vt = value_tier(float(profile["MonthlyCharges"]), int(profile["tenure"]))
    rt = risk_tier(proba)
    seg = segment_label(vt, rt)
    log_prediction(PredictionLog(
        timestamp=datetime.now(timezone.utc).isoformat(),
        customer_id=req.customer_id,
        churn_probability=round(proba, 4),
        segment=seg,
        risk_tier=rt,
        latency_ms=int((time.time() - t0) * 1000),
        endpoint="/predict_churn",
    ))
    return ChurnResponse(
        customer_id=req.customer_id,
        churn_probability=round(proba, 4),
        risk_tier=rt,
        value_tier=vt,
        segment=seg,
        persona=_persona_for(profile),
    )


@app.post("/customer_analysis", response_model=CustomerAnalysisResponse)
async def customer_analysis(req: CustomerRequest) -> CustomerAnalysisResponse:
    profile = _profile_or_404(req.customer_id)
    explainer = app.state.explainer or load_explainer_cache()
    expl = await asyncio.to_thread(explainer.explain_record, profile, 5)
    drivers = [
        ShapDriver(pretty=d.pretty, contribution=round(d.contribution, 4), value=str(d.value) if d.value is not None else None)
        for d in expl.drivers
    ]
    text = (
        f"This customer has a churn probability of {expl.prediction:.0%}. "
        f"The strongest driver is '{drivers[0].pretty if drivers else 'unknown'}'."
    )
    return CustomerAnalysisResponse(
        customer_id=req.customer_id,
        churn_probability=round(expl.prediction, 4),
        explanation_text=text,
        drivers=drivers,
    )


@app.post("/generate_strategy", response_model=StrategyResponse)
async def generate_strategy(req: StrategyRequest) -> StrategyResponse:
    state = await run_insightforge_async(req.customer_id)
    if state.get("errors"):
        log.warning("/generate_strategy errors: %s", state["errors"])
    return StrategyResponse(
        customer_id=req.customer_id,
        segment=state.get("segment", "unknown"),
        recommendation_markdown=state.get("recommendation_markdown", ""),
        primary_offer_key=state.get("primary_offer_key", ""),
        retrieved_sources=list({d["source"] for d in state.get("retrieved_docs", [])}),
    )


@app.post("/customer_roi", response_model=ROIResponse)
async def customer_roi(req: ROIRequest) -> ROIResponse:
    profile = _profile_or_404(req.customer_id)
    monthly = float(profile["MonthlyCharges"])
    if req.offer_key:
        bundle = app.state.model_bundle or load_production()
        pre = app.state.preprocessor or load_preprocessor()
        X = await asyncio.to_thread(transform_one, profile, pre)
        proba = float(bundle["model"].predict_proba(X)[0, 1])
        chosen = estimate(req.offer_key, monthly_charge=monthly, baseline_churn_prob=proba)
        ranked = rank_offers(monthly_charge=monthly, baseline_churn_prob=proba, exclude={req.offer_key})
    else:
        state = await run_insightforge_async(req.customer_id)
        offer_key = state.get("primary_offer_key", "free_tech_support_6mo")
        proba = state.get("churn_probability", 0.5)
        chosen = estimate(offer_key, monthly_charge=monthly, baseline_churn_prob=proba)
        ranked = rank_offers(monthly_charge=monthly, baseline_churn_prob=proba, exclude={offer_key})

    return ROIResponse(
        customer_id=req.customer_id,
        chosen=ROIResponseItem(**to_dict(chosen)),
        alternatives=[ROIResponseItem(**to_dict(e)) for e in ranked],
    )


@app.post("/insightforge/run", response_model=InsightForgeResponse)
async def insightforge_run(req: CustomerRequest) -> InsightForgeResponse:
    """Full multi-agent flow: Profile → Risk → Explanation → Retention → ROI."""
    t0 = time.time()
    state = await run_insightforge_async(req.customer_id)
    if "churn_probability" not in state:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {state.get('errors')}")
    drivers = [ShapDriver(**d) for d in state.get("shap_drivers", [])]
    rec_roi = state.get("recommended_roi") or {}
    alt = state.get("roi_estimates") or []
    response = InsightForgeResponse(
        customer_id=req.customer_id,
        churn_probability=round(state["churn_probability"], 4),
        segment=state.get("segment", "unknown"),
        persona=state.get("persona", "unknown"),
        explanation_text=state.get("explanation_text", ""),
        drivers=drivers,
        recommendation_markdown=state.get("recommendation_markdown", ""),
        primary_offer_key=state.get("primary_offer_key", ""),
        recommended_roi=ROIResponseItem(**rec_roi) if rec_roi else ROIResponseItem(
            offer_key="", description="", expected_revenue_saved=0, offer_cost=0,
            net_value=0, horizon_months=12,
        ),
        alternatives=[ROIResponseItem(**a) for a in alt if a.get("offer_key") != state.get("primary_offer_key")],
        retrieved_sources=list({d["source"] for d in state.get("retrieved_docs", [])}),
        llm_usage=state.get("llm_usage", {}),
        errors=state.get("errors", []),
    )
    llm_usage = state.get("llm_usage", {})
    log_prediction(PredictionLog(
        timestamp=datetime.now(timezone.utc).isoformat(),
        customer_id=req.customer_id,
        churn_probability=round(state["churn_probability"], 4),
        segment=state.get("segment", "unknown"),
        risk_tier=state.get("risk_tier", "unknown"),
        llm_input_tokens=llm_usage.get("input_tokens", 0),
        llm_output_tokens=llm_usage.get("output_tokens", 0),
        latency_ms=int((time.time() - t0) * 1000),
        endpoint="/insightforge/run",
    ))
    return response
