"""Integration tests for all API endpoints using httpx AsyncClient.

Tests the full request/response cycle including model inference, SHAP,
LLM (DummyLLM in CI), and ROI estimation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api.main import app  # noqa: E402


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_health(client):
    r = await client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_metrics_empty(client):
    r = await client.get("/metrics")
    assert r.status_code == 200
    data = r.json()
    assert "total_predictions" in data


@pytest.mark.asyncio
async def test_predict_churn_valid(client, sample_customer_id):
    r = await client.post("/predict_churn", json={"customer_id": sample_customer_id})
    assert r.status_code == 200
    data = r.json()
    assert 0 <= data["churn_probability"] <= 1
    assert data["risk_tier"] in ("High", "Medium", "Low")
    assert data["value_tier"] in ("High", "Medium", "Low")
    assert data["segment"]
    assert data["persona"]


@pytest.mark.asyncio
async def test_predict_churn_404(client):
    r = await client.post("/predict_churn", json={"customer_id": "NONEXISTENT_999"})
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_customer_analysis(client, sample_customer_id):
    r = await client.post("/customer_analysis", json={"customer_id": sample_customer_id})
    assert r.status_code == 200
    data = r.json()
    assert 0 <= data["churn_probability"] <= 1
    assert len(data["drivers"]) == 5
    for d in data["drivers"]:
        assert "pretty" in d
        assert "contribution" in d
    assert data["explanation_text"]


@pytest.mark.asyncio
async def test_customer_analysis_404(client):
    r = await client.post("/customer_analysis", json={"customer_id": "NONEXISTENT_999"})
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_generate_strategy(client, sample_customer_id):
    r = await client.post("/generate_strategy", json={"customer_id": sample_customer_id})
    assert r.status_code == 200
    data = r.json()
    assert data["customer_id"] == sample_customer_id
    assert data["segment"]
    assert data["recommendation_markdown"]
    assert data["primary_offer_key"]


@pytest.mark.asyncio
async def test_customer_roi_with_offer(client, sample_customer_id):
    r = await client.post(
        "/customer_roi",
        json={"customer_id": sample_customer_id, "offer_key": "free_tech_support_6mo"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["chosen"]["offer_key"] == "free_tech_support_6mo"
    assert data["chosen"]["net_value"] is not None
    assert len(data["alternatives"]) > 0


@pytest.mark.asyncio
async def test_customer_roi_without_offer(client, sample_customer_id):
    r = await client.post("/customer_roi", json={"customer_id": sample_customer_id})
    assert r.status_code == 200
    data = r.json()
    assert data["chosen"]["offer_key"]


@pytest.mark.asyncio
async def test_insightforge_run(client, sample_customer_id):
    r = await client.post("/insightforge/run", json={"customer_id": sample_customer_id})
    assert r.status_code == 200
    data = r.json()
    assert data["customer_id"] == sample_customer_id
    assert 0 <= data["churn_probability"] <= 1
    assert data["segment"]
    assert data["persona"]
    assert data["explanation_text"]
    assert len(data["drivers"]) > 0
    assert data["recommendation_markdown"]
    assert data["primary_offer_key"]
    assert data["recommended_roi"]
    assert "llm_usage" in data


@pytest.mark.asyncio
async def test_insightforge_run_404(client):
    r = await client.post("/insightforge/run", json={"customer_id": "NONEXISTENT_999"})
    assert r.status_code == 500 or r.status_code == 404
