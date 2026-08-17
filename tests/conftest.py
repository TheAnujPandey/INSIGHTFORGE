"""Shared fixtures for integration tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session", autouse=True)
def bootstrap_artifacts():
    """Ensure model artifacts are present for the entire test session."""
    from src.bootstrap import ensure_artifacts, ensure_faiss

    ensure_artifacts()
    ensure_faiss()


@pytest.fixture(scope="session")
def sample_customer_id():
    """Return the ID of the first customer in the dataset."""
    from src.data.loader import load_enhanced

    df = load_enhanced()
    return df["customerID"].iloc[0]
