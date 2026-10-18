"""Unit tests for FastAPI endpoints using TestClient."""

from fastapi.testclient import TestClient
import pytest
from resq.api import app

client = TestClient(app)


def test_api_health():
    """Verify health endpoint reports operational status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert data["human_in_the_loop_active"] is True


def test_api_platform():
    """Verify platform telemetry endpoint reports hardware info and Snap ML status."""
    response = client.get("/platform")
    assert response.status_code == 200
    data = response.json()
    assert "machine_architecture" in data
    assert "active_inference_engine" in data


def test_api_predict():
    """Verify POST /predict returns risk scores and explainable rationale."""
    payload = {
        "incident_id": "INC-TEST-99",
        "incident_type": "road_accident",
        "severity": 4,
        "latitude": 28.6139,
        "longitude": 77.2090,
        "people_affected": 6,
        "road_type": "highway",
        "traffic_congestion": 0.80,
        "temperature_c": 30.0,
        "rain_mm": 20.0,
        "wind_kmh": 25.0,
        "visibility_km": 3.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["incident_id"] == "INC-TEST-99"
    assert 0.0 <= data["risk_score"] <= 1.0
    assert 0.0 <= data["urgency_score"] <= 1.0
    assert len(data["plain_language_reason"]) > 5
    assert "Decision support only" in data["disclaimer"]


def test_api_decisions():
    """Verify POST /decisions logs human commander sign-off."""
    payload = {
        "incident_id": "INC-TEST-99",
        "commander_id": "COMM-TESTER",
        "decision": "APPROVE",
        "assigned_unit_id": "POL-01",
        "comments": "Approved via automated test.",
    }
    response = client.post("/decisions", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "RECORDED"
    assert data["decision_id"] > 0
