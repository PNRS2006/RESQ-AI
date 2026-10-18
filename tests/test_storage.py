"""Unit tests for SQLite storage and human decision audit trail."""

import pytest
from resq.storage import ResQStorage


def test_storage_lifecycle(tmp_path):
    """Verify incidents, recommendations, and decisions are stored and retrieved accurately."""
    db_file = tmp_path / "test_resq.db"
    store = ResQStorage(db_path=str(db_file))

    # 1. Save incidents
    incidents = [
        {
            "incident_id": "INC-TEST-01",
            "tick": 1,
            "incident_type": "fire",
            "severity": 4,
            "latitude": 28.61,
            "longitude": 77.20,
            "risk_score": 0.78,
            "urgency_score": 0.82,
        }
    ]
    saved = store.save_incidents(incidents)
    assert saved == 1

    fetched = store.get_incidents()
    assert len(fetched) == 1
    assert fetched[0]["incident_id"] == "INC-TEST-01"

    # 2. Save recommendation
    recs = [
        {
            "incident_id": "INC-TEST-01",
            "recommended_unit_id": "FIR-01",
            "unit_type": "fire_engine",
            "distance_km": 3.4,
            "effective_speed_kmh": 35.0,
            "eta_minutes": 5.8,
            "plain_reason": "Active fire hazard.",
        }
    ]
    store.save_recommendations(recs)
    saved_recs = store.get_recommendations("INC-TEST-01")
    assert len(saved_recs) == 1
    assert saved_recs[0]["recommended_unit_id"] == "FIR-01"

    # 3. Log human decision
    dec_id = store.log_human_decision(
        incident_id="INC-TEST-01",
        commander_id="COMM-CHIEF",
        decision="APPROVE",
        assigned_unit_id="FIR-01",
        comments="Approved priority fire response.",
    )
    assert dec_id > 0

    history = store.get_decision_history()
    assert len(history) == 1
    assert history[0]["decision"] == "APPROVE"
    assert history[0]["commander_id"] == "COMM-CHIEF"

    # Verify incident status updated to APPROVED
    updated_inc = store.get_incidents()
    assert updated_inc[0]["status"] == "APPROVED"
