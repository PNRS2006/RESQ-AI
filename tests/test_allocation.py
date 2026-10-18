"""Unit tests for unit allocation, Haversine distance, and congestion-adjusted ETAs."""

import pandas as pd
from resq.allocation import (
    allocate_units_to_incidents,
    calculate_congestion_adjusted_eta,
    haversine_distance_km,
)
from resq.simulator import EmergencyUnit


def test_haversine_distance_km():
    """Verify Haversine distance gives reasonable spherical distance values."""
    # Delhi Connaught Place to India Gate ~ 2.2 km
    dist = haversine_distance_km(28.6315, 77.2167, 28.6129, 77.2295)
    assert 1.5 <= dist <= 3.0


def test_calculate_congestion_adjusted_eta():
    """Verify congestion slows effective speed and extends ETA."""
    speed_low, eta_low = calculate_congestion_adjusted_eta(10.0, "ambulance", congestion=0.1)
    speed_high, eta_high = calculate_congestion_adjusted_eta(10.0, "ambulance", congestion=0.9)

    assert speed_low > speed_high
    assert eta_high > eta_low
    assert eta_low > 0.0


def test_allocate_units_to_incidents():
    """Verify priority allocation assigns nearest suitable units and identifies unmet demand."""
    units = [
        EmergencyUnit(unit_id="AMB-01", unit_type="ambulance", latitude=28.61, longitude=77.21, status="AVAILABLE"),
        EmergencyUnit(unit_id="FIR-01", unit_type="fire_engine", latitude=28.65, longitude=77.22, status="AVAILABLE"),
    ]

    incidents = pd.DataFrame([
        {"incident_id": "INC-01", "incident_type": "medical_emergency", "ai_rank": 1, "urgency_score": 0.85, "latitude": 28.62, "longitude": 77.21, "traffic_congestion": 0.5},
        {"incident_id": "INC-02", "incident_type": "fire", "ai_rank": 2, "urgency_score": 0.70, "latitude": 28.64, "longitude": 77.22, "traffic_congestion": 0.5},
        {"incident_id": "INC-03", "incident_type": "medical_emergency", "ai_rank": 3, "urgency_score": 0.50, "latitude": 28.60, "longitude": 77.20, "traffic_congestion": 0.5},
    ])

    recs = allocate_units_to_incidents(incidents, units)
    assert len(recs) == 3

    assert recs[0].recommended_unit_id == "AMB-01"
    assert recs[0].unmet_demand is False

    assert recs[1].recommended_unit_id == "FIR-01"
    assert recs[1].unmet_demand is False

    # Fleet exhausted for 3rd incident -> unmet demand
    assert recs[2].recommended_unit_id is None
    assert recs[2].unmet_demand is True
