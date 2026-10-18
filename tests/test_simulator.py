"""Unit tests for emergency scenario simulator, physics, and ground-truth synthetic formulas."""

import pytest
from resq.config import DEFAULT_CITY, SCENARIO_HEAVY_RAIN, SCENARIO_NORMAL
from resq.simulator import EmergencySimulator, calculate_synthetic_risk


def test_calculate_synthetic_risk():
    """Verify synthetic risk formula produces valid probabilities and respect thresholds."""
    risk_low, is_high_low = calculate_synthetic_risk(
        incident_type="road_accident",
        severity=1,
        people_affected=0,
        traffic_congestion=0.1,
        rain_mm=0.0,
        wind_kmh=5.0,
        visibility_km=15.0,
        road_type="local",
        hour=14,
    )
    assert 0.0 <= risk_low <= 1.0
    assert is_high_low == 0

    risk_high, is_high_high = calculate_synthetic_risk(
        incident_type="building_collapse",
        severity=5,
        people_affected=25,
        traffic_congestion=0.9,
        rain_mm=50.0,
        wind_kmh=45.0,
        visibility_km=1.0,
        road_type="highway",
        hour=23,
    )
    assert 0.0 <= risk_high <= 1.0
    assert risk_high > risk_low
    assert is_high_high == 1


def test_simulator_fleet_initialization():
    """Verify emergency fleet is distributed properly within city bounding box."""
    sim = EmergencySimulator(city=DEFAULT_CITY, num_units=20, seed=42)
    assert len(sim.units) == 20
    assert all(u.status == "AVAILABLE" for u in sim.units)
    for u in sim.units:
        assert DEFAULT_CITY.lat_min <= u.latitude <= DEFAULT_CITY.lat_max
        assert DEFAULT_CITY.lon_min <= u.longitude <= DEFAULT_CITY.lon_max


def test_simulator_step_tick():
    """Verify simulator advances ticks and returns valid incidents."""
    sim = EmergencySimulator(city=DEFAULT_CITY, scenario=SCENARIO_NORMAL, seed=42)
    incs, weather, congestion = sim.step_tick()

    assert sim.current_tick == 1
    assert len(incs) >= 1
    assert 0.0 <= congestion <= 1.0
    assert weather.temperature_c > -20.0

    inc = incs[0]
    assert inc.severity in [1, 2, 3, 4, 5]
    assert 0.0 <= inc.true_risk_score <= 1.0
    assert inc.is_high_risk in [0, 1]


def test_simulator_batch_generation():
    """Verify time-series batch dataset generation."""
    sim = EmergencySimulator(scenario=SCENARIO_HEAVY_RAIN, seed=101)
    df = sim.generate_batch_dataset(num_ticks=5)
    assert len(df) >= 10
    assert "true_risk_score" in df.columns
    assert "is_high_risk" in df.columns
