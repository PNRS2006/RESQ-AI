"""Unit tests for data cleaning, feature engineering, and temporal splits."""

import pandas as pd
from resq.config import DEFAULT_CITY
from resq.data_prep import (
    FEATURE_COLUMNS,
    engineer_features,
    prepare_feature_matrix,
    temporal_train_test_split,
    validate_and_clean_incident_data,
)


def test_validate_and_clean_incident_data():
    """Verify data validation handles missing, dirty, and out-of-bound records safely."""
    raw_dirty = pd.DataFrame([{
        "severity": 99,              # out of bounds -> should clamp to 5
        "people_affected": -10,      # negative -> should clamp to 0
        "traffic_congestion": 2.5,   # > 1.0 -> should clamp to 1.0
        "latitude": 5.0,             # outside city box -> should clamp to city bounds
        "longitude": 120.0,
        "incident_type": "unknown_incident",  # unknown -> fallback to default
    }])

    cleaned = validate_and_clean_incident_data(raw_dirty, city=DEFAULT_CITY)
    assert cleaned["severity"].iloc[0] == 5
    assert cleaned["people_affected"].iloc[0] == 0
    assert cleaned["traffic_congestion"].iloc[0] == 1.0
    assert cleaned["latitude"].iloc[0] == DEFAULT_CITY.lat_min
    assert cleaned["longitude"].iloc[0] == DEFAULT_CITY.lon_max
    assert cleaned["incident_type"].iloc[0] == "road_accident"


def test_prepare_feature_matrix():
    """Verify engineered feature matrix contains all required features without NaNs."""
    df = pd.DataFrame([{
        "severity": 4,
        "people_affected": 8,
        "traffic_congestion": 0.8,
        "temperature_c": 32.0,
        "rain_mm": 20.0,
        "wind_kmh": 25.0,
        "visibility_km": 4.0,
        "road_type": "highway",
        "incident_type": "fire",
        "timestamp": "2026-07-15 17:30:00",
    }])

    X = prepare_feature_matrix(df)
    assert list(X.columns) == FEATURE_COLUMNS
    assert X.isna().sum().sum() == 0
    assert X["rain_x_congestion"].iloc[0] == 20.0 * 0.8
    assert X["type_fire"].iloc[0] == 1
    assert X["road_highway"].iloc[0] == 1


def test_temporal_train_test_split():
    """Verify temporal train/test split maintains strict chronological order without leakage."""
    dates = pd.date_range("2026-01-01", periods=100, freq="h")
    df = pd.DataFrame({
        "timestamp": dates.strftime("%Y-%m-%d %H:%M:%S"),
        "val": range(100),
    })

    train_df, test_df = temporal_train_test_split(df, test_ratio=0.20)
    assert len(train_df) == 80
    assert len(test_df) == 20
    assert train_df["val"].max() < test_df["val"].min()
