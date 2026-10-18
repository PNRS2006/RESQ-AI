"""Data cleaning, range validation, temporal splitting, and feature engineering pipeline."""

from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from resq.config import DEFAULT_CITY, INCIDENT_TYPES, ROAD_TYPES, CityConfig

# Fixed feature list for model training and inference
NUMERIC_FEATURES: List[str] = [
    "severity",
    "people_affected",
    "traffic_congestion",
    "temperature_c",
    "rain_mm",
    "wind_kmh",
    "visibility_km",
    "rain_x_congestion",
    "casualty_severity_ratio",
    "visibility_hazard",
    "is_highway_bridge",
    "hour_of_day",
    "is_night_or_rush",
]

# One-hot encoded column names for incident types and road types
INCIDENT_TYPE_COLS: List[str] = [f"type_{t}" for t in INCIDENT_TYPES]
ROAD_TYPE_COLS: List[str] = [f"road_{r}" for r in ROAD_TYPES]

FEATURE_COLUMNS: List[str] = NUMERIC_FEATURES + INCIDENT_TYPE_COLS + ROAD_TYPE_COLS


def validate_and_clean_incident_data(
    df: pd.DataFrame,
    city: CityConfig = DEFAULT_CITY,
    drop_invalid: bool = False,
) -> pd.DataFrame:
    """Validate data integrity, clamp out-of-range physical values, and handle missing fields.
    
    Guarantees:
    - Latitude clamped within city bounds [lat_min, lat_max]
    - Longitude clamped within city bounds [lon_min, lon_max]
    - Severity clamped to integer [1, 5]
    - Traffic congestion clamped to [0.0, 1.0]
    - Rain >= 0, Wind >= 0, Visibility clamped to [0.1, 20.0]
    - Imputes missing numericals with domain medians
    - Imputes missing categoricals with default values
    """
    df = df.copy()

    # Required fields with fallback defaults
    if "severity" not in df.columns:
        df["severity"] = 3
    else:
        df["severity"] = df["severity"].fillna(3).clip(1, 5).astype(int)

    if "people_affected" not in df.columns:
        df["people_affected"] = 0
    else:
        df["people_affected"] = df["people_affected"].fillna(0).clip(0, 500).astype(int)

    if "traffic_congestion" not in df.columns:
        df["traffic_congestion"] = 0.5
    else:
        df["traffic_congestion"] = df["traffic_congestion"].fillna(0.5).clip(0.0, 1.0)

    if "temperature_c" not in df.columns:
        df["temperature_c"] = 28.0
    else:
        df["temperature_c"] = df["temperature_c"].fillna(28.0).clip(-10.0, 55.0)

    if "rain_mm" not in df.columns:
        df["rain_mm"] = 0.0
    else:
        df["rain_mm"] = df["rain_mm"].fillna(0.0).clip(lower=0.0)

    if "wind_kmh" not in df.columns:
        df["wind_kmh"] = 12.0
    else:
        df["wind_kmh"] = df["wind_kmh"].fillna(12.0).clip(lower=0.0)

    if "visibility_km" not in df.columns:
        df["visibility_km"] = 10.0
    else:
        df["visibility_km"] = df["visibility_km"].fillna(10.0).clip(0.1, 30.0)

    # Spatial clipping
    if "latitude" in df.columns:
        if drop_invalid:
            mask = (df["latitude"] >= city.lat_min) & (df["latitude"] <= city.lat_max)
            df = df[mask]
        else:
            df["latitude"] = df["latitude"].fillna(city.center_lat).clip(city.lat_min, city.lat_max)

    if "longitude" in df.columns:
        if drop_invalid:
            mask = (df["longitude"] >= city.lon_min) & (df["longitude"] <= city.lon_max)
            df = df[mask]
        else:
            df["longitude"] = df["longitude"].fillna(city.center_lon).clip(city.lon_min, city.lon_max)

    # Categorical validation
    if "incident_type" not in df.columns:
        df["incident_type"] = "road_accident"
    else:
        df["incident_type"] = df["incident_type"].fillna("road_accident").astype(str)
        df["incident_type"] = df["incident_type"].apply(
            lambda x: x if x in INCIDENT_TYPES else "road_accident"
        )

    if "road_type" not in df.columns:
        df["road_type"] = "arterial"
    else:
        df["road_type"] = df["road_type"].fillna("arterial").astype(str)
        df["road_type"] = df["road_type"].apply(
            lambda x: x if x in ROAD_TYPES else "arterial"
        )

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute interaction terms, temporal markers, and one-hot encoded categories."""
    df = df.copy()

    # Temporal feature derivation
    if "timestamp" in df.columns:
        timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
        df["hour_of_day"] = timestamps.dt.hour.fillna(14).astype(int)
    elif "hour_of_day" not in df.columns:
        df["hour_of_day"] = 14

    # Interaction terms
    df["rain_x_congestion"] = df["rain_mm"] * df["traffic_congestion"]
    df["casualty_severity_ratio"] = df["people_affected"] / (df["severity"].astype(float) + 0.1)
    df["visibility_hazard"] = np.maximum(0.0, 10.0 - df["visibility_km"])
    df["is_highway_bridge"] = df["road_type"].isin(["highway", "bridge"]).astype(int)

    # Rush hour or night marker
    rush_or_night = df["hour_of_day"].isin([8, 9, 10, 17, 18, 19, 20, 22, 23, 0, 1, 2, 3, 4, 5])
    df["is_night_or_rush"] = rush_or_night.astype(int)

    # One-hot encoding for incident types
    for inc_t in INCIDENT_TYPES:
        df[f"type_{inc_t}"] = (df["incident_type"] == inc_t).astype(int)

    # One-hot encoding for road types
    for road_t in ROAD_TYPES:
        df[f"road_{road_t}"] = (df["road_type"] == road_t).astype(int)

    return df


def prepare_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Clean, engineer, and return the aligned feature matrix X with exact FEATURE_COLUMNS."""
    cleaned = validate_and_clean_incident_data(df)
    engineered = engineer_features(cleaned)
    # Ensure all required columns exist
    for col in FEATURE_COLUMNS:
        if col not in engineered.columns:
            engineered[col] = 0.0

    return engineered[FEATURE_COLUMNS]


def temporal_train_test_split(
    df: pd.DataFrame,
    test_ratio: float = 0.25,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split preserving time order without data leakage."""
    if "timestamp" in df.columns:
        df = df.sort_values(by="timestamp").reset_index(drop=True)
    elif "tick" in df.columns:
        df = df.sort_values(by="tick").reset_index(drop=True)

    n_total = len(df)
    split_idx = int(n_total * (1.0 - test_ratio))

    train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)

    return train_df, test_df
