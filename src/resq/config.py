"""Configuration constants, bounding boxes, scenario definitions, and hyperparameters."""

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class CityConfig:
    """Geographic boundaries and center for incident simulation."""
    name: str = "Delhi"
    lat_min: float = 28.40
    lat_max: float = 28.88
    lon_min: float = 76.84
    lon_max: float = 77.35
    center_lat: float = 28.6139
    center_lon: float = 77.2090
    default_zoom: int = 11


# Incident Types and metadata
INCIDENT_TYPES: List[str] = [
    "road_accident",
    "fire",
    "flood",
    "medical_emergency",
    "building_collapse",
]

ROAD_TYPES: List[str] = ["highway", "arterial", "local", "bridge"]

# Emergency response unit categories
UNIT_TYPES: List[str] = [
    "ambulance",
    "fire_engine",
    "police_cruiser",
]

# Preferred unit category by incident type
INCIDENT_UNIT_MAPPING: Dict[str, str] = {
    "road_accident": "police_cruiser",
    "fire": "fire_engine",
    "flood": "fire_engine",
    "medical_emergency": "ambulance",
    "building_collapse": "fire_engine",
}

# Average cruising speeds (km/h) without traffic congestion
UNIT_BASE_SPEEDS: Dict[str, float] = {
    "ambulance": 48.0,
    "fire_engine": 42.0,
    "police_cruiser": 52.0,
}

# Traffic speed degradation factor (0.0 to 1.0)
CONGESTION_PENALTY_FACTOR: float = 0.70

# Urgency ranking weights (must sum to 1.0)
URGENCY_WEIGHTS = {
    "risk": 0.50,          # ML Risk probability (0-1)
    "casualties": 0.30,    # Normalized people affected
    "time_elapsed": 0.20,  # Normalized elapsed time in queue
}

# Scenarios available in simulator and dashboard
SCENARIO_NORMAL: str = "Normal day"
SCENARIO_HEAVY_RAIN: str = "Heavy rain + floods"
SCENARIO_HIGHWAY_ACCIDENT: str = "Multi-vehicle highway accident"
SCENARIO_CITY_SURGE: str = "City-wide surge (many incidents at once)"

SCENARIO_NAMES: List[str] = [
    SCENARIO_NORMAL,
    SCENARIO_HEAVY_RAIN,
    SCENARIO_HIGHWAY_ACCIDENT,
    SCENARIO_CITY_SURGE,
]

# Paths & persistence defaults
DEFAULT_DB_PATH: str = "resq_emergency.db"
DEFAULT_MODELS_DIR: str = "models/saved"
DEFAULT_RESULTS_PATH: str = "docs/results.md"
DEFAULT_SEED_DATA_PATH: str = "data/sample_incidents.csv"

# Random seed for reproducibility
RANDOM_SEED: int = 42

DEFAULT_CITY = CityConfig()
