"""Congestion-aware emergency unit allocation and ETA calculation engine."""

from dataclasses import asdict, dataclass
import math
from typing import Dict, List, Optional, Tuple

import pandas as pd

from resq.config import (
    CONGESTION_PENALTY_FACTOR,
    INCIDENT_UNIT_MAPPING,
    UNIT_BASE_SPEEDS,
)
from resq.simulator import EmergencyUnit


@dataclass
class DispatchRecommendation:
    """Resource dispatch recommendation prepared for human commander sign-off."""
    incident_id: str
    incident_type: str
    urgency_rank: int
    urgency_score: float
    recommended_unit_id: Optional[str]
    unit_type: Optional[str]
    station_name: Optional[str]
    unit_lat: Optional[float]
    unit_lon: Optional[float]
    incident_lat: float
    incident_lon: float
    distance_km: Optional[float]
    effective_speed_kmh: Optional[float]
    estimated_eta_minutes: Optional[float]
    unmet_demand: bool
    status: str = "PROPOSED"  # System NEVER dispatches; requires human commander approval


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate Great Circle distance in kilometers between two GPS coordinates."""
    r_earth = 6371.0  # Earth radius in kilometers

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = math.sin(d_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

    return round(r_earth * c, 2)


def calculate_congestion_adjusted_eta(
    distance_km: float,
    unit_type: str,
    congestion: float,
    penalty_factor: float = CONGESTION_PENALTY_FACTOR,
) -> Tuple[float, float]:
    """Calculate effective speed and travel ETA under current traffic congestion.
    
    Formula:
      effective_speed = max(10.0, base_speed * (1.0 - penalty_factor * congestion))
      eta_minutes = (distance_km / effective_speed) * 60.0
    """
    base_speed = UNIT_BASE_SPEEDS.get(unit_type, 45.0)
    effective_speed = max(10.0, base_speed * (1.0 - (penalty_factor * congestion)))
    eta_mins = (distance_km / effective_speed) * 60.0
    return round(effective_speed, 1), round(max(1.0, eta_mins), 1)


def allocate_units_to_incidents(
    ranked_incidents: pd.DataFrame,
    units: List[EmergencyUnit],
    penalty_factor: float = CONGESTION_PENALTY_FACTOR,
) -> List[DispatchRecommendation]:
    """Greedy priority-ordered allocation recommending nearest available suitable unit per incident.
    
    High-urgency incidents receive first pick of available fleet.
    When suitable units are exhausted, incidents are flagged with unmet_demand=True.
    """
    recommendations: List[DispatchRecommendation] = []
    available_units = {u.unit_id: u for u in units if u.status == "AVAILABLE"}

    for idx, row in ranked_incidents.iterrows():
        inc_id = str(row["incident_id"])
        inc_type = str(row.get("incident_type", "road_accident"))
        urgency_rank = int(row.get("ai_rank", idx + 1))
        urgency_score = float(row.get("urgency_score", 0.5))
        inc_lat = float(row["latitude"])
        inc_lon = float(row["longitude"])
        congestion = float(row.get("traffic_congestion", 0.5))

        preferred_unit_type = INCIDENT_UNIT_MAPPING.get(inc_type, "ambulance")

        # Find closest available unit matching preferred type
        best_unit_id: Optional[str] = None
        min_dist: float = float("inf")

        for u_id, unit in available_units.items():
            if unit.unit_type == preferred_unit_type:
                dist = haversine_distance_km(inc_lat, inc_lon, unit.latitude, unit.longitude)
                if dist < min_dist:
                    min_dist = dist
                    best_unit_id = u_id

        # Fallback to any available unit if preferred type is depleted
        if best_unit_id is None and available_units:
            for u_id, unit in available_units.items():
                dist = haversine_distance_km(inc_lat, inc_lon, unit.latitude, unit.longitude)
                if dist < min_dist:
                    min_dist = dist
                    best_unit_id = u_id

        if best_unit_id is not None:
            chosen_unit = available_units.pop(best_unit_id)
            eff_speed, eta = calculate_congestion_adjusted_eta(
                min_dist, chosen_unit.unit_type, congestion, penalty_factor
            )
            rec = DispatchRecommendation(
                incident_id=inc_id,
                incident_type=inc_type,
                urgency_rank=urgency_rank,
                urgency_score=urgency_score,
                recommended_unit_id=chosen_unit.unit_id,
                unit_type=chosen_unit.unit_type,
                station_name=chosen_unit.station_name,
                unit_lat=chosen_unit.latitude,
                unit_lon=chosen_unit.longitude,
                incident_lat=inc_lat,
                incident_lon=inc_lon,
                distance_km=min_dist,
                effective_speed_kmh=eff_speed,
                estimated_eta_minutes=eta,
                unmet_demand=False,
                status="PROPOSED",
            )
        else:
            rec = DispatchRecommendation(
                incident_id=inc_id,
                incident_type=inc_type,
                urgency_rank=urgency_rank,
                urgency_score=urgency_score,
                recommended_unit_id=None,
                unit_type=None,
                station_name=None,
                unit_lat=None,
                unit_lon=None,
                incident_lat=inc_lat,
                incident_lon=inc_lon,
                distance_km=None,
                effective_speed_kmh=None,
                estimated_eta_minutes=None,
                unmet_demand=True,
                status="PROPOSED",
            )

        recommendations.append(rec)

    return recommendations
