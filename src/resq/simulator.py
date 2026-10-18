"""Scenario simulator generating incidents, micro-weather, traffic, and emergency response units over time ticks."""

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
import math
import random
from typing import Dict, List, Optional, Tuple

import pandas as pd

from resq.config import (
    DEFAULT_CITY,
    INCIDENT_TYPES,
    ROAD_TYPES,
    SCENARIO_CITY_SURGE,
    SCENARIO_HEAVY_RAIN,
    SCENARIO_HIGHWAY_ACCIDENT,
    SCENARIO_NORMAL,
    UNIT_TYPES,
    CityConfig,
)
from resq.weather import WeatherConditions, get_current_weather


@dataclass
class Incident:
    """Raw emergency incident report."""
    incident_id: str
    tick: int
    timestamp: str
    incident_type: str
    severity: int                  # 1 (low) to 5 (critical)
    latitude: float
    longitude: float
    people_affected: int
    road_type: str
    traffic_congestion: float      # 0.0 (empty) to 1.0 (gridlock)
    temperature_c: float
    rain_mm: float
    wind_kmh: float
    visibility_km: float
    weather_source: str
    # Ground truth synthetic metrics (for training & evaluation validation)
    true_risk_score: float
    is_high_risk: int


@dataclass
class EmergencyUnit:
    """Emergency response unit vehicle and availability state."""
    unit_id: str
    unit_type: str                # 'ambulance', 'fire_engine', 'police_cruiser'
    latitude: float
    longitude: float
    status: str                   # 'AVAILABLE', 'DISPATCHED', 'MAINTENANCE'
    assigned_incident_id: Optional[str] = None
    station_name: str = ""


# Baseline intrinsic severity weight by incident type
TYPE_BASE_RISK: Dict[str, float] = {
    "building_collapse": 0.85,
    "fire": 0.75,
    "flood": 0.70,
    "road_accident": 0.55,
    "medical_emergency": 0.60,
}


def calculate_synthetic_risk(
    incident_type: str,
    severity: int,
    people_affected: int,
    traffic_congestion: float,
    rain_mm: float,
    wind_kmh: float,
    visibility_km: float,
    road_type: str,
    hour: int = 14,
    noise_std: float = 0.04,
    rng: Optional[random.Random] = None,
) -> Tuple[float, int]:
    """Documented synthetic formula for ground-truth risk calculation with realistic noise.
    
    Formula components:
    - Severity contribution: (severity / 5.0) * 0.35
    - Casualty potential: min(people_affected / 30.0, 1.0) * 0.25
    - Base incident danger: TYPE_BASE_RISK[type] * 0.15
    - Weather stress: (min(rain_mm / 50.0, 1.0) * 0.05 + max(0, 10 - visibility_km)/10 * 0.03 + min(wind_kmh/60, 1.0)*0.02)
    - Traffic vulnerability: traffic_congestion * 0.08
    - Road criticality: (0.07 if highway or bridge else 0.02)
    - Time of day: +0.03 if night (22-05) or rush hour (08-10, 17-20)
    + Gaussian noise N(0, noise_std)
    
    Threshold: is_high_risk = 1 if score >= 0.55 else 0
    """
    if rng is None:
        rng = random.Random()

    comp_severity = (severity / 5.0) * 0.35
    comp_casualties = min(people_affected / 30.0, 1.0) * 0.25
    comp_type = TYPE_BASE_RISK.get(incident_type, 0.5) * 0.15

    weather_stress = (
        min(rain_mm / 50.0, 1.0) * 0.05
        + max(0.0, (10.0 - visibility_km) / 10.0) * 0.03
        + min(wind_kmh / 60.0, 1.0) * 0.02
    )

    comp_traffic = traffic_congestion * 0.08
    comp_road = 0.07 if road_type in ("highway", "bridge") else 0.02

    is_rush_or_night = (hour in (8, 9, 10, 17, 18, 19, 20)) or (hour >= 22 or hour <= 5)
    comp_time = 0.03 if is_rush_or_night else 0.0

    raw_score = (
        comp_severity
        + comp_casualties
        + comp_type
        + weather_stress
        + comp_traffic
        + comp_road
        + comp_time
        + rng.gauss(0.0, noise_std)
    )

    clamped_score = max(0.02, min(0.99, round(raw_score, 4)))
    is_high = 1 if clamped_score >= 0.55 else 0
    return clamped_score, is_high


class EmergencySimulator:
    """Stateful tick simulator for emergency incidents and response units."""

    def __init__(
        self,
        city: CityConfig = DEFAULT_CITY,
        scenario: str = SCENARIO_NORMAL,
        seed: int = 42,
        start_time: Optional[datetime] = None,
        num_units: int = 30,
    ):
        self.city = city
        self.scenario = scenario
        self.seed = seed
        self.rng = random.Random(seed)
        self.current_tick = 0
        self.start_time = start_time or datetime(2026, 7, 15, 14, 0, 0)
        self.active_incidents: List[Incident] = []
        self.all_incidents_history: List[Incident] = []
        self.units: List[EmergencyUnit] = []
        self._initialize_fleet(num_units)

    def _initialize_fleet(self, num_units: int) -> None:
        """Create initial distribution of emergency fleet throughout the city."""
        stations = [
            ("Central Station", 28.6250, 77.2180),
            ("North Hub", 28.7100, 77.1600),
            ("East Hub", 28.6300, 77.2900),
            ("South Hub", 28.5300, 77.2200),
            ("West Hub", 28.6400, 77.1100),
            ("Airport Division", 28.5600, 77.1000),
        ]

        type_dist = ["ambulance"] * int(num_units * 0.45) + \
                    ["fire_engine"] * int(num_units * 0.25) + \
                    ["police_cruiser"] * int(num_units * 0.30)
        
        while len(type_dist) < num_units:
            type_dist.append("ambulance")

        self.units = []
        for idx, u_type in enumerate(type_dist[:num_units]):
            st_name, st_lat, st_lon = stations[idx % len(stations)]
            u_lat = st_lat + self.rng.uniform(-0.02, 0.02)
            u_lon = st_lon + self.rng.uniform(-0.02, 0.02)
            self.units.append(
                EmergencyUnit(
                    unit_id=f"{u_type[:3].upper()}-{idx+1:02d}",
                    unit_type=u_type,
                    latitude=round(u_lat, 5),
                    longitude=round(u_lon, 5),
                    status="AVAILABLE",
                    station_name=st_name,
                )
            )

    def step_tick(self, count_override: Optional[int] = None) -> Tuple[List[Incident], WeatherConditions, float]:
        """Advance simulation by 1 tick (15 minutes of simulated time)."""
        self.current_tick += 1
        current_time = self.start_time + timedelta(minutes=15 * self.current_tick)
        hour = current_time.hour

        # Generate weather for tick
        weather = get_current_weather(
            lat=self.city.center_lat,
            lon=self.city.center_lon,
            scenario=self.scenario,
            hour=hour,
            allow_live=False,  # Use reproducible seed-based weather during simulation step
            seed=self.seed + self.current_tick * 17,
        )

        # Congestion modulated by scenario and weather
        base_congestion = 0.35 + 0.30 * math.sin((hour - 7) * math.pi / 8.0)
        if self.scenario == SCENARIO_HEAVY_RAIN:
            congestion = min(0.95, base_congestion + 0.30 + self.rng.uniform(-0.05, 0.10))
        elif self.scenario == SCENARIO_HIGHWAY_ACCIDENT:
            congestion = min(0.92, base_congestion + 0.25 + self.rng.uniform(-0.05, 0.05))
        elif self.scenario == SCENARIO_CITY_SURGE:
            congestion = min(0.98, base_congestion + 0.35 + self.rng.uniform(0.0, 0.05))
        else:
            congestion = max(0.10, min(0.85, base_congestion + self.rng.uniform(-0.08, 0.08)))

        # Incident count per tick
        if count_override is not None:
            num_new = count_override
        elif self.scenario == SCENARIO_CITY_SURGE:
            num_new = self.rng.randint(7, 15)
        elif self.scenario == SCENARIO_HEAVY_RAIN:
            num_new = self.rng.randint(4, 9)
        elif self.scenario == SCENARIO_HIGHWAY_ACCIDENT:
            num_new = self.rng.randint(3, 7)
        else:
            num_new = self.rng.randint(1, 4)

        new_incidents: List[Incident] = []
        for i in range(num_new):
            inc_id = f"INC-{self.current_tick:03d}-{i+1:02d}"

            # Incident type probabilities
            if self.scenario == SCENARIO_HEAVY_RAIN:
                inc_type = self.rng.choices(
                    INCIDENT_TYPES, weights=[0.25, 0.10, 0.40, 0.15, 0.10]
                )[0]
                sev = self.rng.choices([2, 3, 4, 5], weights=[0.2, 0.3, 0.3, 0.2])[0]
            elif self.scenario == SCENARIO_HIGHWAY_ACCIDENT and i == 0:
                inc_type = "road_accident"
                sev = 5
            elif self.scenario == SCENARIO_HIGHWAY_ACCIDENT:
                inc_type = self.rng.choices(
                    INCIDENT_TYPES, weights=[0.60, 0.15, 0.05, 0.15, 0.05]
                )[0]
                sev = self.rng.choices([2, 3, 4, 5], weights=[0.2, 0.3, 0.3, 0.2])[0]
            elif self.scenario == SCENARIO_CITY_SURGE:
                inc_type = self.rng.choice(INCIDENT_TYPES)
                sev = self.rng.choices([1, 2, 3, 4, 5], weights=[0.1, 0.2, 0.3, 0.25, 0.15])[0]
            else:
                inc_type = self.rng.choices(
                    INCIDENT_TYPES, weights=[0.40, 0.15, 0.05, 0.35, 0.05]
                )[0]
                sev = self.rng.choices([1, 2, 3, 4, 5], weights=[0.3, 0.3, 0.25, 0.1, 0.05])[0]

            # Casualties
            if sev >= 4:
                people = self.rng.randint(3, 28)
            elif sev == 3:
                people = self.rng.randint(1, 8)
            else:
                people = self.rng.randint(0, 3)

            # Location within city box
            lat = self.rng.uniform(self.city.lat_min + 0.05, self.city.lat_max - 0.05)
            lon = self.rng.uniform(self.city.lon_min + 0.05, self.city.lon_max - 0.05)

            # Road type
            if self.scenario == SCENARIO_HIGHWAY_ACCIDENT and i == 0:
                road = "highway"
            else:
                road = self.rng.choices(ROAD_TYPES, weights=[0.30, 0.40, 0.25, 0.05])[0]

            true_risk, is_high = calculate_synthetic_risk(
                incident_type=inc_type,
                severity=sev,
                people_affected=people,
                traffic_congestion=congestion,
                rain_mm=weather.rain_mm,
                wind_kmh=weather.wind_kmh,
                visibility_km=weather.visibility_km,
                road_type=road,
                hour=hour,
                rng=self.rng,
            )

            inc = Incident(
                incident_id=inc_id,
                tick=self.current_tick,
                timestamp=current_time.strftime("%Y-%m-%d %H:%M:%S"),
                incident_type=inc_type,
                severity=sev,
                latitude=round(lat, 5),
                longitude=round(lon, 5),
                people_affected=people,
                road_type=road,
                traffic_congestion=round(congestion, 3),
                temperature_c=weather.temperature_c,
                rain_mm=weather.rain_mm,
                wind_kmh=weather.wind_kmh,
                visibility_km=weather.visibility_km,
                weather_source=weather.source_description,
                true_risk_score=true_risk,
                is_high_risk=is_high,
            )
            new_incidents.append(inc)

        self.active_incidents = new_incidents
        self.all_incidents_history.extend(new_incidents)

        return new_incidents, weather, round(congestion, 3)

    def generate_batch_dataset(self, num_ticks: int = 120) -> pd.DataFrame:
        """Generate a sequential time-series dataframe of incidents across multiple ticks."""
        records = []
        for _ in range(num_ticks):
            new_incs, _, _ = self.step_tick()
            for inc in new_incs:
                records.append(asdict(inc))
        return pd.DataFrame(records)
