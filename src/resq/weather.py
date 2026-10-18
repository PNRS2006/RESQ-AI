"""Weather telemetry module fetching live conditions from Open-Meteo with fallback to simulation."""

from dataclasses import dataclass
import logging
import math
import random
from typing import Optional
import requests

from resq.config import DEFAULT_CITY

logger = logging.getLogger(__name__)


@dataclass
class WeatherConditions:
    """Current weather metrics at a specific location."""
    temperature_c: float
    rain_mm: float
    wind_kmh: float
    visibility_km: float
    is_live: bool
    source_description: str


def fetch_live_weather(
    lat: float = DEFAULT_CITY.center_lat,
    lon: float = DEFAULT_CITY.center_lon,
    timeout_sec: float = 2.0,
) -> Optional[WeatherConditions]:
    """Attempt to fetch live current weather from the free Open-Meteo API.
    
    Returns WeatherConditions if successful, or None if the network request fails.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "current": "temperature_2m,precipitation,wind_speed_10m,visibility",
    }
    try:
        response = requests.get(url, params=params, timeout=timeout_sec)
        if response.status_code == 200:
            data = response.json()
            current = data.get("current", {})
            temp = float(current.get("temperature_2m", 28.0))
            rain = float(current.get("precipitation", 0.0))
            wind = float(current.get("wind_speed_10m", 12.0))
            # Visibility is in meters from open-meteo, convert to km
            vis_m = current.get("visibility", 10000.0)
            vis_km = float(vis_m) / 1000.0 if vis_m is not None else 10.0

            return WeatherConditions(
                temperature_c=temp,
                rain_mm=rain,
                wind_kmh=wind,
                visibility_km=max(0.1, vis_km),
                is_live=True,
                source_description="Open-Meteo Live API",
            )
    except Exception as exc:
        logger.debug("Live weather fetch failed (%s); falling back to simulation.", exc)
    return None


def get_simulated_weather(
    hour: int = 14,
    scenario: str = "Normal day",
    seed: Optional[int] = None,
) -> WeatherConditions:
    """Generate realistic synthetic weather based on hour of day and scenario context."""
    rng = random.Random(seed) if seed is not None else random.Random()

    # Base diurnal temperature wave
    base_temp = 25.0 + 8.0 * math.sin((hour - 8) * math.pi / 12.0) + rng.uniform(-1.5, 1.5)

    if scenario == "Heavy rain + floods":
        rain = rng.uniform(25.0, 65.0)
        wind = rng.uniform(28.0, 55.0)
        visibility = rng.uniform(0.8, 3.5)
        temp = max(18.0, base_temp - 6.0)
    elif scenario == "Multi-vehicle highway accident":
        rain = rng.uniform(4.0, 15.0)
        wind = rng.uniform(15.0, 30.0)
        visibility = rng.uniform(1.2, 4.5)  # reduced visibility causes accidents
        temp = base_temp
    elif scenario == "City-wide surge (many incidents at once)":
        rain = rng.uniform(10.0, 30.0)
        wind = rng.uniform(20.0, 40.0)
        visibility = rng.uniform(2.5, 6.0)
        temp = base_temp
    else:  # Normal day
        rain = rng.uniform(0.0, 1.5) if rng.random() > 0.85 else 0.0
        wind = rng.uniform(5.0, 18.0)
        visibility = rng.uniform(7.0, 12.0)
        temp = base_temp

    return WeatherConditions(
        temperature_c=round(temp, 1),
        rain_mm=round(rain, 1),
        wind_kmh=round(wind, 1),
        visibility_km=round(visibility, 1),
        is_live=False,
        source_description="Simulated Weather (Realistic Physics Model)",
    )


def get_current_weather(
    lat: float = DEFAULT_CITY.center_lat,
    lon: float = DEFAULT_CITY.center_lon,
    scenario: str = "Normal day",
    hour: int = 14,
    allow_live: bool = True,
    seed: Optional[int] = None,
) -> WeatherConditions:
    """Get current weather: tries live Open-Meteo first, gracefully falling back to simulation."""
    if allow_live and scenario == "Normal day":
        live = fetch_live_weather(lat=lat, lon=lon)
        if live is not None:
            return live

    return get_simulated_weather(hour=hour, scenario=scenario, seed=seed)
