"""Cliente OpenWeatherMap para condiciones climáticas del venue al kickoff.

API free tier: 1000 req/día. Usamos /forecast (5 días por 3h) que devuelve hasta 40 entries.
Si el kickoff está dentro de 5 días, agarramos el entry más cercano al kickoff_utc.
Si está más lejos (>5 días), no podemos forecastear todavía y devolvemos None.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import yaml

log = logging.getLogger(__name__)

OWM_BASE = "https://api.openweathermap.org/data/2.5"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENUES_PATH = PROJECT_ROOT / "config" / "venues.yaml"


def _key() -> str | None:
    k = os.environ.get("OPENWEATHER_API_KEY", "")
    return k if k and not k.startswith("xxx") else None


def _load_venues() -> dict:
    if not VENUES_PATH.exists():
        return {}
    return yaml.safe_load(VENUES_PATH.read_text()) or {}


def resolve_venue(venue_str: str | None) -> dict | None:
    """Mapea el string del venue (como aparece en fixtures.yaml) a coordenadas."""
    if not venue_str:
        return None
    cfg = _load_venues()
    venues = cfg.get("venues", {})
    name_to_key = cfg.get("name_to_key", {})

    # 1. Match directo en name_to_key
    if venue_str in name_to_key:
        key = name_to_key[venue_str]
        return venues.get(key)

    # 2. Buscar por nombre dentro del string
    vlow = venue_str.lower()
    for key, venue in venues.items():
        if venue["name"].lower() in vlow or key.replace("_", " ") in vlow:
            return venue
    return None


def get_forecast_at_kickoff(
    lat: float,
    lon: float,
    kickoff_utc: datetime,
) -> dict | None:
    """Devuelve forecast del clima en (lat, lon) cerca de kickoff_utc.

    Returns: dict con {temp_c, feels_like, humidity, wind_speed, conditions, rain_3h, snow_3h}
             o None si la API falla o el kickoff está muy lejos.
    """
    key = _key()
    if not key:
        return None

    now = datetime.now(timezone.utc)
    delta_h = (kickoff_utc - now).total_seconds() / 3600
    if delta_h > 120:   # >5 días, forecast no llega
        log.info("Kickoff a %.0fh — forecast no disponible (max 5 días)", delta_h)
        return None
    if delta_h < -3:    # ya pasó
        return None

    try:
        with httpx.Client(timeout=10.0) as c:
            r = c.get(f"{OWM_BASE}/forecast", params={
                "lat": lat,
                "lon": lon,
                "appid": key,
                "units": "metric",
            })
        if r.status_code != 200:
            log.warning("OpenWeatherMap %d: %s", r.status_code, r.text[:200])
            return None
        data = r.json()
    except Exception as e:
        log.warning("OpenWeatherMap falló: %s", e)
        return None

    # Encontrar entry más cercana al kickoff
    items = data.get("list", [])
    if not items:
        return None

    best = min(items, key=lambda x: abs(x["dt"] - kickoff_utc.timestamp()))
    return {
        "temp_c": best["main"].get("temp"),
        "feels_like_c": best["main"].get("feels_like"),
        "humidity_pct": best["main"].get("humidity"),
        "wind_kmh": round(best["wind"].get("speed", 0) * 3.6, 1),
        "conditions": best["weather"][0]["description"] if best.get("weather") else "?",
        "rain_3h_mm": best.get("rain", {}).get("3h", 0),
        "snow_3h_mm": best.get("snow", {}).get("3h", 0),
        "forecast_for_utc": datetime.fromtimestamp(best["dt"], tz=timezone.utc).isoformat(),
        "delta_h_from_kickoff": round((best["dt"] - kickoff_utc.timestamp()) / 3600, 1),
    }


def get_weather_for_match(venue_str: str | None, kickoff_utc: datetime) -> dict | None:
    """Wrapper: venue del fixture → coords → forecast → texto legible para LLM."""
    venue = resolve_venue(venue_str)
    if not venue:
        log.info("Venue no resuelto: %r", venue_str)
        return None

    if venue.get("indoor"):
        return {
            "summary": f"Indoor ({venue.get('name')}) — clima irrelevante",
            "indoor": True,
            "venue_name": venue.get("name"),
        }

    forecast = get_forecast_at_kickoff(venue["lat"], venue["lon"], kickoff_utc)
    if not forecast:
        return None

    # Descripción legible
    parts = [
        f"{forecast['temp_c']:.0f}°C",
        forecast['conditions'],
    ]
    if forecast['rain_3h_mm'] > 0:
        parts.append(f"lluvia {forecast['rain_3h_mm']:.1f}mm")
    if forecast['wind_kmh'] > 25:
        parts.append(f"viento fuerte {forecast['wind_kmh']:.0f} km/h")
    if venue.get("elevation_m", 0) > 1500:
        parts.append(f"altura {venue['elevation_m']}m")

    return {
        "summary": " · ".join(parts),
        "venue_name": venue.get("name"),
        "indoor": False,
        **forecast,
    }
