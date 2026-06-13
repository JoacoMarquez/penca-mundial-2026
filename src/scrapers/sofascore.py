"""Scraper de SofaScore para lineups + estado del partido vía Playwright.

SofaScore tiene una API privada que bloquea curl/httpx (403), pero **acepta requests
desde browser real**. Usamos Playwright headless con flags realistas para emularlo.

Endpoints útiles (después de pasar la verificación inicial):
    /api/v1/search/events?q={home}+{away}
    /api/v1/event/{event_id}/lineups
    /api/v1/event/{event_id}        # info general

Estrategia:
    1. Abrir página de search en sofascore.com → resuelve cookies/JS challenges.
    2. Reusar el contexto del browser para los siguientes fetches a la API privada.
    3. Parsear JSON.

Fallback gracioso: si todo falla, retornar {} (igual que API-Football).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

log = logging.getLogger(__name__)


SOFA_API = "https://api.sofascore.com/api/v1"


def _new_browser_context():
    """Crea un browser headless Playwright con flags realistas."""
    from playwright.sync_api import sync_playwright
    p = sync_playwright().start()
    browser = p.chromium.launch(
        headless=True,
        args=[
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
            "--disable-dev-shm-usage",
        ],
    )
    context = browser.new_context(
        user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
        ),
        viewport={"width": 1280, "height": 720},
        locale="en-US",
    )
    # Anti-detection: ocultar webdriver flag
    context.add_init_script("""
        Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
        Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
    """)
    return p, browser, context


def _api_get(context, path: str) -> dict | None:
    """Hace un GET a la API privada de SofaScore desde dentro del context del browser."""
    page = context.new_page()
    try:
        # Ir primero a sofascore.com para establecer cookies
        page.goto("https://www.sofascore.com/", wait_until="domcontentloaded", timeout=15000)
        # Ahora pedir la API
        resp = page.goto(f"{SOFA_API}{path}", wait_until="domcontentloaded", timeout=15000)
        if resp is None or resp.status >= 400:
            log.warning("SofaScore %s → %s", path, resp.status if resp else "no response")
            return None
        text = page.evaluate("() => document.body.innerText")
        return json.loads(text)
    except Exception as e:
        log.warning("SofaScore %s falló: %s", path, e)
        return None
    finally:
        page.close()


def _match_event(search_data: dict | None, home: str, away: str) -> dict | None:
    """Elige el event correcto del resultado de búsqueda por matching de nombres. Puro/testeable."""
    if not search_data:
        return None
    home_low = home.lower()
    away_low = away.lower()
    for ev in search_data.get("events", []):
        t1 = ev.get("homeTeam", {}).get("name", "").lower()
        t2 = ev.get("awayTeam", {}).get("name", "").lower()
        if (home_low in t1 or t1 in home_low) and (away_low in t2 or t2 in away_low):
            return ev
    return None


def search_event(home: str, away: str, date: datetime | None = None) -> dict | None:
    """Busca el event_id de SofaScore para un partido."""
    p = browser = context = None
    try:
        p, browser, context = _new_browser_context()
        q = f"{home} {away}".replace(" ", "+")
        data = _api_get(context, f"/search/events/?q={q}")
        return _match_event(data, home, away)
    except Exception as e:
        log.warning("search_event falló: %s", e)
        return None
    finally:
        if browser: browser.close()
        if p: p.stop()


def get_event_lineups(event_id: int) -> dict | None:
    """Lineups de un event de SofaScore. Retorna {home: {...}, away: {...}} o None."""
    p = browser = context = None
    try:
        p, browser, context = _new_browser_context()
        data = _api_get(context, f"/event/{event_id}/lineups")
        return data
    except Exception as e:
        log.warning("get_event_lineups falló: %s", e)
        return None
    finally:
        if browser: browser.close()
        if p: p.stop()


def parse_lineups(data: dict | None) -> dict:
    """Parsea el payload de /event/{id}/lineups a un dict plano y testeable.

    El payload de SofaScore tiene la forma::

        {"confirmed": bool,
         "home": {"formation": "4-3-3", "players": [{"player": {"name": ...}, "substitute": bool}, ...]},
         "away": {...}}

    Returns dict con (cuando hay datos):
        - lineup_confirmed: bool  — SofaScore marca True solo cuando es el XI oficial
        - home_lineup_change / away_lineup_change: "Formación: 4-3-3"
        - home_xi / away_xi: lista de nombres del once titular (substitute == False)
        - lineup_source: "sofascore"
    """
    if not data:
        return {}
    out: dict[str, Any] = {"lineup_confirmed": bool(data.get("confirmed")), "lineup_source": "sofascore"}
    found_any = False
    for side in ("home", "away"):
        side_data = data.get(side) or {}
        formation = side_data.get("formation")
        if formation:
            out[f"{side}_lineup_change"] = f"Formación: {formation}"
            found_any = True
        xi = []
        for entry in side_data.get("players") or []:
            if entry.get("substitute"):
                continue
            name = (entry.get("player") or {}).get("name")
            if name:
                xi.append(name)
        if xi:
            out[f"{side}_xi"] = xi
            found_any = True
    return out if found_any else {}


def collect_match_context_sofascore(
    home: str,
    away: str,
    kickoff_utc: datetime,
) -> dict:
    """API de alto nivel — equivalente al collect_match_context de football_api.py.

    Usa un solo browser context para search + lineups (antes lanzaba 2 browsers).

    Returns dict con keys (cuando los datos están disponibles):
        - lineup_confirmed: bool
        - home_lineup_change / away_lineup_change: "Formación: 4-3-3"
        - home_xi / away_xi: nombres del once titular
    """
    p = browser = context = None
    try:
        p, browser, context = _new_browser_context()
        q = f"{home} {away}".replace(" ", "+")
        search = _api_get(context, f"/search/events/?q={q}")
        event = _match_event(search, home, away) if search else None
        if not event:
            return {}
        out: dict[str, Any] = {"sofascore_event_id": event["id"]}
        lineups = _api_get(context, f"/event/{event['id']}/lineups")
        out.update(parse_lineups(lineups))
        return out
    except Exception as e:
        log.warning("collect_match_context_sofascore falló: %s", e)
        return {}
    finally:
        if browser: browser.close()
        if p: p.stop()
