"""FastAPI dashboard — multi-page, mobile-first + desktop full-width.

Run local: uvicorn src.dashboard.app:app --host 0.0.0.0 --port 8000 --reload
Páginas:
    /dash/<token>/             → Próximo partido (home)
    /dash/<token>/pencas/      → Tus pencas vs pool
    /dash/<token>/history/     → Postmortems
    /dash/<token>/system/      → Health + costos
    /dash/<token>/api/data     → JSON con todo (debug)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from src.dashboard.data_loader import (
    STRATEGY_DISPLAY_EMOJI,
    STRATEGY_DISPLAY_LABEL,
    build_penca_labels,
    load_live_match_data,
    load_match_detail,
    load_matches_by_day,
    load_my_pencas_standings,
    load_next_match_data,
    load_penca_detail,
    load_pool_intelligence,
    load_postmortem_history,
    load_recent_postmortems,
    load_strategy_metrics,
    load_system_health,
)

HERE = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(HERE / "templates"))

# Etiquetas/emojis de estrategia disponibles en TODOS los templates (una sola fuente de
# verdad en data_loader, antes duplicadas con `{% set %}` en cada template).
templates.env.globals["strategy_label"] = STRATEGY_DISPLAY_LABEL
templates.env.globals["strategy_emoji"] = STRATEGY_DISPLAY_EMOJI

app = FastAPI(title="Penca Mundial Dashboard")


def _check_token(token: str) -> None:
    expected = os.environ.get("DASHBOARD_TOKEN", "")
    if not expected:
        raise HTTPException(503, "Dashboard token no configurado")
    if token != expected:
        raise HTTPException(404, "Not found")


@app.get("/")
def root():
    return {"status": "ok", "hint": "use /dash/<token>/"}


@app.get("/dash/{token}/", response_class=HTMLResponse)
def page_home(request: Request, token: str, match: Optional[str] = None):
    _check_token(token)
    # `match` (query param) elige cuál de los simultáneos mostrar; default = el primero.
    nxt = load_next_match_data(match_id=match)
    detail = load_match_detail(nxt["match_id"]) if nxt else None
    data = {
        "live": load_live_match_data(),
        "next": nxt,
        "match": detail,
        "health": load_system_health(),
    }
    return templates.TemplateResponse(request, "index.html", {"data": data, "token": token, "penca_labels": build_penca_labels()})


@app.get("/dash/{token}/matches/", response_class=HTMLResponse)
def page_matches(request: Request, token: str):
    _check_token(token)
    data = {
        "days": load_matches_by_day(days_back=2, days_ahead=21),
        "health": load_system_health(),
    }
    return templates.TemplateResponse(request, "matches.html", {"data": data, "token": token, "penca_labels": build_penca_labels()})


@app.get("/dash/{token}/match/{match_id}/", response_class=HTMLResponse)
def page_match_detail(request: Request, token: str, match_id: str):
    _check_token(token)
    detail = load_match_detail(match_id)
    if not detail:
        raise HTTPException(404, "Match not found")
    data = {
        "match": detail,
        "health": load_system_health(),
    }
    return templates.TemplateResponse(request, "match_detail.html", {"data": data, "token": token, "penca_labels": build_penca_labels()})


@app.get("/dash/{token}/pencas/", response_class=HTMLResponse)
def page_pencas(request: Request, token: str):
    _check_token(token)
    data = {
        "standings": load_my_pencas_standings(),
        "health": load_system_health(),
    }
    return templates.TemplateResponse(request, "pencas.html", {"data": data, "token": token, "penca_labels": build_penca_labels()})


@app.get("/dash/{token}/penca/{penca_id}/", response_class=HTMLResponse)
def page_penca_detail(request: Request, token: str, penca_id: int):
    _check_token(token)
    data = load_penca_detail(penca_id)
    return templates.TemplateResponse(request, "penca_detail.html", {"data": data, "token": token, "penca_labels": build_penca_labels()})


@app.get("/dash/{token}/pool/", response_class=HTMLResponse)
def page_pool(request: Request, token: str):
    _check_token(token)
    data = {
        "pool": load_pool_intelligence(),
        "health": load_system_health(),
    }
    return templates.TemplateResponse(request, "pool.html", {"data": data, "token": token, "penca_labels": build_penca_labels()})


@app.get("/dash/{token}/metricas/", response_class=HTMLResponse)
def page_metricas(request: Request, token: str):
    _check_token(token)
    data = {
        "metrics": load_strategy_metrics(),
        "health": load_system_health(),
    }
    return templates.TemplateResponse(request, "metricas.html", {"data": data, "token": token, "penca_labels": build_penca_labels()})


@app.get("/dash/{token}/history/", response_class=HTMLResponse)
def page_history(request: Request, token: str):
    _check_token(token)
    data = {
        "history": load_postmortem_history(),
        "health": load_system_health(),
    }
    return templates.TemplateResponse(request, "history.html", {"data": data, "token": token, "penca_labels": build_penca_labels()})


@app.get("/dash/{token}/system/", response_class=HTMLResponse)
def page_system(request: Request, token: str):
    _check_token(token)
    data = {"health": load_system_health()}
    return templates.TemplateResponse(request, "system.html", {"data": data, "token": token})


@app.get("/dash/{token}/clausura/", response_class=HTMLResponse)
def page_clausura(request: Request, token: str, fecha: Optional[int] = None):
    """Penca Supermatch del Clausura 2026 — planilla, ranking en vivo y especiales.

    Independiente del pipeline del Mundial: lee data/predictions/clausura/ y el
    penca-api público, así que funciona aunque el resto del data/ esté vacío.
    """
    _check_token(token)
    from src.clausura.dashboard_loader import load_clausura_page
    data = load_clausura_page(fecha_q=fecha)
    return templates.TemplateResponse(request, "clausura.html", {"data": data, "token": token})


@app.get("/dash/{token}/api/data")
def api_data(token: str):
    _check_token(token)
    return JSONResponse({
        "next_match": load_next_match_data(),
        "standings": load_my_pencas_standings(),
        "postmortems": load_recent_postmortems(limit=20),
        "health": load_system_health(),
    })
