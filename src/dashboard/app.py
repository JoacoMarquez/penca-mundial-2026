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

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from src.dashboard.data_loader import (
    build_penca_labels,
    load_match_detail,
    load_matches_by_day,
    load_my_pencas_standings,
    load_next_match_data,
    load_penca_detail,
    load_recent_postmortems,
    load_system_health,
)

HERE = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(HERE / "templates"))

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
def page_home(request: Request, token: str):
    _check_token(token)
    data = {
        "next_match": load_next_match_data(),
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
    return templates.TemplateResponse(request, "penca_detail.html", {"data": data, "token": token})


@app.get("/dash/{token}/history/", response_class=HTMLResponse)
def page_history(request: Request, token: str):
    _check_token(token)
    data = {
        "postmortems": load_recent_postmortems(limit=20),
        "health": load_system_health(),
    }
    return templates.TemplateResponse(request, "history.html", {"data": data, "token": token, "penca_labels": build_penca_labels()})


@app.get("/dash/{token}/system/", response_class=HTMLResponse)
def page_system(request: Request, token: str):
    _check_token(token)
    data = {"health": load_system_health()}
    return templates.TemplateResponse(request, "system.html", {"data": data, "token": token})


@app.get("/dash/{token}/api/data")
def api_data(token: str):
    _check_token(token)
    return JSONResponse({
        "next_match": load_next_match_data(),
        "standings": load_my_pencas_standings(),
        "postmortems": load_recent_postmortems(limit=20),
        "health": load_system_health(),
    })
