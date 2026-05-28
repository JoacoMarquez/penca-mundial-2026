"""FastAPI dashboard — mobile-first, sin auth (URL secreta).

Run local: uvicorn src.dashboard.app:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from src.dashboard.data_loader import (
    load_my_pencas_standings,
    load_next_match_data,
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
    """Root no expone nada — sirve solo como sanity check."""
    return {"status": "ok", "hint": "use /dash/<token>/"}


@app.get("/dash/{token}/", response_class=HTMLResponse)
def dashboard_html(request: Request, token: str):
    _check_token(token)
    data = _gather()
    return templates.TemplateResponse(request, "index.html", {"data": data, "token": token})


@app.get("/dash/{token}/api/data")
def dashboard_data(token: str):
    _check_token(token)
    return JSONResponse(_gather())


def _gather() -> dict:
    return {
        "next_match": load_next_match_data(),
        "standings": load_my_pencas_standings(),
        "postmortems": load_recent_postmortems(limit=5),
        "health": load_system_health(),
    }
