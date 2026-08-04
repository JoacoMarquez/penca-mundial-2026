"""Dashboard standalone de la Penca Supermatch — Clausura 2026.

App aparte del dashboard del Mundial (src/dashboard/), a propósito: aquel queda
como archivo histórico apuntando a la data del VPS; este muestra SOLO el Clausura
y funciona con lo que hay en el checkout (config, data/predictions/clausura/ y el
penca-api público).

Run local:
    DASHBOARD_TOKEN=<token> uvicorn src.clausura.webapp:app --port 8000
    → http://127.0.0.1:8000/dash/<token>/          (+ ?fecha=N para otra fecha)

Reusa el mismo DASHBOARD_TOKEN del .env que el dashboard viejo.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from src.clausura.dashboard_loader import load_clausura_page

HERE = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(HERE / "templates"))

app = FastAPI(title="Penca Clausura 2026 Dashboard")


def _check_token(token: str) -> None:
    expected = os.environ.get("DASHBOARD_TOKEN", "")
    if not expected:
        raise HTTPException(503, "DASHBOARD_TOKEN no configurado")
    if token != expected:
        raise HTTPException(404, "Not found")


@app.get("/")
def root():
    return {"status": "ok", "hint": "use /dash/<token>/"}


@app.get("/dash/{token}/", response_class=HTMLResponse)
def page_clausura(request: Request, token: str, fecha: Optional[int] = None):
    _check_token(token)
    data = load_clausura_page(fecha_q=fecha)
    return templates.TemplateResponse(request, "clausura.html", {"data": data, "token": token})


@app.get("/dash/{token}/api/data")
def api_data(token: str, fecha: Optional[int] = None):
    """JSON crudo de la página (debug / consumo externo)."""
    _check_token(token)
    return JSONResponse(load_clausura_page(fecha_q=fecha))
