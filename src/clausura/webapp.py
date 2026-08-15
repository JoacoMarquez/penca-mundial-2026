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

import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from src.clausura.dashboard_loader import load_clausura_page, load_pool_page

log = logging.getLogger(__name__)

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


@app.get("/dash/{token}/carga/", response_class=HTMLResponse)
def page_carga(request: Request, token: str, fecha: Optional[int] = None):
    """Modo carga: una tarjeta por participación en el orden de la web, con
    checkbox de progreso COMPARTIDO entre dispositivos (ver carga_state): la marca
    vive en localStorage para que la pantalla responda sin red, y se sincroniza
    contra el servidor para que cargar mitad en la PC y mitad en el celular sea una
    sola lista de trabajo."""
    _check_token(token)
    data = load_clausura_page(fecha_q=fecha)
    return templates.TemplateResponse(request, "carga.html", {"data": data, "token": token})


@app.get("/dash/{token}/pool/", response_class=HTMLResponse)
def page_pool(request: Request, token: str):
    """Estado competitivo: el pool entero, el premio y dónde caen nuestras 12."""
    _check_token(token)
    return templates.TemplateResponse(
        request, "pool.html", {"data": load_pool_page(), "token": token})


@app.get("/dash/{token}/api/data")
def api_data(token: str, fecha: Optional[int] = None):
    """JSON crudo de la página (debug / consumo externo)."""
    _check_token(token)
    return JSONResponse(load_clausura_page(fecha_q=fecha))


@app.get("/dash/{token}/api/pool")
def api_pool(token: str):
    _check_token(token)
    return JSONResponse(load_pool_page())


@app.get("/dash/{token}/api/verificar")
def api_verificar(token: str, fecha: Optional[int] = None):
    """Lo cargado en la web vs la planilla, pick por pick.

    Lo dispara el botón del modo carga. Son ~24 requests al penca-api con pacing,
    así que va cacheado: un doble-clic no lanza dos escaneos.
    """
    _check_token(token)
    from src.clausura.dashboard_loader import _cached
    from src.clausura.verificar_carga import VERIF_TTL, verificar

    return JSONResponse(_cached(f"verif:{fecha}", VERIF_TTL, lambda: verificar(fecha)))


@app.get("/dash/{token}/api/carga-marcas")
def api_carga_marcas(token: str):
    """Marcas del modo carga, compartidas entre dispositivos (ver carga_state)."""
    _check_token(token)
    from src.clausura.carga_state import leer

    return JSONResponse({"marcas": leer()})


@app.post("/dash/{token}/api/carga-marcas")
async def api_carga_marcas_set(token: str, request: Request):
    """Marca/desmarca una participación, o sube las marcas locales la primera vez.

    Dos formas de cuerpo:
      {"clave": "carga:v2:2:899258848:2094", "valor": "1-0"}   marca
      {"clave": "...", "valor": null}                          desmarca
      {"fusionar": {clave: valor, ...}}                        subida inicial
    """
    _check_token(token)
    from src.clausura.carga_state import aplicar, fusionar

    body = await request.json()
    try:
        if "fusionar" in body:
            marcas = fusionar(body.get("fusionar") or {})
        else:
            marcas = aplicar(body.get("clave", ""), body.get("valor"))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except OSError as e:                       # disco lleno / FS de solo lectura
        log.warning("no pude guardar las marcas de carga: %s", e)
        raise HTTPException(503, "no pude guardar")
    return JSONResponse({"marcas": marcas})
