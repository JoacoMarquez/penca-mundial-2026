"""Modo carga de la penca LUB: la última planilla en el formato del template del Clausura.

Reusa src/clausura/templates/carga.html (tarjetas por participación, marcas con el
marcador cargado, sincronización entre dispositivos) con otro prefijo de claves y otro
archivo de marcas, para que las dos pencas no se pisen. Sin botón de verificar: la
LUB no tiene todavía verificación post-cierre contra la web.

Muestra los partidos marcados "hoy" de la planilla (los que hay que cargar ese día);
una planilla corrida a mano sin ventana los muestra todos.
"""

from __future__ import annotations

from datetime import datetime, timezone

from src.lub.data import PENCA_ID, TZ_UY, mis_numeros_env
from src.lub.picks import TEMPORADA, ultima_planilla

MARCA_PREFIJO = "lub:v1:"


def _uy(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).astimezone(TZ_UY).strftime("%d/%m %H:%M")
    except ValueError:
        return iso


def load_lub_carga(token: str, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    base = {
        "torneo": f"🏀 Penca {TEMPORADA}", "fecha_n": 0, "fecha_label": "LUB",
        "marca_prefijo": MARCA_PREFIJO, "verificar": False, "ancla_ev": False,
        "api_marcas_url": f"/dash/{token}/lub/api/carga-marcas",
        "api_data_url": f"/dash/{token}/lub/api/data",
        "planilla_url": f"/dash/{token}/lub/api/data",
        "cmd_hint": "python -m src.lub.picks",
        "supermatch_url": f"https://www.supermatch.com.uy/pencas/1/{PENCA_ID}/penca",
    }
    pl = ultima_planilla()
    if not pl:
        return {**base, "ok": False, "planilla": None}
    filas = [f for f in pl["partidos"] if f.get("hoy", True)] or pl["partidos"]
    esp = None
    if pl.get("especiales_libres"):
        gol = pl.get("goleador") or [None] * pl["k"]
        esp = {"por_participacion": [{"campeon": c or "—", "goleador": g}
                                     for c, g in zip(pl["campeon"], gol)]}
    return {
        **base, "ok": True, "fecha_label": pl["fecha"],
        "mis_numeros": pl.get("numeros") or mis_numeros_env(),
        "planilla": {
            "version_file": pl["_archivo"], "generado_uy": _uy(pl["generado_utc"]),
            "n_participaciones": pl["k"], "descartadas_despues": 0, "especiales": esp,
            "picks": [{
                "evento_id": f["evento_id"], "preferencial": f["preferencial"],
                "partido": f"{f['local']} vs {f['visitante']} · {_uy(f['cierre_utc'])}",
                "scores_fmt": [f"{a}-{b}" for a, b in f["picks"]],
                "cerrado": datetime.fromisoformat(f["cierre_utc"]) <= now,
            } for f in filas],
        },
    }
