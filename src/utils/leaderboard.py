"""Fetch del leaderboard con cache a disco (stale-while-error).

El endpoint /leaderboard de la API cuelga a veces (timeout server-side, sin devolver
ni un 504). Para que ni el dashboard ni el pipeline de picks queden ciegos cuando eso
pasa, persistimos en disco el último leaderboard bueno y lo servimos como fallback
marcado `stale`. Segundo fallback (si en esta corrida nunca se cacheó): el snapshot de
pool más reciente que ya deja la calibración en data/pool_snapshots/.

Uso:
    res = fetch_leaderboard(base, key, timeout=8.0, max_stale_seconds=12*3600)
    res["entries"]  → list[dict] (vacía si no hay nada, ni fresco ni cacheado)
    res["stale"]    → True si vienen de disco (API falló), False si frescas
    res["error"]    → motivo del fallo de la API (o None si fue fresco)
    res["age_seconds"] → antigüedad de los datos stale (None si frescos o sin dato)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def _data_dir() -> Path:
    from src.utils.env import get_str
    return Path(get_str("DATA_DIR", "data"))


def _cache_path() -> Path:
    return _data_dir() / "leaderboard_cache.json"


def _write_cache(entries: list[dict]) -> None:
    """Persiste el último leaderboard bueno. Escritura atómica (tmp + replace)."""
    if not entries:
        return
    try:
        p = _cache_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"fetched_at": time.time(), "entries": entries}, ensure_ascii=False))
        tmp.replace(p)
    except Exception as e:  # nunca romper el fetch por un problema de disco
        log.warning("no se pudo escribir leaderboard_cache: %s", e)


def _read_cache() -> tuple[list[dict], float | None]:
    """Devuelve (entries, fetched_at) del último leaderboard bueno.

    Primero el cache dedicado; si no existe, cae al snapshot de pool más reciente
    (los deja la calibración post-partido) para no arrancar sin nada tras un deploy.
    """
    p = _cache_path()
    try:
        if p.exists():
            d = json.loads(p.read_text())
            entries = d.get("entries") or []
            if entries:
                return entries, d.get("fetched_at")
    except Exception:
        pass
    return _read_latest_snapshot()


def _read_latest_snapshot() -> tuple[list[dict], float | None]:
    sdir = _data_dir() / "pool_snapshots"
    if not sdir.exists():
        return [], None
    best_mtime: float | None = None
    best_file: Path | None = None
    for f in sdir.glob("*.json"):
        try:
            mt = f.stat().st_mtime
        except Exception:
            continue
        if best_mtime is None or mt > best_mtime:
            best_mtime, best_file = mt, f
    if best_file is None:
        return [], None
    try:
        d = json.loads(best_file.read_text())
        return (d.get("entries") or []), best_mtime
    except Exception:
        return [], None


def fetch_leaderboard(
    base: str,
    key: str,
    timeout: float = 8.0,
    max_stale_seconds: float | None = None,
) -> dict[str, Any]:
    """GET /leaderboard con fallback stale a disco.

    - éxito: entries frescas, stale=False; persiste a disco.
    - fallo con cache utilizable: entries del disco, stale=True.
    - fallo sin cache, o cache más vieja que max_stale_seconds: entries=[], stale=False.

    max_stale_seconds acota cuán vieja puede ser la cache para servirse (None = sin
    límite; el dashboard no limita, el pipeline sí para no picar sobre un corte ancestral).
    """
    base = (base or "").rstrip("/")
    err: str | None = None

    if base and key:
        try:
            import httpx
            with httpx.Client(timeout=timeout, headers={"Authorization": f"Bearer {key}"}) as c:
                r = c.get(f"{base}/leaderboard")
            if r.status_code == 200:
                entries = r.json().get("entries", []) or []
                if entries:
                    _write_cache(entries)
                    return {"entries": entries, "stale": False, "error": None,
                            "fetched_at": time.time(), "age_seconds": 0.0}
                err = "leaderboard vacío"
            else:
                err = f"leaderboard {r.status_code}"
        except Exception as e:
            err = str(e)
    else:
        err = "API no configurada"

    # Fallo de la API → intentar disco / snapshot.
    entries, ts = _read_cache()
    if entries:
        age = (time.time() - ts) if ts else None
        if max_stale_seconds is not None and age is not None and age > max_stale_seconds:
            log.warning(
                "leaderboard stale demasiado viejo (%.1fh > cap %.1fh); no se usa",
                age / 3600, max_stale_seconds / 3600,
            )
            return {"entries": [], "stale": False, "error": err, "fetched_at": None, "age_seconds": None}
        return {"entries": entries, "stale": True, "error": err, "fetched_at": ts, "age_seconds": age}

    return {"entries": [], "stale": False, "error": err, "fetched_at": None, "age_seconds": None}
