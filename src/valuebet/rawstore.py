"""Snapshots crudos de cuotas por scan: data/valuebet/raw/{book}/{fecha}/{hora}.json.

Sin esto no queda rastro de lo que se capturó: no se puede depurar un match
sospechoso post-hoc, ni reconstruir el historial de una línea, ni backtestear el
matcher contra datos reales. Best-effort: un fallo de disco no voltea el scan.

Retención: prune() borra los directorios de fecha con más de RETENTION_DAYS días
(se llama en cada save — barato, son unos pocos dirs por book).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.valuebet.types import OddsQuote

log = logging.getLogger(__name__)

RETENTION_DAYS = 30


def _raw_dir() -> Path:
    return Path(os.environ.get("VALUEBET_DATA_DIR", "data/valuebet")) / "raw"


def save_snapshot(book: str, quotes: list[OddsQuote], base: Path | None = None) -> Path | None:
    """Guarda las cuotas de una fuente tal como se capturaron. None si falla (con warning)."""
    now = datetime.now(timezone.utc)
    d = (base or _raw_dir()) / book / now.strftime("%Y-%m-%d")
    path = d / f"{now.strftime('%H%M%S')}.json"
    try:
        d.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump([asdict(q) for q in quotes], f, ensure_ascii=False)
        prune(base)
        return path
    except Exception as e:
        log.warning("rawstore: no pude guardar snapshot de %s: %s", book, e)
        return None


def prune(base: Path | None = None, days: int = RETENTION_DAYS) -> None:
    """Borra directorios de fecha más viejos que `days`. Best-effort."""
    root = base or _raw_dir()
    if not root.exists():
        return
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    for book_dir in root.iterdir():
        if not book_dir.is_dir():
            continue
        for date_dir in book_dir.iterdir():
            if date_dir.is_dir() and date_dir.name < cutoff:
                try:
                    shutil.rmtree(date_dir)
                except Exception as e:
                    log.warning("rawstore: no pude borrar %s: %s", date_dir, e)
