"""Ordenamiento correcto de archivos de predicción `v<N>_<timestamp>.json`.

`sorted(glob("v*_*.json"))` ordena LEXICOGRÁFICAMENTE → "v10" < "v9", así que con ≥10
versiones `[-1]` devuelve la versión equivocada (ej. v9 en vez de v20). Estas helpers
ordenan por el ENTERO de versión.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

_VERSION_RE = re.compile(r"^v(\d+)_")


def version_num(path: Path) -> int:
    """Extrae N de `vN_<ts>.json`. -1 si no matchea (queda al final del orden)."""
    m = _VERSION_RE.match(path.name)
    return int(m.group(1)) if m else -1


def sort_versions(paths: Iterable[Path]) -> list[Path]:
    """Lista ordenada por número de versión ascendente."""
    return sorted(paths, key=version_num)


def latest_version(paths: Iterable[Path]) -> Path | None:
    """Archivo con el número de versión más alto, o None si no hay."""
    items = list(paths)
    return max(items, key=version_num) if items else None
