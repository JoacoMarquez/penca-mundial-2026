"""Helpers para parsear variables de entorno robustamente.

Especialmente importante porque systemd's EnvironmentFile NO strip-ea comentarios inline.
Si el .env tiene `PENCA_IDS=1651,1652  # comentario`, el valor literal queda con el comment.
"""

from __future__ import annotations

import os
import re


def _strip_inline_comment(s: str) -> str:
    """Remueve comentarios inline tipo `# foo`. Solo si hay un # precedido por espacio o al inicio."""
    if not s:
        return s
    # Buscar '#' precedido por whitespace o al inicio
    m = re.search(r"(?:^|\s)#", s)
    if m:
        s = s[: m.start()]
    return s.strip()


def get_int_list(key: str, default: list[int] | None = None) -> list[int]:
    """Parsea variable CSV de enteros. Tolera comentarios y espacios."""
    raw = _strip_inline_comment(os.environ.get(key, ""))
    if not raw:
        return list(default or [])
    out = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        # extraer solo dígitos (por si quedó algún sufijo raro)
        m = re.match(r"-?\d+", part)
        if m:
            out.append(int(m.group(0)))
    return out


def get_str(key: str, default: str = "") -> str:
    """Get string sin comentario inline."""
    return _strip_inline_comment(os.environ.get(key, default))
