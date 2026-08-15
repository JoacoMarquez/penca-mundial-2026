"""Marcas del modo carga, compartidas entre dispositivos.

POR QUÉ EXISTE. Las marcas ("esta participación ya la cargué en la web") vivían solo
en `localStorage`, que es por navegador Y por dispositivo. Cargar 12 planillas desde
la PC y después seguir desde el celular mostraba la lista entera pendiente otra vez:
las dos mitades del trabajo no se ven. Y como el gate del penca-api no deja verificar
lo cargado hasta que el partido cierra, la marca manual es la ÚNICA memoria de por
dónde vas mientras cargás — perderla es perder el hilo justo en la parte que no se
puede reconstruir.

QUÉ SE GUARDA. Lo mismo que guardaba el navegador: clave → EL VALOR cargado (el
marcador), no un sí/no. Eso es lo que permite que, si el pipeline mueve un pick
después de que lo cargaste, la pantalla te diga "cargaste 1-0 y ahora dice 2-1" en
vez de mostrarlo como hecho. Ver el comentario largo en templates/carga.html.

QUÉ NO SE GUARDA. Las preferencias de pantalla (el filtro, la marca de migración)
son del dispositivo y se quedan en localStorage: sincronizarlas haría que esconder
lo cargado en el celular te lo esconda en la PC.

MODELO DE CONCURRENCIA. Un solo usuario con dos dispositivos: última escritura gana.
No hay merge ni revisiones porque no hay conflicto real — nadie desmarca desde un
lado mientras marca desde el otro. El servidor es la verdad y el navegador refresca.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
MARCAS_PATH = ROOT / "data" / "state" / "carga_marcas.json"

# Las claves las arma el navegador (kRow/kEsp en carga.html). Se validan acá igual:
# este endpoint ESCRIBE en disco y está expuesto en internet detrás de un token en la
# URL, así que no se acepta cualquier string como nombre de archivo lógico.
CLAVE_RE = re.compile(r"^carga:v2:(esp:)?[0-9]{1,12}(:[0-9]{1,12}){0,2}$")

# Un marcador ("2-1") o el id de un especial. Corto a propósito.
MAX_VALOR = 32

# 12 participaciones × 15 fechas × 8 partidos ≈ 1.440, más especiales. El tope existe
# para que un bug del cliente no llene el disco del droplet, no como límite real.
MAX_MARCAS = 5_000


def clave_valida(clave: str) -> bool:
    return bool(CLAVE_RE.match(clave))


def leer(path: Path | None = None) -> dict[str, str]:
    p = path or MARCAS_PATH
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        log.warning("marcas de carga ilegibles (%s) — arranco vacío", e)
        return {}
    return {k: str(v) for k, v in d.items()
            if isinstance(k, str) and clave_valida(k)}


def _escribir(marcas: dict[str, str], path: Path) -> None:
    """Escritura atómica: el celular y la PC pueden pegarle casi a la vez, y un
    archivo truncado a la mitad borraría el progreso de la carga."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(marcas, f, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise


def aplicar(clave: str, valor: str | None, path: Path | None = None) -> dict[str, str]:
    """Marca (valor) o desmarca (None) una participación. Devuelve el estado nuevo.

    Lanza ValueError si la clave o el valor no pasan validación — el endpoint lo
    traduce a 400, así que un cliente roto falla ruidoso en vez de escribir basura.
    """
    if not clave_valida(clave):
        raise ValueError(f"clave inválida: {clave[:64]!r}")
    if valor is not None and (not isinstance(valor, str) or len(valor) > MAX_VALOR):
        raise ValueError("valor inválido")

    p = path or MARCAS_PATH
    marcas = leer(p)
    if valor is None:
        marcas.pop(clave, None)
    else:
        if clave not in marcas and len(marcas) >= MAX_MARCAS:
            raise ValueError("demasiadas marcas guardadas")
        marcas[clave] = valor
    _escribir(marcas, p)
    return marcas


def fusionar(locales: dict[str, str], path: Path | None = None) -> dict[str, str]:
    """Sube las marcas que el navegador tenía y el servidor todavía no conoce.

    Es la migración del esquema viejo: el primer dispositivo que abre el modo carga
    después de este cambio trae su historial en localStorage y hay que conservarlo.
    El servidor NO se pisa — si una clave ya existe arriba, gana la del servidor,
    porque puede venir de otro dispositivo que cargó después.
    """
    p = path or MARCAS_PATH
    marcas = leer(p)
    nuevas = {k: str(v) for k, v in (locales or {}).items()
              if clave_valida(k) and k not in marcas
              and isinstance(v, str) and len(v) <= MAX_VALOR}
    if nuevas and len(marcas) + len(nuevas) <= MAX_MARCAS:
        marcas.update(nuevas)
        _escribir(marcas, p)
        log.info("marcas de carga: %d subidas desde el navegador", len(nuevas))
    return marcas
