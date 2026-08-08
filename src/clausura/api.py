"""Cliente de LECTURA del penca-api de Supermatch (endpoints públicos, sin auth).

Descubierto 2026-08-04 leyendo el bundle Angular de www.supermatch.com.uy:
    base: https://penca.supermatch.com.uy/penca-api/v1
    GET front/pencas/visibles                      → pencas activas con premios
    GET front/campeonatos/{campId}/fechas          → fechas del campeonato
    GET front/campeonatos/fechas/{fechaId}/eventos → partidos de una fecha
    GET front/pencas/{pencaId}/ranking             → ranking paginado (Spring Page)
    GET front/pencas/{pencaId}/reglamento          → reglamento

Los endpoints responden a httpx directo sin Cloudflare (mismo patrón que el
Elasticsearch de odds en src/valuebet/books/supermatch.py). El ranking expone
`cantResultadosExactos` por participación — canal directo para calibrar el pool.

La escritura (private/…/pronosticoEvento etc.) requiere Bearer de Keycloak y vive
en el publisher, no acá.

IDs vigentes (2026-08-04): penca paga id=46, gratuita id=47, campeonato Clausura
id=44, fechas 280..294. No hardcodear en el código consumidor: usar
config/clausura2026.yaml generado por src.clausura.sync.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

import httpx

log = logging.getLogger(__name__)

BASE = "https://penca.supermatch.com.uy/penca-api/v1"

# Backoff corto: destraba el throttle pasajero sin colgar 15 min a un timer. El
# bloqueo duro se deja salir como error (pool_snapshot lo maneja con más paciencia).
RATE_LIMIT_BACKOFF_S = 60
RATE_LIMIT_REINTENTOS = 3

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"),
    "Accept": "application/json",
    "Origin": "https://www.supermatch.com.uy",
    "Referer": "https://www.supermatch.com.uy/",
    "Accept-Language": "es-UY,es;q=0.9",
}

# La API devuelve fechas "07-08-2026 19:00:00" en hora uruguaya (UTC-3, sin DST).
TZ_UY = timezone(timedelta(hours=-3))


@dataclass(frozen=True)
class Premio:
    nombre: str
    tipo: str       # "PENCA" | "FECHA" | "GRUPOAMIGO"
    monto: float
    descripcion: str


@dataclass(frozen=True)
class Penca:
    id: int
    nombre: str
    precio: float
    campeonato_id: int
    estado: str
    premios: tuple[Premio, ...]


@dataclass(frozen=True)
class Evento:
    id: int
    fecha_id: int
    local: str
    visitante: str
    local_id: int
    visitante_id: int
    inicio_utc: datetime
    cierre_pronostico_utc: datetime
    estadio: str
    preferencial: bool      # partido estrella: puntos x2
    estado: str             # "PENDIENTE" | ...


@dataclass(frozen=True)
class RankingRow:
    participacion_id: int
    numero_participacion: int
    puntos_totales: int
    puntos_por_fecha: int
    posicion_general: int
    cant_resultados_exactos: int


def _parse_dt(s: str) -> datetime:
    return datetime.strptime(s, "%d-%m-%Y %H:%M:%S").replace(tzinfo=TZ_UY).astimezone(timezone.utc)


# -------------------- parsers puros (testeables con fixtures) --------------------

def parse_pencas(data: list[dict]) -> list[Penca]:
    return [
        Penca(
            id=p["id"],
            nombre=p["nombre"],
            precio=float(p.get("precio", 0)),
            campeonato_id=p["campeonatoId"],
            estado=p.get("estado", ""),
            premios=tuple(
                Premio(
                    nombre=pr.get("nombre", ""),
                    tipo=pr.get("tipo", ""),
                    monto=float(pr.get("monto", 0)),
                    descripcion=pr.get("descripcion", ""),
                )
                for pr in p.get("premios", [])
            ),
        )
        for p in data
    ]


def parse_eventos(data: list[dict], fecha_id: int) -> list[Evento]:
    out = []
    for e in data:
        out.append(Evento(
            id=e["id"],
            fecha_id=fecha_id,
            local=e["equipoLocal"]["nombre"],
            visitante=e["equipoVisitante"]["nombre"],
            local_id=e["equipoLocal"]["id"],
            visitante_id=e["equipoVisitante"]["id"],
            inicio_utc=_parse_dt(e["fechaInicio"]),
            cierre_pronostico_utc=_parse_dt(e["fechaCierrePronostico"]),
            estadio=e.get("locacion", ""),
            preferencial=bool(e.get("preferencial", False)),
            estado=e.get("estado", ""),
        ))
    return sorted(out, key=lambda ev: ev.inicio_utc)


def parse_ranking_page(page: dict) -> tuple[list[RankingRow], int, int]:
    """Página Spring → (rows, total_elements, total_pages)."""
    rows = [
        RankingRow(
            participacion_id=r["participacionId"],
            numero_participacion=r["numeroParticipacion"],
            puntos_totales=r.get("puntosTotales", 0),
            puntos_por_fecha=r.get("puntosPorFecha", 0),
            posicion_general=r.get("posicionGeneral", -1),
            cant_resultados_exactos=r.get("cantResultadosExactos", 0),
        )
        for r in page.get("content", [])
    ]
    return rows, page.get("totalElements", len(rows)), page.get("totalPages", 1)


# -------------------- cliente HTTP --------------------

class PencaApiClient:
    def __init__(self, timeout: float = 25.0):
        self._client = httpx.Client(base_url=BASE, timeout=timeout, headers=HEADERS)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "PencaApiClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _get_429(self, path: str, params: dict | None = None) -> httpx.Response:
        """GET que aguanta un 429 transitorio en vez de tumbar al que llama.

        El API limita por IP (429 a las ~200 requests seguidas, medido el 7/8) y
        todo lo que consume este cliente —drift audit, vigía, postmortem— moría con
        HTTPStatusError crudo: la auditoría de la Fecha 1 se perdió así, con falso
        OnFailure incluido. Backoff corto a propósito: destraba el throttle pasajero
        y deja que el bloqueo DURO salga como error (el escaneo del pool tiene su
        propia política, más paciente, encima de esta)."""
        for intento in range(RATE_LIMIT_REINTENTOS + 1):
            r = self._client.get(path, params=params)
            if r.status_code != 429 or intento == RATE_LIMIT_REINTENTOS:
                return r
            log.warning("429 en %s — backoff %ds (intento %d/%d)",
                        path, RATE_LIMIT_BACKOFF_S, intento + 1, RATE_LIMIT_REINTENTOS)
            time.sleep(RATE_LIMIT_BACKOFF_S)
        return r

    def _get(self, path: str, **params) -> dict | list:
        r = self._get_429(path, params or None)
        r.raise_for_status()
        body = r.json()
        return body.get("data", body) if isinstance(body, dict) else body

    def pencas_visibles(self) -> list[Penca]:
        return parse_pencas(self._get("/front/pencas/visibles"))

    def fechas(self, campeonato_id: int) -> dict[str, int]:
        """{'Fecha 1': 280, ...} ordenado por número de fecha."""
        data = self._get(f"/front/campeonatos/{campeonato_id}/fechas")
        pairs = {d["nombre"]: d["id"] for d in data}
        return dict(sorted(pairs.items(), key=lambda kv: int(kv[0].split()[-1])))

    def eventos(self, fecha_id: int) -> list[Evento]:
        return parse_eventos(self._get(f"/front/campeonatos/fechas/{fecha_id}/eventos"), fecha_id)

    def ranking(self, penca_id: int) -> list[RankingRow]:
        """Ranking completo, siguiendo la paginación.

        OJO: el parámetro `page` de este endpoint es 1-BASED (verificado 2026-08-04:
        page=0 devuelve 500, page=1 es la primera página, y el `number` del body sí
        viene 0-based). Spring clásico sería 0-based — no "arreglar" sin re-probar.
        """
        rows: list[RankingRow] = []
        page_n = 1
        while True:
            r = self._get_429(f"/front/pencas/{penca_id}/ranking", {"page": page_n})
            r.raise_for_status()
            page_rows, _total, total_pages = parse_ranking_page(r.json())
            rows.extend(page_rows)
            page_n += 1
            if page_n > total_pages or not page_rows:
                return rows

    def reglamento(self, penca_id: int) -> dict | list:
        return self._get(f"/front/pencas/{penca_id}/reglamento")
