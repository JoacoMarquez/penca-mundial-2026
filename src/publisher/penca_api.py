"""Cliente de la API de la penca JMLM 2026.

⚠️ SPECS TBD: el admin de la penca está implementando la API. Cuando esté lista, completar:
    - URL del endpoint (default asumido: POST /api/predictions)
    - Método de auth (asumido: bearer token en header Authorization)
    - Schema del body (asumido: {match_id, penca_id, score_local, score_visit})

Mientras tanto se usa NullPublisher (solo loguea) para que el resto del pipeline funcione.

DRY_RUN=true en .env desactiva la publicación real incluso si la API está disponible.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

import httpx

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PredictionPayload:
    match_id: str          # de fixtures.yaml
    penca_id: str          # uno de los 5 IDs de tus pencas
    score_local: int
    score_visit: int


@dataclass(frozen=True)
class PublishResult:
    ok: bool
    status_code: int | None
    detail: str | None


class Publisher(ABC):
    """Interfaz abstracta. Toda implementación tiene que ser idempotente: re-publicar la misma
    predicción no debe duplicar entradas (la API debe hacer upsert sobre (match_id, penca_id))."""

    @abstractmethod
    def publish(self, payload: PredictionPayload) -> PublishResult: ...

    def publish_batch(self, payloads: list[PredictionPayload]) -> list[PublishResult]:
        return [self.publish(p) for p in payloads]


class NullPublisher(Publisher):
    """No publica nada. Solo loguea. Útil para DRY_RUN y mientras esperamos la API real."""

    def publish(self, payload: PredictionPayload) -> PublishResult:
        log.info(
            "DRY_RUN publish | match=%s penca=%s score=%d-%d",
            payload.match_id, payload.penca_id, payload.score_local, payload.score_visit,
        )
        return PublishResult(ok=True, status_code=None, detail="dry_run")


class HttpPublisher(Publisher):
    """Cliente HTTP contra la API real de la penca.

    Asumimos: POST {base_url}/predictions con bearer token. Cuando lleguen las specs reales,
    ajustar `endpoint`, schema del body, y método de auth.
    """

    def __init__(self, base_url: str, api_key: str, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._client = httpx.Client(
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )

    def publish(self, payload: PredictionPayload) -> PublishResult:
        url = f"{self.base_url}/predictions"
        body = {
            "match_id": payload.match_id,
            "penca_id": payload.penca_id,
            "score_local": payload.score_local,
            "score_visit": payload.score_visit,
        }
        try:
            resp = self._client.post(url, json=body)
            ok = 200 <= resp.status_code < 300
            return PublishResult(ok=ok, status_code=resp.status_code, detail=resp.text[:500])
        except httpx.HTTPError as e:
            return PublishResult(ok=False, status_code=None, detail=f"HTTPError: {e}")


def get_publisher_from_env() -> Publisher:
    """Factory que elige Publisher según variables de entorno.

    - DRY_RUN=true → NullPublisher (default)
    - PENCA_API_BASE_URL + PENCA_API_KEY presentes y DRY_RUN=false → HttpPublisher
    """
    dry_run = os.environ.get("DRY_RUN", "true").lower() == "true"
    base_url = os.environ.get("PENCA_API_BASE_URL", "")
    api_key = os.environ.get("PENCA_API_KEY", "")

    if dry_run or not base_url or not api_key:
        log.info("Publisher: NullPublisher (dry_run=%s, has_api=%s)", dry_run, bool(base_url and api_key))
        return NullPublisher()
    return HttpPublisher(base_url=base_url, api_key=api_key)
