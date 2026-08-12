"""Snapshot del pool: los picks de TODAS las participaciones, leídos del penca-api.

Desde el inicio del campeonato (gate del API: "Solo puede ver pronósticos si el
campeonato ya inicio"), los pronósticos de cada participación son públicos:

    GET front/pencas/{participacionId}/pronosticosEventos      → marcadores
    GET front/pencas/{participacionId}/pronosticoCampeonGoleador → especiales

(OJO: el path param es participacionId — sale del ranking — NO el id de la penca.)

Esto convierte la Capa 5 de modelada a EMPÍRICA: la distribución Q de picks por
partido se observa, no se infiere. El prior con bias de popularidad queda solo como
arranque en frío (Fecha 1) y como relleno donde falten observaciones.

Son 2 requests por participación (~1.370 con 683 inscriptas) y el API tira 429 a
las ~200 seguidas, así que el escaneo va paceado, reintenta ante 429, y desde el
lock de especiales reusa los ya observados (mitad de requests). Los timers además
pasan --max-edad-h para no re-escanear lo que ya está fresco: la cadencia útil es
UNA pasada por día de partidos, no una por partido (los picks de un partido recién
se publican cuando cierra, y para entonces el nuestro también está sellado).

Uso:
    python -m src.clausura.pool_snapshot                 # toma snapshot (error si no inició)
    python -m src.clausura.pool_snapshot --if-started    # sale 0 en silencio pre-inicio
                                                          (para el ExecStartPre del timer)
    python -m src.clausura.pool_snapshot --reusar-especiales --max-edad-h 6

El snapshot se versiona en data/pool_snapshots/clausura/ (nunca sobreescribe).
Riesgo simétrico documentado en memoria: nuestros picks también son públicos.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np

from src.clausura.api import BASE, HEADERS, PencaApiClient
from src.clausura.economics import MAX_GOALS, N_SCORES, score_index

log = logging.getLogger(__name__)

SNAP_DIR = Path(__file__).resolve().parents[2] / "data" / "pool_snapshots" / "clausura"

# El API tira 429 a las ~200 requests seguidas a ~8 req/s (medido la noche del
# kickoff, 7/8) y el bloqueo dura decenas de minutos. A ~3 req/s el escaneo
# completo (683 participaciones) tarda ~8 min y no lo despierta.
REQUEST_PAUSE_S = 0.35
RATE_LIMIT_BACKOFF_S = 420
# 2 y no más: un bloqueo que sobrevive dos backoffs (14 min) es duro y se despeja
# en horas, no en minutos — seguir reintentando solo alimenta el contador y demora
# al pipeline que espera este escaneo (medido el 7/8: los 429 transitorios se
# despejaron con UN backoff; el duro aguantó los cuatro).
RATE_LIMIT_REINTENTOS = 2
GATE_MSG = "campeonato ya inicio"


@dataclass
class RivalPicks:
    participacion_id: int
    numero: int
    puntos: int
    exactos: int
    # evento_id → (goles_local, goles_visitante)
    picks: dict[int, tuple[int, int]] = field(default_factory=dict)
    campeon: str | None = None
    campeon_id: int | None = None
    goleador: str | None = None
    goleador_id: int | None = None


class CampeonatoNoIniciado(RuntimeError):
    pass


class RateLimited(RuntimeError):
    """El API nos tiene bloqueados y el backoff no alcanzó.

    Aborta el escaneo ENTERO a propósito. Seguir de largo produciría filas vacías
    que se guardarían como snapshot válido — el pool fantasma que ya nos mordió el
    7/8. Y reintentar rival por rival contra un bloqueo duro son 14 min de espera
    POR PARTICIPACIÓN: el escaneo no termina nunca (medido: trabado en el mismo
    rival 28 min). Mejor abortar sin guardar y reintentar dentro de unas horas."""


# -------------------- fetch --------------------

def _get_pacing(
    c: httpx.Client,
    url: str,
    pause_s: float,
    backoff_s: float = RATE_LIMIT_BACKOFF_S,
    reintentos: int = RATE_LIMIT_REINTENTOS,
) -> httpx.Response:
    """GET con pausa fija y backoff ante 429.

    Reintentar es obligatorio, no opcional: un 429 tratado como fallo deja la fila
    VACÍA y el snapshot truncado se guarda igual, envenenando la Capa 5 con un pool
    fantasma (pasó el 7/8: 683 filas, 97 con datos). Ante 429 se espera y se
    reintenta la MISMA request; si ni así, se aborta el escaneo (RateLimited)."""
    for intento in range(reintentos + 1):
        r = c.get(url)
        time.sleep(pause_s)
        if r.status_code != 429:
            return r
        if intento == reintentos:
            raise RateLimited(f"429 persistente en {url} tras {reintentos} reintentos")
        log.warning("429 en %s — backoff %.0fs (intento %d/%d)",
                    url, backoff_s, intento + 1, reintentos)
        time.sleep(backoff_s)
    raise RateLimited(url)   # inalcanzable; mantiene el tipo de retorno honesto


def _ranking_pacing(
    penca_id: int,
    backoff_s: float = RATE_LIMIT_BACKOFF_S,
    reintentos: int = RATE_LIMIT_REINTENTOS,
):
    """Ranking con backoff: si el 429 pega acá, el escaneo entero muere en la
    request #1 y todo el pacing de abajo no sirve de nada."""
    for intento in range(reintentos + 1):
        try:
            with PencaApiClient() as api:
                return api.ranking(penca_id)
        except httpx.HTTPStatusError as e:
            if e.response.status_code != 429:
                raise
            if intento == reintentos:
                raise RateLimited(f"429 persistente en el ranking tras {reintentos} "
                                  "reintentos") from e
            log.warning("ranking → 429; backoff %.0fs (intento %d/%d)",
                        backoff_s, intento + 1, reintentos)
            time.sleep(backoff_s)


def lock_especiales_utc() -> datetime | None:
    """Instante en que los especiales dejaron de ser editables.

    Art. 4: 15 min antes del primer partido del campeonato — que es EXACTAMENTE el
    cierre de pronóstico del primer evento del fixture (2026-08-07T21:45Z)."""
    try:
        import yaml
        from src.clausura.picks import CONFIG_PATH, flat_eventos
        cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        cierres = [datetime.fromisoformat(e["cierre_pronostico_utc"])
                   for e in flat_eventos(cfg)]
        return min(cierres) if cierres else None
    except Exception as e:                                    # config rota o ausente
        log.warning("no se pudo determinar el lock de especiales (%s)", e)
        return None


def especiales_conocidos(
    snapshot: dict | None = None,
    lock_utc: datetime | None = None,
) -> dict[int, dict]:
    """{participacion_id: campeón/goleador} ya observados, para no re-pedirlos.

    Post-lock los especiales quedan FIJOS hasta el final del torneo: volver a
    pedirlos en cada escaneo gasta la mitad de los requests al pedo y acerca el 429.

    `lock_utc` es el seguro: un snapshot tomado ANTES del lock no sirve de base
    (sus especiales todavía podían cambiar — nosotros mismos editamos los nuestros
    el 7/8 a las 14:45, media hora después de la ventana del gate que produjo el
    snapshot de las 14:13). Sin este chequeo, la primera corrida post-lock
    congelaría el pool con datos viejos.

    Solo cuentan como conocidas las filas que YA tienen campeón — las vacías
    (participación comprada después, o gate cerrado cuando se tomó el snapshot)
    se vuelven a pedir."""
    if snapshot is None:
        snapshot = load_latest_snapshot()
    if not snapshot:
        return {}
    if lock_utc is not None:
        generado = datetime.fromisoformat(snapshot["generado_utc"])
        if generado < lock_utc:
            log.warning("el snapshot más nuevo (%s) es PRE-lock (%s) — sus especiales "
                        "todavía podían cambiar: se re-piden todos",
                        generado.isoformat(timespec="minutes"),
                        lock_utc.isoformat(timespec="minutes"))
            return {}
    campos = ("campeon", "campeon_id", "goleador", "goleador_id")
    return {
        int(p["participacion_id"]): {k: p.get(k) for k in campos}
        for p in snapshot.get("participaciones", [])
        if p.get("campeon")
    }


def fetch_snapshot(
    penca_id: int,
    pause_s: float = REQUEST_PAUSE_S,
    strict_gate: bool = True,
    especiales_previos: dict[int, dict] | None = None,
) -> list[RivalPicks]:
    """Baja los picks de todas las participaciones del ranking. Lanza
    CampeonatoNoIniciado si el gate del API sigue cerrado.

    `strict_gate=False` NO aborta ante el 400 del gate: guarda lo que el API sí
    entregue (los gates de marcadores y especiales cambian por separado — el 7/8
    hubo una ventana con especiales abiertos y marcadores cerrados; el vigía
    captura ventanas parciales).

    `especiales_previos` (ver `especiales_conocidos`) saltea el GET de
    campeón/goleador de las participaciones que ya lo tienen: la mitad de los
    requests. Válido solo post-lock, cuando los especiales ya no se editan."""
    ranking = _ranking_pacing(penca_id)
    previos = especiales_previos or {}
    log.info("snapshot: %d participaciones en el ranking%s", len(ranking),
             f" ({len(previos)} con especiales ya conocidos)" if previos else "")

    out: list[RivalPicks] = []
    with httpx.Client(base_url=BASE, timeout=20.0, headers=HEADERS) as c:
        for i, row in enumerate(ranking):
            pid = row.participacion_id
            rival = RivalPicks(
                participacion_id=pid,
                numero=row.numero_participacion,
                puntos=row.puntos_totales,
                exactos=row.cant_resultados_exactos,
            )

            r = _get_pacing(c, f"/front/pencas/{pid}/pronosticosEventos", pause_s)
            if r.status_code == 400 and GATE_MSG in r.text and strict_gate:
                raise CampeonatoNoIniciado(r.text[:120])
            if r.status_code == 200:
                for p in r.json().get("data", []):
                    gl, gv = p.get("golesEquipoLocal"), p.get("golesEquipoVisitante")
                    eid = p.get("encuentroId")
                    if gl is not None and gv is not None and eid is not None:
                        rival.picks[int(eid)] = (int(gl), int(gv))
            else:
                log.warning("pronosticosEventos %d → %d", pid, r.status_code)

            conocido = previos.get(pid)
            if conocido:
                rival.campeon = conocido.get("campeon")
                rival.campeon_id = conocido.get("campeon_id")
                rival.goleador = conocido.get("goleador")
                rival.goleador_id = conocido.get("goleador_id")
            else:
                r = _get_pacing(c, f"/front/pencas/{pid}/pronosticoCampeonGoleador", pause_s)
                if r.status_code == 400 and GATE_MSG in r.text and strict_gate:
                    raise CampeonatoNoIniciado(r.text[:120])
                if r.status_code == 200:
                    d = r.json()
                    eq = d.get("equipoCampeon") or {}
                    gol = d.get("opcionGoleador") or {}
                    rival.campeon, rival.campeon_id = eq.get("nombre"), eq.get("id")
                    rival.goleador, rival.goleador_id = gol.get("goleador"), gol.get("id")

            out.append(rival)
            if (i + 1) % 50 == 0:
                log.info("  %d/%d", i + 1, len(ranking))
    return out


def save_snapshot(rivales: list[RivalPicks], penca_id: int) -> Path:
    from src.utils.versions import latest_version, version_num
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    prev = latest_version(SNAP_DIR.glob("v*_*.json"))
    n = version_num(prev) + 1 if prev else 1
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = SNAP_DIR / f"v{n}_{ts}.json"
    payload = {
        "generado_utc": datetime.now(timezone.utc).isoformat(),
        "penca_id": penca_id,
        "n_participaciones": len(rivales),
        "participaciones": [
            {**asdict(r), "picks": {str(k): list(v) for k, v in r.picks.items()}}
            for r in rivales
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def load_latest_snapshot(max_age_hours: float | None = None) -> dict | None:
    from src.utils.versions import latest_version
    latest = latest_version(SNAP_DIR.glob("v*_*.json")) if SNAP_DIR.exists() else None
    if latest is None:
        return None
    data = json.loads(latest.read_text(encoding="utf-8"))
    if max_age_hours is not None:
        age = (datetime.now(timezone.utc)
               - datetime.fromisoformat(data["generado_utc"])).total_seconds() / 3600
        if age > max_age_hours:
            log.info("snapshot más viejo que %.0fh (%.1fh) — se ignora", max_age_hours, age)
            return None
    return data


# -------------------- Q empírica --------------------

def empirical_counts(
    snapshot: dict, mis_numeros: set[int] | None = None,
) -> dict[int, np.ndarray]:
    """evento_id → conteos de picks sobre los N_SCORES índices.

    `mis_numeros` excluye nuestras participaciones: la Q empírica modela a los
    RIVALES; contarnos (12 planillas anti-chalk sobre ~150) infla justo los
    marcadores diferenciados y hace desaparecer el hueco que explotamos.
    """
    mis = mis_numeros or set()
    counts: dict[int, np.ndarray] = {}
    for r in snapshot.get("participaciones", []):
        if int(r.get("numero", -1)) in mis:
            continue
        for eid_str, (gl, gv) in r.get("picks", {}).items():
            eid = int(eid_str)
            if eid not in counts:
                counts[eid] = np.zeros(N_SCORES)
            counts[eid][score_index(min(int(gl), MAX_GOALS), min(int(gv), MAX_GOALS))] += 1
    return counts


def blended_q(prior_q: np.ndarray, counts: np.ndarray | None, strength: float = 25.0) -> np.ndarray:
    """Posterior Dirichlet: Q = (conteos + strength·prior) / (n + strength).

    Con pocas observaciones domina el prior; con muchas, lo observado. strength≈25
    equivale a "el prior vale 25 rivales imaginarios".
    """
    if counts is None or counts.sum() == 0:
        return prior_q
    q = counts + strength * prior_q
    return q / q.sum()


def empirical_campeon_counts(
    snapshot: dict, equipo_idx: dict[str, int], n_teams: int,
    mis_numeros: set[int] | None = None,
) -> np.ndarray:
    """Conteos de picks de campeón por equipo (índices del optimizador).

    `mis_numeros` excluye nuestras participaciones, igual que en empirical_counts.
    """
    mis = mis_numeros or set()
    counts = np.zeros(n_teams)
    for r in snapshot.get("participaciones", []):
        if int(r.get("numero", -1)) in mis:
            continue
        nombre = r.get("campeon")
        if nombre and nombre in equipo_idx:
            counts[equipo_idx[nombre]] += 1
    return counts


def empirical_goleador_counts(
    snapshot: dict, opciones: list, mis_numeros: set[int] | None = None,
) -> np.ndarray:
    """Conteos de picks de goleador por opción del menú (mismo orden que `opciones`).

    `opciones` son las OpcionGoleador de especiales.fetch_opciones. Matchea por id
    de opción y cae a nombre exacto (el snapshot guarda ambos).
    """
    mis = mis_numeros or set()
    idx_por_id = {o.id: i for i, o in enumerate(opciones)}
    idx_por_nombre = {o.nombre: i for i, o in enumerate(opciones)}
    counts = np.zeros(len(opciones))
    for r in snapshot.get("participaciones", []):
        if int(r.get("numero", -1)) in mis:
            continue
        i = idx_por_id.get(r.get("goleador_id"))
        if i is None:
            i = idx_por_nombre.get(r.get("goleador"))
        if i is not None:
            counts[i] += 1
    return counts


def exact_rate_desde_snapshot(
    snapshot: dict | None,
    resultados: dict[int, tuple[int, int]],
    mis_numeros: set[int] | None = None,
) -> float | None:
    """Tasa de exactos del pool contada de los PICKS PÚBLICOS, no del contador del API.

    Reemplaza a `observed_exact_rate_from_ranking`, que dependía de
    `cantResultadosExactos` — un campo que el API liquida recién al CIERRE de la fecha
    y que trajo dos problemas encadenados:

      * **Canal muerto**: en vivo viene en 0 aunque ya haya exactos. Tomarlo literal
        calibraba al pool más disperso (T=3.0) cuando el real estaba en el más
        concentrado (T=0.5) — al revés, y del lado que nos hace jugar chalk.
      * **Dilución**: el denominador contaba TODOS los partidos jugados de la
        temporada, incluidos los de la fecha en curso que el numerador todavía no
        puede ver. Cuanto más avanzada la fecha, más abajo la tasa.

    Acá el numerador y el denominador salen del MISMO conjunto —los pares
    (rival, partido) donde vemos su pick y el resultado— así que no pueden desfasarse.
    Todo el insumo ya estaba en el snapshot; no hace falta ningún endpoint nuevo.

    `mis_numeros` excluye nuestras participaciones: el observable calibra a los
    RIVALES, y nuestras 12 planillas diversificadas sesgarían la tasa.
    """
    if not snapshot or not resultados:
        return None
    mis = mis_numeros or set()
    aciertos = pares = 0
    for part in snapshot.get("participaciones") or []:
        if part.get("numero") in mis:
            continue
        for eid, pick in (part.get("picks") or {}).items():
            real = resultados.get(int(eid))
            if real is None or not pick:
                continue
            pares += 1
            aciertos += (int(pick[0]), int(pick[1])) == (int(real[0]), int(real[1]))
    if not pares:
        return None
    log.info("tasa de exactos del pool: %.3f (%d aciertos en %d pares rival-partido "
             "observados, calculada de los picks públicos)", aciertos / pares, aciertos, pares)
    return aciertos / pares


def snapshot_summary(snapshot: dict) -> str:
    """Resumen humano para el log/Telegram."""
    n = snapshot.get("n_participaciones", 0)
    counts = empirical_counts(snapshot)
    con_picks = sum(1 for r in snapshot.get("participaciones", []) if r.get("picks"))
    con_campeon = sum(1 for r in snapshot.get("participaciones", []) if r.get("campeon"))
    return (f"{n} participaciones · {con_picks} con marcadores cargados · "
            f"{con_campeon} con campeón · {len(counts)} eventos observados")


# -------------------- CLI --------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--if-started", action="store_true",
                    help="si el campeonato no inició, salir 0 en silencio (para timers)")
    ap.add_argument("--reusar-especiales", action="store_true",
                    help="no re-pedir campeón/goleador ya observados: la MITAD de los "
                         "requests. Válido post-lock, cuando ya no se editan")
    ap.add_argument("--max-edad-h", type=float, default=None,
                    help="si el snapshot más nuevo es más joven que esto, no escanear "
                         "(exit 0) — evita que los timers horarios re-escaneen al pedo")
    ap.add_argument("--pausa", type=float, default=REQUEST_PAUSE_S,
                    help=f"segundos entre requests (default {REQUEST_PAUSE_S})")
    args = ap.parse_args()

    import yaml
    from src.clausura.picks import CONFIG_PATH
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    penca_id = cfg["pencas"]["paga"]["id"]

    if args.max_edad_h is not None and load_latest_snapshot(max_age_hours=args.max_edad_h):
        print(f"snapshot vigente (menos de {args.max_edad_h:g}h) — escaneo omitido")
        sys.exit(0)

    previos = (especiales_conocidos(lock_utc=lock_especiales_utc())
               if args.reusar_especiales else None)

    try:
        rivales = fetch_snapshot(penca_id, pause_s=args.pausa, especiales_previos=previos)
    except RateLimited as e:
        # Sin guardar NADA: media lista es peor que ninguna. Los units lo invocan
        # con "-" para que esto no bloquee al pipeline, que sigue con el snapshot
        # previo (o con la Q modelada si no hay).
        print(f"ERROR: el API nos está limitando ({e}) — escaneo abortado sin guardar",
              file=sys.stderr)
        sys.exit(1)
    except CampeonatoNoIniciado:
        if args.if_started:
            print("campeonato no iniciado — snapshot omitido")
            sys.exit(0)
        print("ERROR: el campeonato no inició; los picks aún no son públicos", file=sys.stderr)
        sys.exit(1)

    path = save_snapshot(rivales, penca_id)
    snap = json.loads(path.read_text(encoding="utf-8"))
    print(f"guardado {path}")
    print(snapshot_summary(snap))


if __name__ == "__main__":
    main()
