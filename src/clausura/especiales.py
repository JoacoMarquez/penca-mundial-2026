"""Especiales de la penca: Campeón y Goleador (25 puntos cada uno, Art. 6).

Son los dos pronósticos de la pestaña "Especial". Pesan 50 puntos contra los ~2.5
de EV de un partido común: elegirlos bien —y sobre todo, DISTINTOS entre
participaciones— es la palanca de diversificación más barata del portfolio.

**Campeón** se modela gratis: el SeasonSimulator ya sortea los 120 resultados de la
temporada en cada simulación, así que el campeón es una función determinística de
esos sorteos (tabla de posiciones: 3/1/0 puntos). Eso captura la correlación que
importa: la participación que pica "campeón Nacional" quiere que sus marcadores y
su campeón ganen JUNTOS en los mismos escenarios.

**Goleador** no se puede derivar de las grillas (no hay nivel jugador). Se modela
como categórica independiente con prior configurable: fuerza de ataque del equipo
(ratings) × share del goleador estrella dentro del equipo. Es la capa más débil del
sistema y está marcada como tal; cuando Supermatch publique las opciones (hoy el
endpoint devuelve 500 = no configurado) y/o haya cuotas de mercado, recalibrar.

Estado del API (2026-08-04):
    GET front/pencas/46/opcionesEquiposCampeon  → 500 (aún no configurado)
    GET front/pencas/46/opcionesGoleador        → 500 (ídem)
    (el shape se verificó contra la penca 41 del Apertura 2026, que sí los tiene)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx
import numpy as np

from src.clausura.api import BASE, HEADERS
from src.clausura.economics import SIDE, index_score

log = logging.getLogger(__name__)

PUNTOS_ESPECIAL = 25


# -------------------- opciones desde el API --------------------

@dataclass(frozen=True)
class OpcionGoleador:
    id: int
    nombre: str
    equipo_id: int


def fetch_opciones(penca_id: int) -> tuple[list[dict] | None, list[OpcionGoleador] | None]:
    """(equipos_campeon, goleadores) — None si el mercado aún no está configurado (500)."""
    campeon, goleador = None, None
    with httpx.Client(base_url=BASE, timeout=20.0, headers=HEADERS) as c:
        r = c.get(f"/front/pencas/{penca_id}/opcionesEquiposCampeon")
        if r.status_code == 200:
            campeon = r.json().get("opcionesEquiposCampeon", {}).get("data", [])
        r = c.get(f"/front/pencas/{penca_id}/opcionesGoleador")
        if r.status_code == 200:
            goleador = [
                OpcionGoleador(id=o["id"], nombre=o["goleador"], equipo_id=o.get("equipoId", -1))
                for o in r.json().get("opcionesGoleador", {}).get("data", [])
            ]
    return campeon, goleador


# -------------------- campeón: tabla de posiciones vectorizada --------------------

def outcome_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(pts_local, pts_visita, dif_gol) indexadas por score_index. Precalculadas."""
    n = SIDE * SIDE
    pl, pv, gd = np.zeros(n, dtype=np.int32), np.zeros(n, dtype=np.int32), np.zeros(n, dtype=np.int32)
    for idx in range(n):
        gl, gv = index_score(idx)
        gd[idx] = gl - gv
        if gl > gv:
            pl[idx] = 3
        elif gl < gv:
            pv[idx] = 3
        else:
            pl[idx] = pv[idx] = 1
    return pl, pv, gd


_PTS_L, _PTS_V, _GD = outcome_tables()


def champions_from_results(
    actual: np.ndarray,          # (n_matches, S) índices de score
    local_de: np.ndarray,        # (n_matches,) índice de equipo local
    visita_de: np.ndarray,       # (n_matches,) índice de equipo visitante
    n_teams: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Campeón por simulación: (S,) índice de equipo.

    Tabla: 3/1/0. Desempate por diferencia de gol, luego goles a favor, luego azar.
    Nota honesta: el reglamento AUF define el título del Clausura con desempate por
    partido extra si hay igualdad de puntos en la cima; a efectos del modelo lo
    aproximamos con dif de gol — el error afecta solo a los escenarios empatados.
    """
    n_matches, S = actual.shape
    pts = np.zeros((n_teams, S), dtype=np.int64)
    gd = np.zeros((n_teams, S), dtype=np.int64)
    gf = np.zeros((n_teams, S), dtype=np.int64)

    goles_l = np.array([index_score(i)[0] for i in range(SIDE * SIDE)])
    goles_v = np.array([index_score(i)[1] for i in range(SIDE * SIDE)])

    for m in range(n_matches):
        a = actual[m]                      # (S,)
        li, vi = local_de[m], visita_de[m]
        pts[li] += _PTS_L[a]
        pts[vi] += _PTS_V[a]
        gd[li] += _GD[a]
        gd[vi] -= _GD[a]
        gf[li] += goles_l[a]
        gf[vi] += goles_v[a]

    # clave lexicográfica: puntos ≫ dif gol ≫ goles a favor ≫ ruido para desempatar
    ruido = rng.random((n_teams, S))
    clave = pts * 1e9 + gd * 1e5 + gf * 1e2 + ruido
    return np.argmax(clave, axis=0)


def p_campeon(champs: np.ndarray, n_teams: int) -> np.ndarray:
    """P(campeón) por equipo desde las simulaciones."""
    return np.bincount(champs, minlength=n_teams) / len(champs)


def p_campeon_from_grids(
    grids: list[np.ndarray],
    local_de: np.ndarray,
    visita_de: np.ndarray,
    n_teams: int,
    n_sims: int = 3000,
    seed: int = 20260807,
) -> np.ndarray:
    """P(campeón) sorteando temporadas desde las grillas (para el prior del pool).

    Es un pre-cálculo independiente del simulador del portfolio (que después deriva
    su propio campeón de SUS sorteos, correlacionado con los marcadores).
    """
    from src.clausura.economics import flatten_grid
    rng = np.random.default_rng(seed)
    actual = np.stack([
        rng.choice(SIDE * SIDE, size=n_sims, p=flatten_grid(g)) for g in grids
    ])
    champs = champions_from_results(actual, local_de, visita_de, n_teams, rng)
    return p_campeon(champs, n_teams)


# -------------------- priors del pool para especiales --------------------

def pool_campeon_distribution(
    p_champ: np.ndarray,
    equipos: list[str],
    chalk_strength: float = 0.8,
    big_club_bias: float = 2.5,
) -> np.ndarray:
    """Qué campeón pica el pencista típico.

    El pool sobre-pica a los grandes (Peñarol/Nacional) incluso más allá de su
    probabilidad real — sesgo de hincha + folclore. big_club_bias multiplica su masa.
    """
    eps = 1e-9
    score = np.log(p_champ + eps) * chalk_strength
    for i, e in enumerate(equipos):
        if e in ("Peñarol", "Nacional"):
            score[i] += np.log(big_club_bias)
    score -= score.max()
    q = np.exp(score)
    return q / q.sum()


def goleador_prior_from_ratings(
    opciones: list[OpcionGoleador],
    equipos: list[str],
    equipo_id_por_nombre: dict[str, int],
    ataque: dict[str, float],
    star_share: float = 0.6,
    otros_mass: float = 0.15,
) -> np.ndarray:
    """Prior P(goleador) por opción del menú.

    P(equipo del goleador) ∝ exp(ataque) (los goleadores salen de los equipos que
    más convierten), repartido dentro del equipo: `star_share` para el primero
    listado de cada equipo (Supermatch lista primero al candidato natural) y el
    resto uniforme. `star_share` debe ser > 1/k o la "estrella" quedaría debajo de
    los suplentes (con k=2: 0.6 vs 0.4 ✓). "Otros" recibe `otros_mass` fijo.

    CAPA DÉBIL: sin datos por jugador esto es un prior razonable, no un modelo.
    Recalibrar con cuotas de mercado cuando existan.
    """
    id_a_nombre = {v: k for k, v in equipo_id_por_nombre.items()}
    peso_equipo: dict[int, float] = {}
    for o in opciones:
        nombre = id_a_nombre.get(o.equipo_id)
        if nombre is not None and nombre in ataque:
            peso_equipo[o.equipo_id] = float(np.exp(2.0 * ataque[nombre]))

    total_eq = sum(peso_equipo.values()) or 1.0
    p = np.zeros(len(opciones))
    vistos_por_equipo: dict[int, int] = {}
    n_por_equipo: dict[int, int] = {}
    for o in opciones:
        n_por_equipo[o.equipo_id] = n_por_equipo.get(o.equipo_id, 0) + 1

    for i, o in enumerate(opciones):
        if o.equipo_id not in peso_equipo:      # "Otros" y equipos sin rating
            continue
        p_eq = (1 - otros_mass) * peso_equipo[o.equipo_id] / total_eq
        k = n_por_equipo[o.equipo_id]
        orden = vistos_por_equipo.get(o.equipo_id, 0)
        vistos_por_equipo[o.equipo_id] = orden + 1
        if k == 1:
            p[i] = p_eq
        elif orden == 0:
            p[i] = p_eq * star_share
        else:
            p[i] = p_eq * (1 - star_share) / (k - 1)

    sin_equipo = [i for i, o in enumerate(opciones) if o.equipo_id not in peso_equipo]
    for i in sin_equipo:
        p[i] = otros_mass / max(len(sin_equipo), 1)

    return p / p.sum() if p.sum() > 0 else np.full(len(opciones), 1 / len(opciones))


def pool_goleador_distribution(p_gol: np.ndarray, chalk_strength: float = 0.9) -> np.ndarray:
    """El pool pica goleador ~proporcional al prior, algo más concentrado en el top."""
    eps = 1e-9
    score = np.log(p_gol + eps) * chalk_strength
    score -= score.max()
    q = np.exp(score)
    return q / q.sum()
