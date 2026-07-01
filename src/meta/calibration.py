"""Calibración online del modelo del pool (Capa 5) por ranking-inversion.

La web no expone picks individuales, solo el leaderboard. Pero después de cada partido,
el DELTA de puntos de cada entry entre dos snapshots consecutivos revela cuántos puntos
sacó cada uno en ese partido. Dado el marcador real, eso particiona al pool en clases
de puntos {5, 4, 3, 1, 0} cuyas proporciones observadas se comparan con las que predice
`pool_pick_distribution(grid, config)` — y se ajustan los hiperparámetros por grid search.

Parámetros calibrados:
    chalk_strength: cuánto sigue el pool al mercado.
    bias_scale (β): exponente sobre el popular_score_bias (β=1 → prior tal cual, β=0 → sin sesgo).
    no_show_frac: fracción del pool que no cargó pick (sacan 0 sin información sobre Q).

Regularización: penalty L2 hacia el prior, decae con 1/n_observaciones — con 1 partido
la calibración se mueve poco; con 10+ domina la evidencia.

Flujo:
    1. Postmortem de cada partido → `snapshot_leaderboard()` → data/pool_snapshots/{match_id}.json
    2. `build_observations()` arma (shares observados, grid del mercado, marcador real) por partido.
    3. `calibrate()` → grid search → data/pool_calibration.json (current + history).
    4. El pipeline llama `get_pool_config()` en vez de `PoolModelConfig()`.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.meta.pool import DEFAULT_POPULAR_SCORE_BIAS, PoolModelConfig, pool_pick_distribution

log = logging.getLogger(__name__)

# Clases de puntos del sistema JMLM (regla vigente desde 2026-06-12: 6/4/3/0).
# OJO: si la regla cambia, los snapshots viejos quedan en otra escala — archivarlos
# y arrancar la cadena de observaciones de cero (la regularización cubre el arranque).
POINT_CLASSES = (6, 4, 3, 0)

# Prior (tiene que coincidir con los defaults de PoolModelConfig)
PRIOR_CHALK = 0.7
PRIOR_BIAS_SCALE = 1.0
PRIOR_NO_SHOW = 0.05

# Espacio de búsqueda del grid search
CHALK_GRID = np.round(np.arange(0.20, 1.65, 0.05), 2)
BIAS_GRID = np.round(np.arange(0.0, 2.05, 0.10), 2)
NO_SHOW_GRID = np.round(np.arange(0.0, 0.31, 0.05), 2)

# Peso del penalty hacia el prior (se divide por n_obs)
REG_WEIGHT = 0.05


def _data_dir() -> Path:
    return Path(os.environ.get("DATA_DIR", "data"))


def snapshots_dir() -> Path:
    return _data_dir() / "pool_snapshots"


def calibration_path() -> Path:
    return _data_dir() / "pool_calibration.json"


# ============ snapshots ============

def snapshot_leaderboard(
    match_id: str | int,
    finished_matches: list[str | int],
    entries: list[dict] | None = None,
) -> Path | None:
    """Persiste el estado del leaderboard tras el partido `match_id`.

    `finished_matches` = ids con resultado al momento del snapshot (incluyendo este).
    Si `entries` es None, lo trae de la API.
    """
    if entries is None:
        entries = _fetch_leaderboard_entries()
    if not entries:
        log.warning("snapshot_leaderboard: sin entries para %s", match_id)
        return None
    out = {
        "match_id": match_id,
        "taken_at": datetime.now(timezone.utc).isoformat(),
        "finished_matches": sorted(str(m) for m in finished_matches),
        "entries": [
            {
                "penca_id": int(e.get("penca_id", 0)),
                "points_total": int(e.get("points_total", 0)),
                "exact_scores": int(e.get("exact_scores", 0)),
                "correct_winners": int(e.get("correct_winners", 0)),
                "predictions_made": int(e.get("predictions_made", 0)),
            }
            for e in entries
        ],
    }
    d = snapshots_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{match_id}.json"
    path.write_text(json.dumps(out, ensure_ascii=False))
    log.info("snapshot leaderboard guardado: %s (%d entries)", path, len(out["entries"]))
    return path


def _fetch_leaderboard_entries() -> list[dict] | None:
    # Un snapshot debe ser el leaderboard REAL post-partido: exigimos datos frescos y
    # nunca persistimos stale (rechazamos res["stale"]). El fetch exitoso sí refresca el
    # cache compartido de src.utils.leaderboard para los demás consumidores.
    from src.utils.leaderboard import fetch_leaderboard
    res = fetch_leaderboard(
        os.environ.get("PENCA_API_BASE_URL", ""),
        os.environ.get("PENCA_API_KEY", ""),
        timeout=10.0,
    )
    if res["stale"] or not res["entries"]:
        if res.get("error"):
            log.warning("fetch leaderboard falló: %s", res["error"])
        return None
    return res["entries"]


# ============ observaciones ============

@dataclass(frozen=True)
class Observation:
    """Lo observado del pool en UN grupo de partidos co-puntuados (≥1) — TRES canales:
    shares de clases de puntos CONJUNTAS + fracción de exactos + fracción de ganadores.

    Un grupo de 1 partido (el caso normal) se reduce exactamente al comportamiento de
    siempre. Los grupos de 2+ aparecen cuando varios partidos se puntúan en el mismo tick
    (p.ej. la 3ª jornada de grupos, simultánea por diseño FIFA): el leaderboard no permite
    separarlos, pero la distribución de la SUMA de puntos es la convolución de las
    distribuciones individuales, que sí conocemos.
    """
    match_id: str                          # "106" o "106+107" para grupos
    actuals: tuple[tuple[int, int], ...]   # marcador real de cada partido del grupo
    shares: dict[int, float]               # clase de puntos CONJUNTA → proporción (suman 1)
    n_entries: int
    grids: tuple[np.ndarray, ...]          # grid Poisson de cada partido del grupo
    exact_frac: float | None = None        # fracción media (por partido) de exactos
    winner_frac: float | None = None       # fracción media (por partido) de ganadores

    @property
    def n_matches(self) -> int:
        return len(self.grids)


def _joint_point_classes(n_matches: int) -> tuple[int, ...]:
    """Clases de puntos conjuntas alcanzables sumando `n_matches` partidos (cada uno
    aporta una clase de POINT_CLASSES). Para n=1 → las clases de siempre."""
    sums = {0}
    for _ in range(n_matches):
        sums = {s + c for s in sums for c in POINT_CLASSES}
    return tuple(sorted(sums))


def _convolve_point_classes(per_match_shares: list[dict[int, float]]) -> dict[int, float]:
    """Convoluciona las distribuciones de clases de puntos de varios partidos →
    distribución de la SUMA de puntos. Para un solo partido devuelve sus shares tal cual.

    Las colisiones (p.ej. 6+0 y 3+3 dan ambos 6) suman probabilidad — por eso no hay
    mala atribución: modelamos la suma, no quién aportó cada parte.
    """
    dist: dict[int, float] = {0: 1.0}
    for sh in per_match_shares:
        nxt: dict[int, float] = {}
        for s, ps in dist.items():
            for c, pc in sh.items():
                nxt[s + c] = nxt.get(s + c, 0.0) + ps * pc
        dist = nxt
    return dist


def _points_delta_shares(
    prev_entries: list[dict] | None,
    cur_entries: list[dict],
    valid_classes: tuple[int, ...] = POINT_CLASSES,
    n_matches: int = 1,
) -> tuple[dict[int, float], int, float | None, float | None]:
    """Shares de puntos CONJUNTOS + fracción media de exactos + de ganadores entre dos snapshots.

    prev=None → baseline 0 para todos. Excluye entries con predictions_made == 0
    (no-shows totales). Deltas de puntos fuera de `valid_classes` (entró tarde, etc.)
    se descartan; los deltas válidos de exactos/ganadores caen en {0..n_matches}.
    Las fracciones de exactos/ganadores se normalizan por partido (delta combinado / n_matches).
    """
    prev_by = {e["penca_id"]: e for e in (prev_entries or [])}
    counts: dict[int, int] = {c: 0 for c in valid_classes}
    n = 0
    exact_hits = exact_n = 0
    winner_hits = winner_n = 0
    for e in cur_entries:
        if not e.get("predictions_made"):
            continue
        prev = prev_by.get(e["penca_id"], {})
        delta = e["points_total"] - prev.get("points_total", 0)
        if delta in counts:
            counts[delta] += 1
            n += 1
        d_exact = e.get("exact_scores", 0) - prev.get("exact_scores", 0)
        if 0 <= d_exact <= n_matches:
            exact_hits += d_exact
            exact_n += 1
        d_winner = e.get("correct_winners", 0) - prev.get("correct_winners", 0)
        if 0 <= d_winner <= n_matches:
            winner_hits += d_winner
            winner_n += 1
    if n == 0:
        return {}, 0, None, None
    shares = {c: counts[c] / n for c in valid_classes}
    exact_frac = (exact_hits / (exact_n * n_matches)) if exact_n else None
    winner_frac = (winner_hits / (winner_n * n_matches)) if winner_n else None
    return shares, n, exact_frac, winner_frac


def build_observations(data_dir: Path | None = None) -> list[Observation]:
    """Arma observaciones desde los snapshots + postmortems + predicciones persistidas.

    Los `finished_matches` crecen monótonamente (= todos los postmortems en disco), así
    que los snapshots forman una cadena totalmente ordenada por inclusión. El predecesor
    de un snapshot S es el mayor subconjunto estricto de S.finished presente; el GRUPO de
    partidos nuevos = S.finished − predecesor. Si el grupo tiene 2+ partidos (se puntuaron
    en el mismo tick, p.ej. jornada 3 simultánea), la observación es conjunta y su
    distribución predicha es la convolución de los partidos del grupo.
    """
    from src.model.poisson import score_grid
    from src.utils.versions import latest_version

    base = data_dir or _data_dir()
    sdir = base / "pool_snapshots"
    if not sdir.exists():
        return []

    snaps = []
    for f in sorted(sdir.glob("*.json")):
        try:
            snaps.append(json.loads(f.read_text()))
        except Exception:
            continue

    # Dedup por finished + orden por tamaño → cadena de inclusión
    by_finished = {frozenset(s["finished_matches"]): s for s in snaps}
    chain = sorted(by_finished.items(), key=lambda kv: len(kv[0]))

    observations = []
    for i, (cur, s) in enumerate(chain):
        # predecesor = mayor subconjunto estricto de `cur` presente (el más cercano hacia atrás)
        prev_entries = None
        group = cur
        for j in range(i - 1, -1, -1):
            pset = chain[j][0]
            if pset < cur:
                prev_entries = chain[j][1]["entries"]
                group = cur - pset
                break
        if not group:
            continue

        group_ids = sorted(group)
        actuals: list[tuple[int, int]] = []
        grids: list = []
        ok = True
        for gid in group_ids:
            pm_path = base / "postmortems" / f"{gid}.json"
            if not pm_path.exists():
                ok = False
                break
            try:
                pm = json.loads(pm_path.read_text())
                actuals.append((int(pm["actual_home"]), int(pm["actual_away"])))
            except Exception:
                ok = False
                break
            pdir = base / "predictions" / gid
            latest = latest_version(pdir.glob("v*_*.json")) if pdir.exists() else None
            if latest is None:
                ok = False
                break
            try:
                c = json.loads(latest.read_text())["constraints"]
                grids.append(score_grid(c["lambda_L"], c["lambda_V"], c.get("lambda_12", 0.1), max_goals=7))
            except Exception:
                ok = False
                break
        if not ok:
            continue

        n_m = len(group_ids)
        shares, n, exact_frac, winner_frac = _points_delta_shares(
            prev_entries, s["entries"], _joint_point_classes(n_m), n_m,
        )
        if n < 30:  # muy pocos datos → ruido
            continue
        observations.append(Observation(
            match_id="+".join(group_ids), actuals=tuple(actuals), shares=shares,
            n_entries=n, grids=tuple(grids), exact_frac=exact_frac, winner_frac=winner_frac,
        ))

    return observations


# ============ predicción de shares ============

def point_class_masks(actual: tuple[int, int], n: int, points_rule=None) -> dict[int, np.ndarray]:
    """Máscara booleana (n×n) por clase de puntos para un marcador real dado."""
    if points_rule is None:
        from src.model.poisson import jmlm_points
        points_rule = jmlm_points
    masks = {c: np.zeros((n, n), dtype=bool) for c in POINT_CLASSES}
    for gL in range(n):
        for gV in range(n):
            pts = points_rule((gL, gV), actual)
            if pts in masks:
                masks[pts][gL, gV] = True
    return masks


def _single_match_shares(
    grid: np.ndarray, actual: tuple[int, int], config: PoolModelConfig,
) -> dict[int, float]:
    """Shares de clases de puntos {6,4,3,0} para UN partido (sin no_show)."""
    q = pool_pick_distribution(grid, config)
    masks = point_class_masks(actual, grid.shape[0])
    return {c: float(q[m].sum()) for c, m in masks.items()}


def predicted_group_shares(
    grids,
    actuals,
    config: PoolModelConfig,
    no_show_frac: float = 0.0,
) -> dict[int, float]:
    """Shares de clases de puntos CONJUNTAS que el modelo predice para un grupo de
    partidos co-puntuados — convolución de las shares individuales.

    no-shows silenciosos: una penca que no cargó puntúa 0 en TODOS los partidos del
    grupo → masa extra en la clase conjunta 0.
    """
    per_match = [_single_match_shares(g, a, config) for g, a in zip(grids, actuals)]
    shares = _convolve_point_classes(per_match)
    if no_show_frac > 0:
        shares = {c: v * (1 - no_show_frac) for c, v in shares.items()}
        shares[0] = shares.get(0, 0.0) + no_show_frac
    return shares


def predicted_shares(
    grid: np.ndarray,
    actual: tuple[int, int],
    config: PoolModelConfig,
    no_show_frac: float = 0.0,
) -> dict[int, float]:
    """Shares de un solo partido (equivale a `predicted_group_shares` con grupo de 1)."""
    return predicted_group_shares([grid], [actual], config, no_show_frac)


def _config_for(chalk: float, bias_scale: float) -> PoolModelConfig:
    return PoolModelConfig(
        chalk_strength=float(chalk),
        popular_score_bias={k: v ** bias_scale for k, v in DEFAULT_POPULAR_SCORE_BIAS.items()},
    )


def predicted_exact_winner(
    grids,
    actuals,
    config: PoolModelConfig,
    no_show_frac: float = 0.0,
) -> tuple[float, float]:
    """Fracción MEDIA (por partido) de exactos y de ganadores que el modelo predice para
    el grupo. La esperanza de la suma = suma de esperanzas → promediar los partidos basta
    (sin convolución), igual que el canal escalar observado."""
    tot_e = tot_w = 0.0
    for grid, actual in zip(grids, actuals):
        q = pool_pick_distribution(grid, config)
        n = grid.shape[0]
        agL, agV = actual
        tot_e += float(q[agL, agV])
        aw = "H" if agL > agV else ("A" if agL < agV else "D")
        for gL in range(n):
            for gV in range(n):
                pw = "H" if gL > gV else ("A" if gL < gV else "D")
                if pw == aw:
                    tot_w += float(q[gL, gV])
    n_m = max(len(grids), 1)
    return (tot_e / n_m) * (1 - no_show_frac), (tot_w / n_m) * (1 - no_show_frac)


# Pesos de los canales en el loss. Los 3 canales miden lo mismo (la forma de Q) con
# granularidad distinta: el de exactos es el más informativo sobre concentración.
W_POINTS, W_EXACT, W_WINNER = 1.0, 2.0, 1.0


def _loss(observations: list[Observation], chalk: float, bias_scale: float, no_show: float) -> float:
    """L2 multicanal (clases de puntos + frac. exactos + frac. ganadores), promediado
    entre partidos, + penalty hacia el prior."""
    cfg = _config_for(chalk, bias_scale)
    total = 0.0
    for ob in observations:
        pred = predicted_group_shares(ob.grids, ob.actuals, cfg, no_show)
        classes = set(pred) | set(ob.shares)
        l = W_POINTS * sum((pred.get(c, 0.0) - ob.shares.get(c, 0.0)) ** 2 for c in classes)
        if ob.exact_frac is not None or ob.winner_frac is not None:
            pe, pw = predicted_exact_winner(ob.grids, ob.actuals, cfg, no_show)
            if ob.exact_frac is not None:
                l += W_EXACT * (pe - ob.exact_frac) ** 2
            if ob.winner_frac is not None:
                l += W_WINNER * (pw - ob.winner_frac) ** 2
        total += l
    total /= max(len(observations), 1)
    reg = REG_WEIGHT / max(len(observations), 1) * (
        (chalk - PRIOR_CHALK) ** 2
        + 0.5 * (bias_scale - PRIOR_BIAS_SCALE) ** 2
        + (no_show - PRIOR_NO_SHOW) ** 2
    )
    return total + reg


# ============ calibración ============

def calibrate(observations: list[Observation]) -> dict[str, Any] | None:
    """Grid search sobre (chalk_strength, bias_scale, no_show_frac). Devuelve el fit."""
    if not observations:
        return None
    best = None
    for chalk in CHALK_GRID:
        for beta in BIAS_GRID:
            for ns in NO_SHOW_GRID:
                l = _loss(observations, chalk, beta, ns)
                if best is None or l < best[0]:
                    best = (l, chalk, beta, ns)
    loss, chalk, beta, ns = best
    prior_loss = _loss(observations, PRIOR_CHALK, PRIOR_BIAS_SCALE, PRIOR_NO_SHOW)
    fit = {
        "chalk_strength": float(chalk),
        "bias_scale": float(beta),
        "no_show_frac": float(ns),
        "loss": float(loss),
        "prior_loss": float(prior_loss),
        "improvement_pct": float((1 - loss / prior_loss) * 100) if prior_loss > 0 else 0.0,
        "n_observations": len(observations),
        "matches": [ob.match_id for ob in observations],
        "fitted_at": datetime.now(timezone.utc).isoformat(),
    }
    log.info(
        "calibración: chalk=%.2f β=%.2f no_show=%.2f | loss %.5f (prior %.5f, −%.1f%%) | n=%d",
        chalk, beta, ns, loss, prior_loss, fit["improvement_pct"], len(observations),
    )
    return fit


def save_calibration(fit: dict[str, Any]) -> Path:
    path = calibration_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"current": None, "history": []}
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except Exception:
            pass
    data["current"] = fit
    data.setdefault("history", []).append(fit)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    return path


def load_calibration() -> dict[str, Any] | None:
    path = calibration_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text()).get("current")
    except Exception:
        return None


def get_pool_config() -> PoolModelConfig:
    """Config del pool para el pipeline: calibrada si existe, prior si no."""
    cal = load_calibration()
    if not cal:
        return PoolModelConfig()
    return _config_for(cal["chalk_strength"], cal["bias_scale"])


def recalibrate_from_disk() -> dict[str, Any] | None:
    """Rearma observaciones desde disco, calibra y persiste. Llamar post-snapshot."""
    obs = build_observations()
    if not obs:
        log.info("recalibrate: sin observaciones todavía")
        return None
    fit = calibrate(obs)
    if fit:
        save_calibration(fit)
    return fit


# ============ CLI ============

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    ap = argparse.ArgumentParser(description="Calibración del pool por ranking-inversion")
    ap.add_argument("--recalibrate", action="store_true", help="recalibrar desde snapshots en disco")
    ap.add_argument("--show", action="store_true", help="mostrar calibración actual")
    args = ap.parse_args()

    if args.show:
        print(json.dumps(load_calibration(), indent=2, ensure_ascii=False))
    if args.recalibrate:
        fit = recalibrate_from_disk()
        print(json.dumps(fit, indent=2, ensure_ascii=False))
