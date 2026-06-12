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
    import httpx
    base = os.environ.get("PENCA_API_BASE_URL", "").rstrip("/")
    key = os.environ.get("PENCA_API_KEY", "")
    if not base or not key:
        return None
    try:
        with httpx.Client(timeout=10.0, headers={"Authorization": f"Bearer {key}"}) as c:
            r = c.get(f"{base}/leaderboard")
        if r.status_code != 200:
            return None
        return r.json().get("entries", [])
    except Exception as e:
        log.warning("fetch leaderboard falló: %s", e)
        return None


# ============ observaciones ============

@dataclass(frozen=True)
class Observation:
    """Lo observado del pool en UN partido."""
    match_id: str
    actual: tuple[int, int]
    shares: dict[int, float]        # clase de puntos → proporción observada (suman 1)
    n_entries: int
    grid: np.ndarray                # grid Poisson del partido (modelo nuestro)


def _points_delta_shares(prev_entries: list[dict] | None, cur_entries: list[dict]) -> tuple[dict[int, float], int]:
    """Shares de puntos ganados entre dos snapshots. prev=None → baseline 0 para todos.

    Excluye entries con predictions_made == 0 en el snapshot actual (no-shows totales).
    Deltas que no son clases válidas (entró tarde al pool, doble jornada) se descartan.
    """
    prev_pts = {e["penca_id"]: e["points_total"] for e in (prev_entries or [])}
    counts: dict[int, int] = {c: 0 for c in POINT_CLASSES}
    n = 0
    for e in cur_entries:
        if not e.get("predictions_made"):
            continue
        delta = e["points_total"] - prev_pts.get(e["penca_id"], 0)
        if delta in counts:
            counts[delta] += 1
            n += 1
    if n == 0:
        return {}, 0
    return {c: counts[c] / n for c in POINT_CLASSES}, n


def build_observations(data_dir: Path | None = None) -> list[Observation]:
    """Arma observaciones desde los snapshots + postmortems + predicciones persistidas.

    Un snapshot S es usable si existe otro snapshot cuyo finished_matches sea
    exactamente S.finished_matches − {S.match_id} (o S es el primer partido del torneo).
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

    by_finished = {frozenset(s["finished_matches"]): s for s in snaps}
    observations = []
    for s in snaps:
        mid = str(s["match_id"])
        finished = frozenset(s["finished_matches"])
        prev_set = finished - {mid}
        if prev_set:
            prev = by_finished.get(prev_set)
            if prev is None:
                log.info("snapshot %s sin predecesor exacto — lo salto", mid)
                continue
            prev_entries = prev["entries"]
        else:
            prev_entries = None  # primer partido: baseline 0

        # marcador real desde el postmortem
        pm_path = base / "postmortems" / f"{mid}.json"
        if not pm_path.exists():
            continue
        try:
            pm = json.loads(pm_path.read_text())
            actual = (int(pm["actual_home"]), int(pm["actual_away"]))
        except Exception:
            continue

        # grid del partido desde la última predicción
        pdir = base / "predictions" / mid
        latest = latest_version(pdir.glob("v*_*.json")) if pdir.exists() else None
        if latest is None:
            continue
        try:
            c = json.loads(latest.read_text())["constraints"]
            grid = score_grid(c["lambda_L"], c["lambda_V"], c.get("lambda_12", 0.1), max_goals=7)
        except Exception:
            continue

        shares, n = _points_delta_shares(prev_entries, s["entries"])
        if n < 30:  # muy pocos datos → ruido
            continue
        observations.append(Observation(match_id=mid, actual=actual, shares=shares, n_entries=n, grid=grid))

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


def predicted_shares(
    grid: np.ndarray,
    actual: tuple[int, int],
    config: PoolModelConfig,
    no_show_frac: float = 0.0,
) -> dict[int, float]:
    """Shares de clases de puntos que el modelo del pool predice para este partido."""
    q = pool_pick_distribution(grid, config)
    masks = point_class_masks(actual, grid.shape[0])
    shares = {c: float(q[m].sum()) for c, m in masks.items()}
    # no-shows silenciosos: masa extra en la clase 0
    if no_show_frac > 0:
        shares = {c: v * (1 - no_show_frac) for c, v in shares.items()}
        shares[0] = shares.get(0, 0.0) + no_show_frac
    return shares


def _config_for(chalk: float, bias_scale: float) -> PoolModelConfig:
    return PoolModelConfig(
        chalk_strength=float(chalk),
        popular_score_bias={k: v ** bias_scale for k, v in DEFAULT_POPULAR_SCORE_BIAS.items()},
    )


def _loss(observations: list[Observation], chalk: float, bias_scale: float, no_show: float) -> float:
    """L2 sobre shares por clase, promediado entre partidos, + penalty hacia el prior."""
    cfg = _config_for(chalk, bias_scale)
    total = 0.0
    for ob in observations:
        pred = predicted_shares(ob.grid, ob.actual, cfg, no_show)
        total += sum((pred[c] - ob.shares.get(c, 0.0)) ** 2 for c in POINT_CLASSES)
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
