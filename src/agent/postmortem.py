"""Postmortem automático después de cada partido.

Para cada partido finalizado:
    1. Obtiene resultado real (penca API).
    2. Recupera la última predicción nuestra (la versión publicada).
    3. Calcula puntos por penca con la regla JMLM.
    4. Compara contra el pool (top, mediana) vía leaderboard real.
    5. Genera insights automáticos:
        - Qué estrategia funcionó mejor / peor
        - ¿Coincidió con el modal del mercado?
        - ¿Sirvió el ajuste LLM?
        - ¿Mejoramos o empeoramos vs la jornada anterior?
    6. Notifica por Telegram + persiste JSON en data/postmortems/.

Trigger: el scheduler corre `find_finished_matches_pending_postmortem()` y dispara este módulo
~150 min después del kickoff (90' match + 30' adicional + 30' margen).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx

from src.model.poisson import jmlm_points
from src.utils.versions import latest_version

log = logging.getLogger(__name__)


# ============ data classes ============

@dataclass
class PencaResult:
    penca_id: int
    penca_name: str | None
    strategy_used: str         # ev | differentiated | tail | upset | variance
    predicted_score: tuple[int, int]
    points_earned: int
    is_exact: bool             # 5 pts
    correct_winner: bool       # ganador correcto


@dataclass
class PostmortemReport:
    match_id: str | int
    home_team: str
    away_team: str
    actual_home: int
    actual_away: int
    final_score: str           # "1-0"

    pencas_results: list[PencaResult] = field(default_factory=list)
    portfolio_max_points: int = 0
    portfolio_total_points: int = 0

    # Pool comparison
    pool_top_points: int | None = None
    pool_median_points: float | None = None
    our_best_penca_id: int | None = None
    our_best_rank_in_pool: int | None = None

    # Diagnostics
    market_modal_score: tuple[int, int] | None = None
    market_modal_was_right: bool | None = None
    llm_adjustment_applied: dict | None = None
    llm_adjustment_was_helpful: str | None = None  # "yes" | "no" | "neutral"

    insights: list[str] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ============ helpers de I/O ============

def _data_dir() -> Path:
    return Path(os.environ.get("DATA_DIR", "data"))


def postmortem_path(match_id: str | int) -> Path:
    return _data_dir() / "postmortems" / f"{match_id}.json"


def postmortem_done(match_id: str | int) -> bool:
    return postmortem_path(match_id).exists()


def _latest_prediction(match_id: str | int) -> dict | None:
    """Última versión de predicción para un partido."""
    pdir = _data_dir() / "predictions" / str(match_id)
    if not pdir.exists():
        return None
    latest = latest_version(pdir.glob("v*_*.json"))
    if latest is None:
        return None
    return json.loads(latest.read_text())


def _fetch_match_status(match_id: str | int) -> dict | None:
    """Pregunta a la penca API el estado actual del partido (status, score)."""
    base = os.environ.get("PENCA_API_BASE_URL", "").rstrip("/")
    key = os.environ.get("PENCA_API_KEY", "")
    if not base or not key:
        return None
    try:
        with httpx.Client(
            timeout=10.0,
            headers={"Authorization": f"Bearer {key}"},
        ) as c:
            r = c.get(f"{base}/matches/{match_id}")
        if r.status_code == 200:
            return r.json()
        # Fallback: buscar en /matches (lista)
        r2 = c.get(f"{base}/matches")
        if r2.status_code == 200:
            for m in r2.json().get("matches", []):
                if str(m.get("id")) == str(match_id):
                    return m
    except Exception as e:
        log.warning("fetch_match_status falló: %s", e)
    return None


def _fetch_full_leaderboard() -> list[dict] | None:
    # El postmortem compara contra el pool del momento: exigimos leaderboard fresco (no
    # stale) para no persistir top/mediana desactualizados en la historia. El fetch exitoso
    # refresca el cache compartido de src.utils.leaderboard.
    from src.utils.leaderboard import fetch_leaderboard
    res = fetch_leaderboard(
        os.environ.get("PENCA_API_BASE_URL", ""),
        os.environ.get("PENCA_API_KEY", ""),
        timeout=10.0,
    )
    if res["stale"] or not res["entries"]:
        return None
    return res["entries"]


# ============ core computation ============

def find_finished_matches_pending_postmortem(fixtures: dict) -> list[str | int]:
    """Devuelve match_ids que: (a) deberían haber terminado ya, (b) tienen predicción nuestra, (c) no tienen postmortem."""
    now = datetime.now(timezone.utc)
    pending = []
    all_matches = (fixtures.get("fase_grupos") or []) + (fixtures.get("eliminatorias") or [])
    for m in all_matches:
        if not m.get("kickoff_utc"):
            continue
        try:
            ko = datetime.fromisoformat(m["kickoff_utc"].replace("Z", "+00:00"))
        except Exception:
            continue
        # Después de 150 min del kickoff lo consideramos "terminado"
        if now < ko + timedelta(minutes=150):
            continue
        mid = m["id"]
        # ¿Tenemos predicción para este partido?
        if not (_data_dir() / "predictions" / str(mid)).exists():
            continue
        # ¿Ya tiene postmortem?
        if postmortem_done(mid):
            continue
        pending.append(mid)
    return pending


def _generate_insights(report: PostmortemReport) -> list[str]:
    """Insights automáticos en base al resultado."""
    out = []
    actual = (report.actual_home, report.actual_away)

    # 1. ¿Qué estrategia ganó?
    if report.pencas_results:
        best = max(report.pencas_results, key=lambda p: p.points_earned)
        out.append(
            f"Mejor estrategia: <b>{best.strategy_used}</b> "
            f"({best.predicted_score[0]}-{best.predicted_score[1]} → {best.points_earned}pts)"
        )

    # 2. ¿Coincidió con el modal del mercado?
    if report.market_modal_score and report.market_modal_was_right is not None:
        if report.market_modal_was_right:
            out.append(f"El mercado tenía razón: modal era {report.market_modal_score[0]}-{report.market_modal_score[1]}, resultado real {actual[0]}-{actual[1]}")
        else:
            out.append(f"Mercado se equivocó: modal era {report.market_modal_score[0]}-{report.market_modal_score[1]} pero salió {actual[0]}-{actual[1]}")

    # 3. ¿Ayudó el LLM?
    if report.llm_adjustment_was_helpful == "yes":
        out.append("✅ El ajuste LLM fue en la dirección correcta (mejoró la predicción).")
    elif report.llm_adjustment_was_helpful == "no":
        out.append("⚠️ El ajuste LLM fue en sentido contrario al resultado real.")

    # 4. Ranking en el pool
    if report.our_best_rank_in_pool is not None and report.pool_top_points is not None:
        out.append(
            f"Mejor penca quedó en posición <b>{report.our_best_rank_in_pool}°</b> del pool "
            f"(top tiene {report.pool_top_points}pts)"
        )

    # 5. Acumulado del portfolio
    out.append(
        f"Portfolio (suma de las 5 pencas): {report.portfolio_total_points} pts. "
        f"Mejor: {report.portfolio_max_points}pts."
    )

    return out


def evaluate_llm_adjustment(
    d_l: float,
    d_v: float,
    post_l: float,
    post_v: float,
    actual: tuple[int, int],
) -> str | None:
    """¿El ajuste cualitativo (Capa 4) acercó los λ al resultado real?

    Los λ persistidos en `constraints` ya incluyen el delta del LLM — el pipeline
    pisa lam_L/lam_V con los ajustados antes de persistir — así que el base sin
    LLM se reconstruye como λ_post − δ (mismo criterio que llm_counterfactual()
    en src/dashboard/data_loader.py).

    Se evalúa por equipo: cada δ no trivial fue útil si dejó su λ más cerca de
    los goles reales de ese equipo. Devuelve "yes" si todos los ajustes ayudaron,
    "no" si todos empeoraron, "neutral" si fue mixto o sin efecto medible, y
    None si el LLM no movió nada.
    """
    if abs(d_l) + abs(d_v) <= 0.01:
        return None
    verdicts = []
    for delta, post, goals in ((d_l, post_l, actual[0]), (d_v, post_v, actual[1])):
        if abs(delta) <= 0.01:
            continue
        base = post - delta
        err_post, err_base = abs(post - goals), abs(base - goals)
        if err_post < err_base:
            verdicts.append("yes")
        elif err_post > err_base:
            verdicts.append("no")
        else:
            verdicts.append("neutral")
    decisive = {v for v in verdicts if v != "neutral"}
    if len(decisive) == 1:
        return decisive.pop()
    return "neutral" if verdicts else None


def compute_postmortem(match_id: str | int) -> PostmortemReport | None:
    """Computa el postmortem completo para un partido finalizado."""
    pred = _latest_prediction(match_id)
    if not pred:
        log.warning("No hay predicción para match %s", match_id)
        return None

    match_status = _fetch_match_status(match_id)
    if not match_status:
        log.warning("No pude consultar status del partido %s", match_id)
        return None

    actual_h = match_status.get("home_score")
    actual_a = match_status.get("away_score")
    if actual_h is None or actual_a is None:
        log.info("Match %s sin score todavía", match_id)
        return None

    actual = (int(actual_h), int(actual_a))

    home_team = (match_status.get("home_team") or {}).get("name", "?")
    away_team = (match_status.get("away_team") or {}).get("name", "?")

    report = PostmortemReport(
        match_id=match_id,
        home_team=home_team,
        away_team=away_team,
        actual_home=actual[0],
        actual_away=actual[1],
        final_score=f"{actual[0]}-{actual[1]}",
    )

    # Por cada penca, computar puntos
    assignment = pred.get("assignment") or []
    for a in assignment:
        score = tuple(a.get("score", [0, 0]))
        points = jmlm_points(score, actual)
        is_exact = score == actual
        # Ganador correcto
        winner_actual = "H" if actual[0] > actual[1] else ("D" if actual[0] == actual[1] else "A")
        winner_pred = "H" if score[0] > score[1] else ("D" if score[0] == score[1] else "A")
        report.pencas_results.append(PencaResult(
            penca_id=a.get("penca_id", 0),
            penca_name=None,
            strategy_used=a.get("objective", "?"),
            predicted_score=score,
            points_earned=points,
            is_exact=is_exact,
            correct_winner=(winner_actual == winner_pred),
        ))

    report.portfolio_max_points = max((r.points_earned for r in report.pencas_results), default=0)
    report.portfolio_total_points = sum(r.points_earned for r in report.pencas_results)

    # Identificar mejor penca (para tracking)
    if report.pencas_results:
        best = max(report.pencas_results, key=lambda p: p.points_earned)
        report.our_best_penca_id = best.penca_id

    # Comparar con pool
    lb = _fetch_full_leaderboard()
    if lb:
        # filtramos las pencas del usuario
        my_penca_ids = set(int(a.get("penca_id", 0)) for a in assignment)
        pool_only = [e for e in lb if int(e.get("penca_id", 0)) not in my_penca_ids]
        if pool_only:
            pool_top = pool_only[0].get("points_total", 0)
            report.pool_top_points = pool_top
            pool_scores = [e.get("points_total", 0) for e in pool_only]
            report.pool_median_points = float(sorted(pool_scores)[len(pool_scores) // 2])
            # Rank de nuestra mejor penca en el leaderboard global
            sorted_lb = sorted(lb, key=lambda e: -e.get("points_total", 0))
            for i, entry in enumerate(sorted_lb):
                if int(entry.get("penca_id", 0)) == report.our_best_penca_id:
                    report.our_best_rank_in_pool = i + 1
                    break

    # Diagnostics del mercado
    constraints = pred.get("constraints", {})
    if constraints:
        lam_L = constraints.get("lambda_L")
        lam_V = constraints.get("lambda_V")
        if lam_L and lam_V:
            # Modal del mercado = scoreline más probable según el modelo
            from src.model.poisson import score_grid
            grid = score_grid(lam_L, lam_V, constraints.get("lambda_12", 0.0))
            import numpy as np
            modal_idx = int(np.argmax(grid))
            modal_pick = (modal_idx // grid.shape[0], modal_idx % grid.shape[0])
            report.market_modal_score = modal_pick
            report.market_modal_was_right = (modal_pick == actual)

    # LLM ajuste evaluation
    qa = pred.get("qualitative_adjustment")
    if qa:
        report.llm_adjustment_applied = qa
        d_l = qa.get("delta_lambda_L", 0)
        d_v = qa.get("delta_lambda_V", 0)
        report.llm_adjustment_was_helpful = evaluate_llm_adjustment(
            d_l, d_v,
            constraints.get("lambda_L", 1.5), constraints.get("lambda_V", 1.0),
            actual,
        )

    # Generar insights
    report.insights = _generate_insights(report)
    return report


def save_postmortem(report: PostmortemReport) -> Path:
    path = postmortem_path(report.match_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False, default=str))
    return path
