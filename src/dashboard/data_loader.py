"""Carga la data del dashboard desde los JSON del pipeline."""

from __future__ import annotations

import json
import os
import glob
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.utils.versions import latest_version, sort_versions


def _data_dir() -> Path:
    from src.utils.env import get_str
    return Path(get_str("DATA_DIR", "data"))


# Cache simple in-memory con TTL para evitar golpear APIs externas en cada request.
_CACHE: dict[str, tuple[float, Any]] = {}


def _cached(key: str, ttl_seconds: float, loader_fn):
    """Memoize loader_fn por key con TTL. Thread-safe naive (FastAPI workers single-threaded por default)."""
    import time
    now = time.time()
    entry = _CACHE.get(key)
    if entry and (now - entry[0]) < ttl_seconds:
        return entry[1]
    value = loader_fn()
    _CACHE[key] = (now, value)
    return value


UY_TZ = timezone(timedelta(hours=-3))


def build_penca_labels() -> dict[int, str]:
    """Devuelve {penca_id: 'short label'} usando el orden de PENCA_IDS en .env.

    PENCA_IDS=1651,1652,1653,1654,1655 → {1651: '1', 1652: '2', ..., 1655: '5'}
    """
    from src.utils.env import get_int_list
    ids = get_int_list("PENCA_IDS")
    return {pid: str(i + 1) for i, pid in enumerate(ids)}


def _to_uy(iso: str) -> str:
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return (dt - timedelta(hours=3)).strftime("%a %d/%m %H:%M UY")
    except Exception:
        return iso


# ---------- próximo partido ----------

def load_next_match_data() -> dict | None:
    """Devuelve la data del próximo partido a jugarse, con su última predicción y dossier."""
    import yaml
    try:
        fixtures_path = Path(__file__).resolve().parents[2] / "config" / "fixtures.yaml"
        fixtures = yaml.safe_load(fixtures_path.read_text()) or {}
    except Exception:
        return None

    all_matches = (fixtures.get("fase_grupos") or []) + (fixtures.get("eliminatorias") or [])
    now = datetime.now(timezone.utc)
    upcoming = []
    for m in all_matches:
        try:
            ko = datetime.fromisoformat(m["kickoff_utc"].replace("Z", "+00:00"))
            if ko > now and (m.get("home_name") or m.get("home")):
                upcoming.append((ko, m))
        except Exception:
            continue
    if not upcoming:
        return None
    upcoming.sort(key=lambda x: x[0])
    ko, match = upcoming[0]

    delta = ko - now
    days, rem = delta.days, delta.seconds
    hours = rem // 3600
    minutes = (rem % 3600) // 60
    if days > 0:
        countdown = f"{days}d {hours}h"
    elif hours > 0:
        countdown = f"{hours}h {minutes}m"
    else:
        countdown = f"{minutes}m"

    out: dict[str, Any] = {
        "match_id": match["id"],
        "home": match.get("home_name") or match.get("home", "?"),
        "away": match.get("away_name") or match.get("away", "?"),
        "kickoff_uy": _to_uy(match["kickoff_utc"]),
        "countdown": countdown,
        "stage": match.get("stage", "?"),
        "group": match.get("group"),
        "venue": match.get("venue"),
    }

    # Última predicción
    pred = _load_latest_prediction(match["id"])
    if pred:
        out["model"] = {
            "p_home": pred["constraints"]["p_home"],
            "p_draw": pred["constraints"]["p_draw"],
            "p_away": pred["constraints"]["p_away"],
            "e_goals_L": pred["constraints"]["e_goals_L"],
            "e_goals_V": pred["constraints"]["e_goals_V"],
        }
        out["assignment"] = pred.get("assignment") or []
        out["assignment_meta"] = pred.get("assignment_meta") or {}
        out["qualitative"] = pred.get("qualitative_adjustment") or {}

    # Dossier
    dossier = _load_latest_dossier(match["id"])
    if dossier:
        out["dossier"] = dossier

    return out


def _load_latest_prediction(match_id: Any) -> dict | None:
    pdir = _data_dir() / "predictions" / str(match_id)
    if not pdir.exists():
        return None
    latest = latest_version(pdir.glob("v*_*.json"))
    if latest is None:
        return None
    return json.loads(latest.read_text())


def _load_latest_dossier(match_id: Any) -> dict | None:
    ddir = _data_dir() / "dossiers" / str(match_id)
    if not ddir.exists():
        return None
    latest = latest_version(ddir.glob("v*.json"))
    if latest is None:
        return None
    return json.loads(latest.read_text())


# ---------- matches agrupados por día ----------

def load_matches_by_day(days_back: int = 2, days_ahead: int = 21) -> list[dict]:
    """Devuelve lista de días con sus partidos. Cada partido tiene picks/resultado si aplica.

    Returns: [{"date_uy": "Jue 11/06", "iso_date": "2026-06-11", "is_today": bool, "matches": [...]}, ...]
    """
    import yaml
    try:
        fixtures_path = Path(__file__).resolve().parents[2] / "config" / "fixtures.yaml"
        fixtures = yaml.safe_load(fixtures_path.read_text()) or {}
    except Exception:
        return []

    all_matches = (fixtures.get("fase_grupos") or []) + (fixtures.get("eliminatorias") or [])
    now = datetime.now(timezone.utc)
    cutoff_back = now - timedelta(days=days_back)
    cutoff_ahead = now + timedelta(days=days_ahead)

    by_day: dict[str, dict] = {}
    for m in all_matches:
        if not m.get("kickoff_utc"):
            continue
        try:
            ko = datetime.fromisoformat(m["kickoff_utc"].replace("Z", "+00:00"))
        except Exception:
            continue
        if ko < cutoff_back or ko > cutoff_ahead:
            continue
        ko_uy = ko - timedelta(hours=3)
        iso_date = ko_uy.strftime("%Y-%m-%d")
        if iso_date not in by_day:
            by_day[iso_date] = {
                "iso_date": iso_date,
                "date_uy": ko_uy.strftime("%a %d/%m"),
                "is_today": iso_date == (now - timedelta(hours=3)).strftime("%Y-%m-%d"),
                "matches": [],
            }
        # Status
        if ko > now + timedelta(minutes=2):
            status = "upcoming"
        elif ko + timedelta(minutes=150) > now:
            status = "live"
        else:
            status = "finished"

        # Cargar predicción más reciente (si hay)
        pred = _load_latest_prediction(m["id"])
        assignment = pred.get("assignment") if pred else None

        # Postmortem (si terminó)
        pm = None
        if status == "finished":
            pm_path = _data_dir() / "postmortems" / f"{m['id']}.json"
            if pm_path.exists():
                try:
                    pm = json.loads(pm_path.read_text())
                except Exception:
                    pm = None

        by_day[iso_date]["matches"].append({
            "match_id": m["id"],
            "home": m.get("home_name") or m.get("home", "?"),
            "away": m.get("away_name") or m.get("away", "?"),
            "kickoff_utc": m["kickoff_utc"],
            "kickoff_uy_time": ko_uy.strftime("%H:%M"),
            "status": status,
            "group": m.get("group"),
            "stage": m.get("stage", "?"),
            "venue": m.get("venue"),
            "assignment": assignment,
            "postmortem": {
                "final_score": pm.get("final_score"),
                "portfolio_max_points": pm.get("portfolio_max_points"),
                "portfolio_total_points": pm.get("portfolio_total_points"),
            } if pm else None,
        })

    # Ordenar por iso_date y dentro de cada día por kickoff
    days_sorted = sorted(by_day.values(), key=lambda d: d["iso_date"])
    for d in days_sorted:
        d["matches"].sort(key=lambda m: m["kickoff_uy_time"])
    return days_sorted


# ---------- match detail con timeline de pasadas ----------

STRATEGY_RATIONALE: dict[str, str] = {
    "ev": "EV puro: marcador con MAYOR esperanza de puntos sobre toda la grilla Poisson.",
    "differentiated": "Diferencial: alto EV PERO castigando popularidad (el marcador modal del pool resta valor).",
    "tail": "Goleada: maximiza puntos esperados condicionado al 10% de outcomes con MÁS goles.",
    "upset": "Sorpresa: argmax E[points] forzando ganador opuesto al favorito de mercado.",
    "variance": "Varianza: entre top-K por EV no usadas, la de mayor desvío estándar (alta varianza).",
}


def _assignment_reason(
    penca_rank: int | None,
    objective: str,
    assignment_meta: dict,
    total_pencas: int | None = None,
) -> str:
    """Por qué a esta penca le tocó esta estrategia en esta pasada."""
    obj_label = {
        "ev": "Favorito 🎯",
        "differentiated": "Diferencial 📊",
        "tail": "Goleada ⚡",
        "upset": "Sorpresa 😲",
        "variance": "Varianza 📈",
    }.get(objective, objective)

    if not penca_rank:
        return f"Estrategia: {obj_label} (sin ranking aún)."

    pos = f"rank {penca_rank}" + (f" de {total_pencas}" if total_pencas else "")
    base_obj = (assignment_meta or {}).get("objective", "")
    goal = "maximizar P(top-3)" if base_obj.startswith("p_top_k") else "maximizar E[máximo del portfolio]"

    if penca_rank == 1:
        return (
            f"Penca {pos} (puntera). Recibe {obj_label}: el optimizer voraz ({goal}) le da "
            "una pick sólida para preservar la ventaja."
        )
    if total_pencas and penca_rank == total_pencas:
        return (
            f"Penca {pos} (última). Recibe {obj_label}: el optimizer la usa para cubrir un "
            "escenario que las demás pencas no cubren, buscando que alguna llegue al cutoff."
        )
    return (
        f"Penca {pos}. Recibe {obj_label}: es la pick que más aporta al objetivo global "
        f"({goal}) dado lo que ya cubren las otras pencas."
    )


def _strategy_rationale_for_pick(objective: str, pick_metrics: dict | None) -> str:
    base = STRATEGY_RATIONALE.get(objective, objective)
    if not pick_metrics:
        return base
    e_pts = pick_metrics.get("e_points")
    pop = pick_metrics.get("pool_popularity")
    p_score = pick_metrics.get("p_scoreline")
    extras = []
    if e_pts is not None:
        extras.append(f"E[pts]={e_pts:.2f}")
    if p_score is not None:
        extras.append(f"P(marcador)={p_score*100:.1f}%")
    if pop is not None:
        extras.append(f"pool pop={pop*100:.1f}%")
    if extras:
        base += "  ·  " + " · ".join(extras)
    return base


def load_match_detail(match_id) -> dict | None:
    """Carga todas las versiones de predicción del partido + diff entre pasadas."""
    import yaml
    try:
        fixtures_path = Path(__file__).resolve().parents[2] / "config" / "fixtures.yaml"
        fixtures = yaml.safe_load(fixtures_path.read_text()) or {}
    except Exception:
        fixtures = {}

    all_matches = (fixtures.get("fase_grupos") or []) + (fixtures.get("eliminatorias") or [])
    match_meta = next((m for m in all_matches if str(m.get("id")) == str(match_id)), None)
    if not match_meta:
        return None

    pdir = _data_dir() / "predictions" / str(match_id)
    if not pdir.exists():
        return {
            "match_id": match_id,
            "home": match_meta.get("home_name") or match_meta.get("home", "?"),
            "away": match_meta.get("away_name") or match_meta.get("away", "?"),
            "kickoff_uy": _to_uy(match_meta["kickoff_utc"]),
            "venue": match_meta.get("venue"),
            "group": match_meta.get("group"),
            "stage": match_meta.get("stage", "?"),
            "versions": [],
            "current_pencas": [],
            "diffs": [],
        }

    files = sort_versions(pdir.glob("v*_*.json"))
    versions = []
    for f in files:
        try:
            v = json.loads(f.read_text())
            versions.append(v)
        except Exception:
            continue

    if not versions:
        return None

    latest = versions[-1]
    # Construir lista de pencas con razón por estrategia.
    # Con N pencas hay objetivos repetidos (ev×k, alt×k) → mapeamos métricas por MARCADOR
    # (que identifica único la pick), con fallback por objetivo.
    menu = (latest.get("portfolio") or {}).get("picks", [])
    portfolio_by_obj = {p["objective"]: p for p in menu}
    portfolio_by_score = {
        f'{p["score"][0]}-{p["score"][1]}': p for p in menu
    }
    latest_assignment = latest.get("assignment") or []
    total_pencas = len(latest_assignment)
    latest_meta = latest.get("assignment_meta") or {}

    current_pencas = []
    for a in latest_assignment:
        obj = a["objective"]
        score_key = f'{a["score"][0]}-{a["score"][1]}'
        pick_metrics = portfolio_by_score.get(score_key) or portfolio_by_obj.get(obj, {})
        current_pencas.append({
            "penca_id": a["penca_id"],
            "rank": a.get("rank"),
            "objective": obj,
            "score": a["score"],
            "p_scoreline": pick_metrics.get("p_scoreline"),
            "e_points": pick_metrics.get("e_points"),
            "pool_popularity": pick_metrics.get("pool_popularity"),
            "assignment_reason": _assignment_reason(a.get("rank"), obj, latest_meta, total_pencas),
            "strategy_rationale": _strategy_rationale_for_pick(obj, pick_metrics),
        })

    # Exposición agregada por marcador (cuántas pencas en cada scoreline), orden por frecuencia.
    from collections import Counter
    exposure_counter = Counter(
        f'{a["score"][0]}-{a["score"][1]}' for a in latest_assignment
    )
    exposure = [
        {"score": s, "count": c}
        for s, c in sorted(exposure_counter.items(), key=lambda kv: (-kv[1], kv[0]))
    ]

    # Resumen por versión (timeline)
    timeline = []
    for v in versions:
        ts = v.get("run_at", "")
        timeline.append({
            "version": v.get("version"),
            "phase": v.get("phase"),
            "run_at_uy": _to_uy(ts) if ts else "",
            "p_home": v.get("constraints", {}).get("p_home"),
            "p_draw": v.get("constraints", {}).get("p_draw"),
            "p_away": v.get("constraints", {}).get("p_away"),
            "lambda_L": v.get("constraints", {}).get("lambda_L"),
            "lambda_V": v.get("constraints", {}).get("lambda_V"),
            "assignment": v.get("assignment") or [],
            "qualitative": v.get("qualitative_adjustment") or {},
            "assignment_meta": v.get("assignment_meta") or {},
        })

    # Diffs entre versiones consecutivas
    diffs = []
    for i in range(1, len(timeline)):
        prev, cur = timeline[i - 1], timeline[i]
        change_lines = []
        # Cambios por penca
        prev_by_pid = {a["penca_id"]: a for a in prev["assignment"]}
        cur_by_pid = {a["penca_id"]: a for a in cur["assignment"]}
        for pid, a in cur_by_pid.items():
            pa = prev_by_pid.get(pid)
            if pa:
                if a["score"] != pa["score"] or a["objective"] != pa["objective"]:
                    change_lines.append({
                        "penca_id": pid,
                        "old": f"{pa['objective']} {pa['score'][0]}-{pa['score'][1]}",
                        "new": f"{a['objective']} {a['score'][0]}-{a['score'][1]}",
                    })
        # Cambios en probabilidades
        def _delta(a, b, label, threshold=0.02):
            if a is None or b is None:
                return None
            d = b - a
            if abs(d) < threshold:
                return None
            return f"{label}: {a*100:.0f}% → {b*100:.0f}% ({d*100:+.1f}pp)"

        ph = _delta(prev["p_home"], cur["p_home"], "P(local)")
        pd = _delta(prev["p_draw"], cur["p_draw"], "P(empate)")
        pa = _delta(prev["p_away"], cur["p_away"], "P(visit)")

        # Cambios en LLM
        llm_change = None
        prev_dL = prev["qualitative"].get("delta_lambda_L", 0)
        cur_dL = cur["qualitative"].get("delta_lambda_L", 0)
        prev_dV = prev["qualitative"].get("delta_lambda_V", 0)
        cur_dV = cur["qualitative"].get("delta_lambda_V", 0)
        if abs(cur_dL - prev_dL) > 0.05 or abs(cur_dV - prev_dV) > 0.05:
            llm_change = (
                f"LLM ajuste: ΔλL {prev_dL:+.2f}→{cur_dL:+.2f}, "
                f"ΔλV {prev_dV:+.2f}→{cur_dV:+.2f}"
            )

        diffs.append({
            "from_phase": prev["phase"],
            "to_phase": cur["phase"],
            "from_version": prev["version"],
            "to_version": cur["version"],
            "from_run_at": prev["run_at_uy"],
            "to_run_at": cur["run_at_uy"],
            "pick_changes": change_lines,
            "market_changes": [x for x in (ph, pd, pa) if x],
            "llm_change": llm_change,
            "cur_llm_reasoning": cur["qualitative"].get("reasoning"),
        })

    return {
        "match_id": match_id,
        "home": match_meta.get("home_name") or match_meta.get("home", "?"),
        "away": match_meta.get("away_name") or match_meta.get("away", "?"),
        "kickoff_uy": _to_uy(match_meta["kickoff_utc"]),
        "venue": match_meta.get("venue"),
        "group": match_meta.get("group"),
        "stage": match_meta.get("stage", "?"),
        "latest_constraints": latest.get("constraints", {}),
        "latest_meta": latest_meta,
        "latest_qualitative": latest.get("qualitative_adjustment") or {},
        "current_pencas": current_pencas,
        "total_pencas": total_pencas,
        "exposure": exposure,
        "timeline": timeline,
        "diffs": diffs,
        "dossier": _load_latest_dossier(match_id),
        "llm_impact": llm_counterfactual(latest),
    }


# ---------- pencas ranking vs pool ----------

def load_my_pencas_standings() -> dict[str, Any]:
    """Wrapper cached (30s TTL) sobre el leaderboard real."""
    return _cached("standings", 30.0, _load_my_pencas_standings_uncached)


def _load_my_pencas_standings_uncached() -> dict[str, Any]:
    """Lee leaderboard real de la penca. Filtra mis pencas."""
    import httpx
    base = os.environ.get("PENCA_API_BASE_URL", "").rstrip("/")
    key = os.environ.get("PENCA_API_KEY", "")
    from src.utils.env import get_int_list
    my_ids = set(get_int_list("PENCA_IDS"))
    if not base or not key or not my_ids:
        return {"error": "API o PENCA_IDS no configurados", "pencas": []}
    try:
        with httpx.Client(timeout=8.0, headers={"Authorization": f"Bearer {key}"}) as c:
            r = c.get(f"{base}/leaderboard")
        if r.status_code != 200:
            return {"error": f"leaderboard {r.status_code}", "pencas": []}
        entries = r.json().get("entries", [])
    except Exception as e:
        return {"error": str(e), "pencas": []}

    sorted_entries = sorted(entries, key=lambda e: -e.get("points_total", 0))
    total_in_pool = len(sorted_entries)

    my_entries = []
    for i, e in enumerate(sorted_entries):
        pid = int(e.get("penca_id", 0))
        if pid in my_ids:
            raw_name = e.get("penca_name") or ""
            # Compactar: "Penca 1" → "1", "Penca Joaco 3" → "Joaco 3", fallback al pid
            short_name = raw_name
            if raw_name.lower().startswith("penca "):
                short_name = raw_name.split(" ", 1)[1].strip()
            if not short_name:
                short_name = str(pid)
            my_entries.append({
                "penca_id": pid,
                "penca_name": short_name,
                "rank": i + 1,
                "points": e.get("points_total", 0),
                "exact_scores": e.get("exact_scores", 0),
                "correct_winners": e.get("correct_winners", 0),
                "predictions_made": e.get("predictions_made", 0),
            })

    # Para cada penca, agregar la estrategia ASIGNADA en el próximo partido (si hay)
    next_match = load_next_match_data()
    strategy_by_pid: dict[int, str] = {}
    if next_match:
        for a in next_match.get("assignment", []):
            strategy_by_pid[int(a["penca_id"])] = a["objective"]
    for e in my_entries:
        e["next_strategy"] = strategy_by_pid.get(e["penca_id"])

    return {
        "pool_size": total_in_pool,
        "pencas": my_entries,
        "pool_top": sorted_entries[0] if sorted_entries else None,
        "pool_median_points": (
            sorted_entries[len(sorted_entries) // 2].get("points_total", 0)
            if sorted_entries else 0
        ),
    }


# ---------- detalle por penca ----------

def llm_counterfactual(pred: dict) -> dict:
    """¿El ajuste cualitativo (Capa 4) cambió la predicción?

    Reconstruye los λ SIN LLM (λ_post − δ) y regenera el menú de 5 objetivos en los dos
    escenarios. Devuelve, por objetivo, el pick sin/ con LLM y si cambió, más el shift de
    probabilidades 1X2. Read-only: no toca el pipeline, solo compara lo ya persistido.
    """
    qa = pred.get("qualitative_adjustment") or {}
    c = pred.get("constraints") or {}
    dL = float(qa.get("delta_lambda_L", 0.0) or 0.0)
    dV = float(qa.get("delta_lambda_V", 0.0) or 0.0)
    out = {
        "ran": bool(qa),
        "delta_lambda_L": dL,
        "delta_lambda_V": dV,
        "confidence": qa.get("confidence"),
        "reasoning": qa.get("reasoning"),
        "moved_lambda": (abs(dL) > 1e-6 or abs(dV) > 1e-6),
        "rows": [],
        "n_changed": 0,
        "probs": None,
    }
    post_L, post_V = c.get("lambda_L"), c.get("lambda_V")
    l12 = c.get("lambda_12", 0.1)
    if post_L is None or post_V is None or not out["moved_lambda"]:
        return out
    try:
        import numpy as np
        from src.model.poisson import score_grid, marginals
        from src.strategy.portfolio import generate_portfolio
        from src.meta.calibration import get_pool_config

        pre_L = max(0.1, post_L - dL)
        pre_V = max(0.1, post_V - dV)
        p_home = c.get("p_home")
        p_away = c.get("p_away")
        grid_pre = score_grid(pre_L, pre_V, l12, max_goals=7)
        grid_post = score_grid(post_L, post_V, l12, max_goals=7)
        pool_cfg = get_pool_config()
        port_pre = generate_portfolio(grid_pre, p_home, p_away, pool_cfg)
        port_post = generate_portfolio(grid_post, p_home, p_away, pool_cfg)

        label = {"ev": "Favorito", "differentiated": "Diferencial", "tail": "Goleada",
                 "upset": "Sorpresa", "variance": "Varianza"}
        rows, n_changed = [], 0
        for a, b in zip(port_pre.picks, port_post.picks):
            pre = (a.score_local, a.score_visit)
            post = (b.score_local, b.score_visit)
            changed = pre != post
            n_changed += int(changed)
            rows.append({
                "objective": a.objective,
                "label": label.get(a.objective, a.objective),
                "pre": list(pre), "post": list(post), "changed": changed,
            })

        def _winp(grid):
            m = marginals(grid)
            return {"p_home": float(m.p_home_win), "p_draw": float(m.p_draw), "p_away": float(m.p_away_win)}

        out["rows"] = rows
        out["n_changed"] = n_changed
        out["pre_lambda"] = [round(pre_L, 2), round(pre_V, 2)]
        out["post_lambda"] = [round(post_L, 2), round(post_V, 2)]
        out["probs"] = {"pre": _winp(grid_pre), "post": _winp(grid_post)}
    except Exception:
        pass
    return out


def _pick_metrics(constraints: dict, score) -> dict:
    """Recomputa métricas del pick desde los λ persistidos: E[pts], P(marcador), popularidad,
    y si es el marcador modal (más probable)."""
    if score is None or score[0] is None:
        return {}
    ll, lv = constraints.get("lambda_L"), constraints.get("lambda_V")
    l12 = constraints.get("lambda_12", 0.1)
    if ll is None or lv is None:
        return {}
    try:
        import numpy as np
        from src.model.poisson import score_grid, jmlm_points
        from src.meta.pool import pool_pick_distribution
        from src.meta.calibration import get_pool_config
        grid = score_grid(ll, lv, l12, max_goals=7)
        n = grid.shape[0]
        gL, gV = int(score[0]), int(score[1])
        if gL >= n or gV >= n:
            return {}
        pool_q = pool_pick_distribution(grid, get_pool_config())
        e_pts = float(sum(grid[i, j] * jmlm_points((gL, gV), (i, j)) for i in range(n) for j in range(n)))
        modal = np.unravel_index(int(np.argmax(grid)), grid.shape)
        return {
            "p_scoreline": float(grid[gL, gV]),
            "e_points": e_pts,
            "pool_popularity": float(pool_q[gL, gV]),
            "is_modal": (gL, gV) == (int(modal[0]), int(modal[1])),
        }
    except Exception:
        return {}


def load_penca_detail(penca_id) -> dict:
    """Detalle de UNA penca: standing + picks con métricas + resumen de rol + diferenciación
    + evolución (puntos por jornada cuando hay resultados)."""
    import yaml
    pid = int(penca_id)

    standings = load_my_pencas_standings()
    me = next((p for p in (standings.get("pencas") or []) if int(p["penca_id"]) == pid), None)

    try:
        fixtures = yaml.safe_load(
            (Path(__file__).resolve().parents[2] / "config" / "fixtures.yaml").read_text()
        ) or {}
    except Exception:
        fixtures = {}
    all_matches = (fixtures.get("fase_grupos") or []) + (fixtures.get("eliminatorias") or [])
    meta = {str(m.get("id")): m for m in all_matches}

    picks = []
    pred_dir = _data_dir() / "predictions"
    if pred_dir.exists():
        for mdir in sorted(pred_dir.iterdir()):
            if not mdir.is_dir() or mdir.name.startswith("_"):
                continue
            latest = latest_version(mdir.glob("v*_*.json"))
            if latest is None:
                continue
            try:
                data = json.loads(latest.read_text())
            except Exception:
                continue
            entry = next(
                (a for a in (data.get("assignment") or []) if int(a.get("penca_id", -1)) == pid),
                None,
            )
            if entry is None:
                continue
            m = meta.get(str(mdir.name), {})
            sc = entry.get("score") or [None, None]
            mx = _pick_metrics(data.get("constraints", {}), sc)
            obj = entry.get("objective")
            rank = entry.get("rank")
            total_pencas = len(data.get("assignment") or [])
            assignment_meta = data.get("assignment_meta") or {}
            points = None
            hs, as_ = m.get("home_score"), m.get("away_score")
            if hs is not None and as_ is not None and sc[0] is not None:
                from src.model.poisson import jmlm_points
                points = jmlm_points((int(sc[0]), int(sc[1])), (int(hs), int(as_)))
            picks.append({
                "match_id": mdir.name,
                "home": m.get("home_name") or m.get("home") or "?",
                "away": m.get("away_name") or m.get("away") or "?",
                "kickoff_uy": _to_uy(m["kickoff_utc"]) if m.get("kickoff_utc") else "",
                "kickoff_raw": m.get("kickoff_utc", ""),
                "score": sc,
                "objective": obj,
                "rank": rank,
                "phase": data.get("phase"),
                "actual": [hs, as_] if hs is not None and as_ is not None else None,
                "points": points,
                "assignment_reason": _assignment_reason(rank, obj, assignment_meta, total_pencas),
                "strategy_rationale": _strategy_rationale_for_pick(obj, mx),
                **mx,
            })
    picks.sort(key=lambda p: p.get("kickoff_raw") or "")

    # ----- Resumen / rol -----
    from collections import Counter
    metricked = [p for p in picks if p.get("p_scoreline") is not None]
    n = len(picks)
    n_modal = sum(1 for p in picks if p.get("is_modal"))
    deviations = [p for p in picks if p.get("p_scoreline") is not None and not p.get("is_modal")]
    chalk_frac = (n_modal / n) if n else 0
    summary = {
        "n_picks": n,
        "e_total": round(sum(p["e_points"] for p in metricked), 1) if metricked else None,
        "avg_popularity": round(sum(p["pool_popularity"] for p in metricked) / len(metricked), 3) if metricked else None,
        "n_modal": n_modal,
        "n_deviations": len(deviations),
        "lean": "chalk" if chalk_frac >= 0.6 else ("contrarian" if chalk_frac < 0.3 else "balanceada"),
        "strategies": dict(Counter(p["objective"] for p in picks)),
    }
    # Diferenciación: picks que se apartan del modal, ordenados por menor popularidad (más contrarian)
    differentiation = sorted(deviations, key=lambda p: p["pool_popularity"])

    # Evolución: solo partidos jugados (con puntos), acumulado
    played = [p for p in picks if p.get("points") is not None]
    evolution = []
    cum = 0
    for p in played:
        cum += p["points"]
        evolution.append({"home": p["home"], "away": p["away"], "points": p["points"], "cumulative": cum})

    return {
        "penca": me, "penca_id": pid, "picks": picks,
        "summary": summary, "differentiation": differentiation, "evolution": evolution,
    }


# ---------- postmortems ----------

def load_recent_postmortems(limit: int = 5) -> list[dict]:
    pdir = _data_dir() / "postmortems"
    if not pdir.exists():
        return []
    files = sorted(pdir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    out = []
    for f in files:
        try:
            out.append(json.loads(f.read_text()))
        except Exception:
            continue
    return out


# ---------- system health + costos ----------

def load_system_health() -> dict[str, Any]:
    """Wrapper cached (60s TTL): health no necesita ser super fresh."""
    return _cached("health", 60.0, _load_system_health_uncached)


def _load_system_health_uncached() -> dict[str, Any]:
    """Calcula health desde JSONL + APIs externas."""
    out: dict[str, Any] = {}
    # Conteo de predicciones
    pred_dir = _data_dir() / "predictions"
    if pred_dir.exists():
        all_files = list(pred_dir.rglob("v*_*.json"))
        out["predictions_total"] = len(all_files)
        cutoff = datetime.now().timestamp() - 24 * 3600
        out["predictions_24h"] = sum(1 for f in all_files if f.stat().st_mtime > cutoff)
    else:
        out["predictions_total"] = 0
        out["predictions_24h"] = 0

    # Anthropic usage
    usage_path = _data_dir() / "anthropic_usage.jsonl"
    cost_24h = 0.0
    cost_total = 0.0
    calls_total = 0
    if usage_path.exists():
        cutoff_ts = datetime.now(timezone.utc).timestamp() - 24 * 3600
        for line in usage_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            c = e.get("cost_usd", 0.0)
            cost_total += c
            calls_total += 1
            try:
                ts = datetime.fromisoformat(e["ts"]).timestamp()
                if ts > cutoff_ts:
                    cost_24h += c
            except Exception:
                pass
    out["anthropic_cost_24h"] = round(cost_24h, 4)
    out["anthropic_cost_total"] = round(cost_total, 4)
    out["anthropic_calls_total"] = calls_total

    # DO balance
    do_token = os.environ.get("DO_API_TOKEN", "")
    if do_token:
        try:
            import httpx
            with httpx.Client(timeout=5.0, headers={"Authorization": f"Bearer {do_token}"}) as c:
                r = c.get("https://api.digitalocean.com/v2/customers/my/balance")
            if r.status_code == 200:
                d = r.json()
                out["do_month_to_date"] = float(d.get("month_to_date_usage", 0))
        except Exception:
            out["do_month_to_date"] = None

    # Última ejecución del scheduler (best-effort)
    log_path = Path("/var/lib/penca/logs/scheduler.log")
    if not log_path.exists():
        log_path = _data_dir() / "logs" / "scheduler.log"
    if log_path.exists():
        try:
            stat = log_path.stat()
            last_modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
            out["scheduler_last_log_uy"] = (last_modified - timedelta(hours=3)).strftime("%H:%M UY")
        except Exception:
            pass

    out["dry_run"] = os.environ.get("DRY_RUN", "true").lower() == "true"
    return out
