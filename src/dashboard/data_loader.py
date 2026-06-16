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
        enriched = _attach_p_scoreline(pred)
        out["assignment"] = enriched
        out["score_pscore"] = {
            f'{a["score"][0]}-{a["score"][1]}': a.get("p_scoreline") for a in enriched
        }
        out["assignment_meta"] = pred.get("assignment_meta") or {}
        out["qualitative"] = pred.get("qualitative_adjustment") or {}

    # Dossier
    dossier = _load_latest_dossier(match["id"])
    if dossier:
        out["dossier"] = dossier

    return out


def load_live_match_data() -> dict | None:
    """Partido EN CURSO (status 'live'): nuestras picks + marcador en vivo (si la API lo da)
    + puntos provisionales (si terminara ahora). Cache corto: el marcador puede cambiar.
    """
    return _cached("live_match", 15.0, _load_live_match_data_uncached)


def _load_live_match_data_uncached() -> dict | None:
    import httpx
    base = os.environ.get("PENCA_API_BASE_URL", "").rstrip("/")
    key = os.environ.get("PENCA_API_KEY", "")
    if not base or not key:
        return None
    try:
        with httpx.Client(timeout=8.0, headers={"Authorization": f"Bearer {key}"}) as c:
            r = c.get(f"{base}/matches")
        if r.status_code != 200:
            return None
        matches = r.json().get("matches", [])
    except Exception:
        return None
    live = next((m for m in matches if m.get("status") == "live"), None)
    if not live:
        return None
    mid = live["id"]
    hs, as_ = live.get("home_score"), live.get("away_score")
    has_score = hs is not None and as_ is not None
    out: dict[str, Any] = {
        "match_id": mid,
        "home": (live.get("home_team") or {}).get("name", "?"),
        "away": (live.get("away_team") or {}).get("name", "?"),
        "group": live.get("group"),
        "stage": live.get("stage"),
        "kickoff_uy": _to_uy(live["kickoff_utc"]) if live.get("kickoff_utc") else None,
        "home_score": hs,
        "away_score": as_,
        "has_score": has_score,
        "picks": [],
        "n_scoring": 0,
        "best_prov": 0,
    }
    pred = _load_latest_prediction(mid)
    if pred:
        from src.model.poisson import jmlm_points
        from collections import Counter
        actual = (int(hs), int(as_)) if has_score else None
        # puntos ACUMULADOS de cada penca en el pool (su standing actual)
        standings = load_my_pencas_standings()
        acc = {int(e["penca_id"]): e.get("points") for e in (standings.get("pencas") or [])}
        for a in pred.get("assignment", []):
            sc = a.get("score", [0, 0])
            pid = a.get("penca_id")
            row = {"penca_id": pid, "objective": a.get("objective"), "score": sc,
                   "points": acc.get(int(pid)) if pid is not None else None}
            if actual is not None:
                row["prov_points"] = jmlm_points((int(sc[0]), int(sc[1])), actual)
            out["picks"].append(row)
        if actual is not None and out["picks"]:
            out["n_scoring"] = sum(1 for p in out["picks"] if p.get("prov_points", 0) > 0)
            out["best_prov"] = max((p.get("prov_points", 0) for p in out["picks"]), default=0)
        # ordenar por puntos ACUMULADOS desc (la que más tiene, primera; desempate por penca_id)
        out["picks"].sort(key=lambda p: (-(p.get("points") or 0), p.get("penca_id", 0)))
        # exposición por marcador (qué jugamos) + puntos provisionales por marcador
        exp = Counter(f'{a["score"][0]}-{a["score"][1]}' for a in pred.get("assignment", []))
        exposure = []
        for s, c in sorted(exp.items(), key=lambda kv: -kv[1]):
            row = {"score": s, "count": c}
            if actual is not None:
                gl, gv = s.split("-")
                row["prov_points"] = jmlm_points((int(gl), int(gv)), actual)
            exposure.append(row)
        # con marcador, ordenar por puntos provisionales (lo que más suma primero)
        if actual is not None:
            exposure.sort(key=lambda r: (-r.get("prov_points", 0), -r["count"]))
        out["exposure"] = exposure
        out["total_pencas"] = len(pred.get("assignment", []))
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


def _attach_p_scoreline(pred: dict | None) -> list[dict]:
    """Enriquece cada pick de la asignación con P(marcador exacto).

    Toma p_scoreline del menú del portfolio cuando el marcador está ahí; si no
    (picks 'alt' fuera del menú), lo recomputa desde los λ persistidos.
    """
    if not pred:
        return []
    assignment = pred.get("assignment") or []
    menu = (pred.get("portfolio") or {}).get("picks", [])
    by_score = {f'{p["score"][0]}-{p["score"][1]}': p for p in menu}
    constraints = pred.get("constraints") or {}
    out = []
    for a in assignment:
        score = a.get("score") or [None, None]
        sk = f'{score[0]}-{score[1]}'
        pm = by_score.get(sk) or _pick_metrics(constraints, score)
        out.append({**a, "p_scoreline": pm.get("p_scoreline")})
    return out


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
        assignment = _attach_p_scoreline(pred) if pred else None

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
    "tail": "Goles: optimiza el pick para escenarios de muchos goles (top 10% de outcomes por goles totales). El marcador resultante puede ser corto.",
    "upset": "Sorpresa: argmax E[points] forzando ganador opuesto al favorito de mercado.",
    "variance": "Varianza: entre top-K por EV no usadas, la de mayor desvío estándar (alta varianza).",
    "alt": "Alternativa: siguiente mejor marcador por EV no cubierto por las otras planillas — amplía la cobertura del abanico.",
}

# Goles totales mínimos para que una pick merezca el rótulo "Goles" (objetivo tail).
_GOLEADA_MIN_GOLES = 3


def _display_objective(objective: str, score) -> str:
    """Rótulo de DISPLAY (solo tablero). El objetivo 'tail' (Goles) a veces elige un
    marcador corto (su pick óptimo no siempre es goleador); en ese caso mostrarlo como
    "Goles" engaña, así que lo rotulamos como alternativa de cobertura.

    NO cambia la pick, ni la asignación, ni el modelo — solo cómo se nombra en la UI.
    """
    if (
        objective == "tail"
        and score
        and score[0] is not None
        and (int(score[0]) + int(score[1])) < _GOLEADA_MIN_GOLES
    ):
        return "alt"
    return objective


# Por qué cada ROL de pick cubre lo que cubre — en criollo, para la tarjeta por penca.
ROLE_WHY = {
    "ev": "le toca el marcador más probable del partido: juega a lo seguro.",
    "differentiated": "recibe un marcador casi tan bueno como el favorito pero que juega menos gente del pool — si pega, te despega del montón.",
    "tail": "cubre el escenario de un partido con muchos goles.",
    "upset": "es tu seguro contra el batacazo: la única que apuesta a que el favorito NO gane.",
    "variance": "recibe una apuesta de techo alto (mucha varianza): rinde poco en promedio, pero mucho si pega justo.",
    "alt": "cubre otro resultado plausible que las planillas mejor ubicadas no agarraron, para ampliar el abanico.",
}


def _reinforcement_flags(assignment: list[dict]) -> dict:
    """Por penca_id: True si su marcador ya lo tomó una planilla mejor ubicada (es un
    'refuerzo', no una cobertura nueva)."""
    seen: set = set()
    flags: dict = {}
    for a in sorted(assignment, key=lambda x: (x.get("rank") is None, x.get("rank") or 0)):
        sk = tuple(a.get("score") or [])
        flags[a.get("penca_id")] = sk in seen
        seen.add(sk)
    return flags


def _assignment_reason(
    penca_rank: int | None,
    objective: str,
    total_pencas: int | None = None,
    is_reinforcement: bool = False,
    score=None,
) -> str:
    """Por qué a ESTA planilla le tocó ESTE marcador — en lenguaje sencillo.

    Combina dónde está parada en la tabla con qué cubre el marcador.
    """
    if not penca_rank:
        return "Todavía sin posición en la tabla: recibe una pick de cobertura."

    if penca_rank == 1:
        pos = "Es tu planilla mejor ubicada (1ª)."
    elif total_pencas and penca_rank == total_pencas:
        pos = f"Es tu planilla más rezagada ({penca_rank}ª de {total_pencas})."
    elif total_pencas:
        pos = f"Va {penca_rank}ª de {total_pencas} entre tus planillas."
    else:
        pos = f"Va {penca_rank}ª entre tus planillas."

    if is_reinforcement:
        sc = f"{score[0]}-{score[1]}" if score and score[0] is not None else "uno de los más fuertes"
        why = (f"Como ya están cubiertos todos los resultados razonables, refuerza el {sc} "
               "(uno de los marcadores más probables) en lugar de gastarse en uno improbable.")
    else:
        why = ROLE_WHY.get(objective, "cubre un escenario que las otras planillas no cubren.")
        why = why[0].upper() + why[1:]

    return f"{pos} {why}"


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
    latest_constraints = latest.get("constraints") or {}

    reinf_flags = _reinforcement_flags(latest_assignment)
    current_pencas = []
    for a in latest_assignment:
        obj = _display_objective(a["objective"], a["score"])
        score_key = f'{a["score"][0]}-{a["score"][1]}'
        # Las picks "alt" (y cualquier marcador fuera del portfolio de 5) no están en el menú
        # nombrado → recomputamos sus métricas desde los λ para que la tarjeta no quede vacía.
        pick_metrics = (
            portfolio_by_score.get(score_key)
            or portfolio_by_obj.get(obj)
            or _pick_metrics(latest_constraints, a["score"])
        )
        is_reinf = reinf_flags.get(a["penca_id"], False)
        current_pencas.append({
            "penca_id": a["penca_id"],
            "rank": a.get("rank"),
            "objective": obj,
            "score": a["score"],
            "p_scoreline": pick_metrics.get("p_scoreline"),
            "e_points": pick_metrics.get("e_points"),
            "pool_popularity": pick_metrics.get("pool_popularity"),
            "assignment_reason": _assignment_reason(a.get("rank"), obj, total_pencas, is_reinf, a["score"]),
            "strategy_rationale": _strategy_rationale_for_pick(obj, pick_metrics),
        })

    # Enriquecer con el standing REAL en el pool (puntaje + posición global entre 150+),
    # no solo el rank interno entre nuestras 15.
    standings = load_my_pencas_standings()
    pool_size = standings.get("pool_size")
    standing_by_pid = {int(p["penca_id"]): p for p in (standings.get("pencas") or [])}
    for p in current_pencas:
        st = standing_by_pid.get(int(p["penca_id"]))
        p["pool_rank"] = st.get("rank") if st else None
        p["pool_points"] = st.get("points") if st else None

    # Ordenar por posición en el pool (mejor primero); las sin standing, al final.
    current_pencas.sort(key=lambda p: (p["pool_rank"] is None, p["pool_rank"] or 0))

    # Exposición agregada por marcador (cuántas pencas en cada scoreline), orden por frecuencia.
    from collections import Counter
    exposure_counter = Counter(
        f'{a["score"][0]}-{a["score"][1]}' for a in latest_assignment
    )
    pscore_by_score = {
        f'{p["score"][0]}-{p["score"][1]}': p.get("p_scoreline") for p in current_pencas
    }
    exposure = [
        {"score": s, "count": c, "p_scoreline": pscore_by_score.get(s)}
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
        "pool_size": pool_size,
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


def _last_match_points(my_ids: set) -> dict:
    """Puntos que sacó cada penca nuestra en el ÚLTIMO partido jugado.

    Toma el postmortem más reciente (por generated_at) y devuelve la info del partido
    + un mapa penca_id → {points, predicted, is_exact, correct_winner}. {} si no hay.
    """
    pmdir = _data_dir() / "postmortems"
    if not pmdir.exists():
        return {}
    latest = None
    for f in pmdir.glob("*.json"):
        if ".bak" in f.name:
            continue
        try:
            pm = json.loads(f.read_text())
        except Exception:
            continue
        if pm.get("actual_home") is None:
            continue
        ts = pm.get("generated_at", "")
        if latest is None or ts > latest[0]:
            latest = (ts, pm)
    if latest is None:
        return {}
    pm = latest[1]
    by_pid = {}
    for r in pm.get("pencas_results", []):
        pid = int(r.get("penca_id", 0))
        if pid not in my_ids:
            continue
        ps = r.get("predicted_score") or [None, None]
        by_pid[pid] = {
            "points": r.get("points_earned", 0),
            "predicted": f"{ps[0]}-{ps[1]}" if ps[0] is not None else None,
            "is_exact": bool(r.get("is_exact")),
            "correct_winner": bool(r.get("correct_winner")),
        }
    return {
        "home": pm.get("home_team", "?"),
        "away": pm.get("away_team", "?"),
        "final_score": pm.get("final_score") or f'{pm.get("actual_home")}-{pm.get("actual_away")}',
        "by_pid": by_pid,
    }


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

    # Puntos del último partido jugado, por penca
    last = _last_match_points(my_ids)
    last_by_pid = last.get("by_pid", {})
    for e in my_entries:
        e["next_strategy"] = strategy_by_pid.get(e["penca_id"])
        e["last_match"] = last_by_pid.get(e["penca_id"])

    return {
        "pool_size": total_in_pool,
        "pencas": my_entries,
        "pool_top": sorted_entries[0] if sorted_entries else None,
        "pool_median_points": (
            sorted_entries[len(sorted_entries) // 2].get("points_total", 0)
            if sorted_entries else 0
        ),
        "last_match": ({"home": last["home"], "away": last["away"],
                        "final_score": last["final_score"]} if last else None),
    }


# ---------- inteligencia del pool ----------

def _rank_key(e: dict) -> tuple:
    """Orden del torneo: más puntos, más exactos, más ganadores (desc)."""
    return (-e.get("points_total", 0), -e.get("exact_scores", 0), -e.get("correct_winners", 0))


def load_pool_intelligence() -> dict[str, Any]:
    """Estado competitivo del pool: zona de premios, dónde estamos, distribución y trayectoria.

    Usa el leaderboard vivo (estado actual) + los snapshots por jornada (trayectoria).
    Read-only. Todo lo que la API expone es agregado por penca — los picks ajenos son ciegos.
    """
    return _cached("pool_intel", 30.0, _load_pool_intelligence_uncached)


def _load_pool_intelligence_uncached() -> dict[str, Any]:
    import httpx
    from src.utils.env import get_int_list

    base = os.environ.get("PENCA_API_BASE_URL", "").rstrip("/")
    key = os.environ.get("PENCA_API_KEY", "")
    my_ids = set(get_int_list("PENCA_IDS"))
    if not base or not key:
        return {"error": "API no configurada"}
    try:
        with httpx.Client(timeout=8.0, headers={"Authorization": f"Bearer {key}"}) as c:
            r = c.get(f"{base}/leaderboard")
        if r.status_code != 200:
            return {"error": f"leaderboard {r.status_code}"}
        entries = r.json().get("entries", [])
    except Exception as e:
        return {"error": str(e)}
    if not entries:
        return {"error": "leaderboard vacío"}

    ranked = sorted(entries, key=_rank_key)
    n = len(ranked)
    pts = [e.get("points_total", 0) for e in ranked]

    # Zona de premios: top-1/2/3 y cuántos están a ≤1 punto del corte (amenazas reales)
    prize_zone = []
    for place in (1, 2, 3):
        if n >= place:
            cutoff = ranked[place - 1].get("points_total", 0)
            within_1 = sum(1 for p in pts if p >= cutoff - 1)
            at_cutoff = sum(1 for p in pts if p == cutoff)
            prize_zone.append({
                "place": place, "cutoff": cutoff,
                "within_1": within_1, "tied_at_cutoff": at_cutoff,
            })

    # Dónde estamos nosotros
    mine = [(i, e) for i, e in enumerate(ranked) if int(e.get("penca_id", 0)) in my_ids]
    us = None
    if mine:
        best_i, best_e = min(mine, key=lambda t: _rank_key(t[1]))
        cutoff3 = prize_zone[-1]["cutoff"] if prize_zone else None
        best_pts = best_e.get("points_total", 0)
        in_prize = sum(1 for i, _ in mine if i < 3)
        # competidores (no nuestros) por delante de nuestra mejor penca
        ahead = sum(1 for j in range(best_i) if int(ranked[j].get("penca_id", 0)) not in my_ids)
        us = {
            "best_rank": best_i + 1,
            "best_points": best_pts,
            "best_name": _short_name(best_e),
            "n_in_top3": in_prize,
            "n_pencas": len(mine),
            "gap_to_top3": (best_pts - cutoff3) if cutoff3 is not None else None,
            "competitors_ahead": ahead,
            "all_ranks": sorted(i + 1 for i, _ in mine),
        }

    # Distribución de puntos del pool
    from collections import Counter
    dist_counter = Counter(pts)
    # Cuántas de NUESTRAS pencas caen en cada puntaje
    mine_counter = Counter(
        e.get("points_total", 0)
        for e in ranked
        if int(e.get("penca_id", 0)) in my_ids
    )
    max_count = max(dist_counter.values()) if dist_counter else 1
    distribution = [
        {
            "points": p,
            "count": c,
            "pct": round(c / n * 100, 1),
            "bar_pct": round(c / max_count * 100),
            "mine": mine_counter.get(p, 0),
        }
        for p, c in sorted(dist_counter.items(), key=lambda kv: -kv[0])
    ]

    # Trayectoria desde los snapshots (uno por partido jugado)
    trajectory = _pool_trajectory(my_ids)

    # Señales de salud de la estrategia (lo que conviene monitorear jornada a jornada)
    signals = _strategy_signals(mine, ranked)

    return {
        "pool_size": n,
        "prize_zone": prize_zone,
        "us": us,
        "distribution": distribution,
        "trajectory": trajectory,
        "median_points": pts[n // 2] if pts else 0,
        "signals": signals,
    }


def _short_name(e: dict) -> str:
    raw = e.get("penca_name") or ""
    if raw.lower().startswith("penca "):
        return raw.split(" ", 1)[1].strip()
    return raw or str(e.get("penca_id", "?"))


def _pool_trajectory(my_ids: set) -> list[dict]:
    """Evolución del top/mediana/nuestra-mejor a lo largo de las jornadas, desde snapshots."""
    sdir = _data_dir() / "pool_snapshots"
    if not sdir.exists():
        return []
    snaps = []
    for f in sdir.glob("*.json"):
        try:
            snaps.append(json.loads(f.read_text()))
        except Exception:
            continue
    # ordenar por cantidad de partidos jugados al momento del snapshot
    snaps.sort(key=lambda s: (len(s.get("finished_matches", [])), s.get("taken_at", "")))
    out = []
    for s in snaps:
        entries = s.get("entries", [])
        if not entries:
            continue
        ranked = sorted(entries, key=_rank_key)
        pts = [e.get("points_total", 0) for e in ranked]
        mine = [(i, e) for i, e in enumerate(ranked) if int(e.get("penca_id", 0)) in my_ids]
        best = min(mine, key=lambda t: _rank_key(t[1])) if mine else None
        out.append({
            "n_played": len(s.get("finished_matches", [])),
            "top": pts[0] if pts else 0,
            "median": pts[len(pts) // 2] if pts else 0,
            "our_best_points": best[1].get("points_total", 0) if best else None,
            "our_best_rank": (best[0] + 1) if best else None,
        })
    return out


def _strategy_signals(mine: list, ranked: list) -> dict:
    """Las 3 señales que importan para el objetivo P(al menos una penca gana el pool).

    Cada una trae un `status`: ok (verde) / warn (amber) / alert (rojo) / info (neutral).
    Mismo criterio que scripts/monitor_estrategia.py.
        1. tail   — ¿tenemos una penca en la cola ganadora? (rank de la mejor + cobertura top-N)
        2. spread — ¿se mantiene la diversificación? (puntos mejor − peor; 0 = convergencia a chalk)
        3. tailbet— ¿el optimizador persigue el corte (p_top_k) o suma puntos (e_max fallback)?
    """
    my_ranks = sorted(i + 1 for i, _ in mine)
    my_pts = [e.get("points_total", 0) for _, e in mine]
    my_ids = {int(e.get("penca_id", 0)) for _, e in mine}
    top_pts = ranked[0].get("points_total", 0) if ranked else 0

    # --- Señal 1: cola ganadora ---
    best_rank = my_ranks[0] if my_ranks else None
    best_pts = max(my_pts) if my_pts else 0
    if best_rank is None:
        tail_status = "alert"
    elif best_rank <= 10:
        tail_status = "ok"
    elif best_rank <= 30:
        tail_status = "warn"
    else:
        tail_status = "alert"
    tail = {
        "best_rank": best_rank,
        "gap_to_top": top_pts - best_pts,
        "in_top10": sum(1 for r in my_ranks if r <= 10),
        "in_top50": sum(1 for r in my_ranks if r <= 50),
        "in_top100": sum(1 for r in my_ranks if r <= 100),
        "status": tail_status,
    }

    # --- Señal 2: diversificación (spread interno) ---
    spread = (max(my_pts) - min(my_pts)) if my_pts else 0
    spread_status = "ok" if spread >= 6 else ("warn" if spread >= 3 else "alert")
    spread_sig = {
        "value": spread,
        "best": max(my_pts) if my_pts else 0,
        "worst": min(my_pts) if my_pts else 0,
        "status": spread_status,
    }

    # --- Señal 3: tail-bet (objetivo del optimizador) ---
    nm = load_next_match_data() or {}
    meta = nm.get("assignment_meta") or {}
    raw_obj = meta.get("objective")   # "p_top_k" | "e_max" | "e_max (P(top-K)=0)" | "none" | None
    obj = raw_obj or "—"
    threshold = meta.get("threshold")
    premium = meta.get("horizon_premium")
    try:
        beta = float(os.environ.get("PENCA_HORIZON_BETA", "2.0"))
    except ValueError:
        beta = 2.0
    remaining = round((premium / beta) ** 2) if (premium and beta) else None
    projected = (threshold + premium) if (threshold is not None and premium) else None

    active = bool(raw_obj) and raw_obj.startswith("p_top_k")
    fallback = bool(raw_obj) and raw_obj.startswith("e_max (P(top-K)=0)")
    # "A ciegas": dice e_max pero NO por el fallback proyectado, sino porque ni siquiera
    # tenemos el corte del pool (no se pudo leer de la API) → no puede perseguir el top-K.
    blind = bool(raw_obj) and raw_obj.startswith("e_max") and not fallback and threshold is None
    no_meta = (not meta) or raw_obj in (None, "none", "")

    # Estado + explicación de POR QUÉ está como está (detecta que no cambió cuando debía).
    if no_meta:
        tb_status, reason = "alert", (
            "No hay asignación para el próximo partido: el pipeline no generó picks "
            "(o no hay próximo partido cargado). Revisá que las pasadas estén corriendo.")
    elif blind:
        tb_status, reason = "alert", (
            "Está en «Sumando» pero NO porque sea temprano: no se pudo leer el corte del pool "
            "desde la API, así que el optimizador no puede perseguir el top-K — vuela a ciegas. "
            "Revisá la API del pool.")
    elif active:
        tb_status, reason = "ok", (
            "Persiguiendo el corte: el corte proyectado ya es alcanzable en este partido, "
            "así que arriesga para meter una penca arriba.")
    elif fallback:
        cut_txt = (f" El corte proyectado ({threshold}+{premium:.0f}={projected:.0f} pts)"
                   if projected is not None else " El corte proyectado")
        if remaining is not None and remaining <= 8:
            tb_status, reason = "alert", (
                f"Sigue en «Sumando» con solo ~{remaining} partidos restantes: a esta altura ya "
                f"debería estar en «A ganar».{cut_txt} no baja lo suficiente — β={beta:g} está "
                f"demasiado alto o el corte del pool no se actualiza. Revisalo.")
        elif remaining is not None and remaining <= 16:
            tb_status, reason = "warn", (
                f"En «Sumando» con ~{remaining} partidos restantes (entrando a eliminatorias). "
                f"Momento de evaluar bajar β (hoy {beta:g}) para que se ponga agresivo antes.")
        else:
            tb_status, reason = "info", (
                f"En «Sumando» (esperado al principio).{cut_txt} es inalcanzable en un solo "
                f"partido, así que junta puntos y diversifica. Pasa solo a «A ganar» cuando el "
                f"premium del horizonte baje con las fechas.")
    else:
        tb_status, reason = "info", "Modo de objetivo del optimizador."

    tailbet = {
        "objective": obj,
        "active": active,
        "fallback": fallback,
        "blind": blind,
        "no_meta": no_meta,
        "threshold": threshold,
        "premium": round(premium, 1) if premium else premium,
        "projected_cutoff": round(projected) if projected is not None else None,
        "beta": beta,
        "matches_remaining": remaining,
        "next_match_id": nm.get("match_id"),
        "reason": reason,
        "status": tb_status,
    }

    secondary = _secondary_signals(my_ids)
    return {
        "tail": tail, "spread": spread_sig, "tailbet": tailbet,
        "modal": secondary["modal"], "draws": secondary["draws"],
    }


def _secondary_signals(my_ids: set) -> dict:
    """Señales de contexto (no son pasa/falla), leídas de los postmortems ya computados.

        modal — qué tan seguido el resultado más probable del mercado le pega al ganador.
                Bajo = torneo impredecible → la varianza nos conviene. Siempre informativo.
        draws — en los partidos que terminaron empate, cuántas de nuestras picks lo previeron.
                Cobertura baja con racha de empates = sesgo pro-favorito a revisar (warn).
    """
    pmdir = _data_dir() / "postmortems"
    modal_n = modal_win = modal_exact = 0
    draw_matches = draw_covered = draw_slots = draw_union = 0
    modal_log: list[dict] = []   # un registro por partido con modal, para listar los últimos
    draw_log: list[dict] = []    # un registro por partido que terminó empate
    if pmdir.exists():
        for f in pmdir.glob("*.json"):
            if ".bak" in f.name:
                continue
            try:
                pm = json.loads(f.read_text())
            except Exception:
                continue
            ah, aa = pm.get("actual_home"), pm.get("actual_away")
            if ah is None or aa is None:
                continue
            home = pm.get("home_team", "?")
            away = pm.get("away_team", "?")
            ts = pm.get("generated_at", "")
            mm = pm.get("market_modal_score")
            if mm and len(mm) == 2:
                modal_n += 1
                _w = lambda h, a: 0 if h > a else (1 if h == a else 2)
                hit = _w(mm[0], mm[1]) == _w(ah, aa)
                if hit:
                    modal_win += 1
                if list(mm) == [ah, aa]:
                    modal_exact += 1
                modal_log.append({
                    "home": home, "away": away, "ts": ts,
                    "actual": f"{ah}-{aa}", "modal": f"{mm[0]}-{mm[1]}", "hit": hit,
                })
            if ah == aa:
                ours = [r for r in pm.get("pencas_results", []) if int(r.get("penca_id", 0)) in my_ids]
                if ours:
                    covered_here = sum(
                        1 for r in ours
                        if (r.get("predicted_score") or [0, 1])[0] == (r.get("predicted_score") or [0, 1])[1]
                    )
                    draw_matches += 1
                    draw_slots += len(ours)
                    draw_covered += covered_here
                    if covered_here > 0:
                        draw_union += 1   # objetivo de la UNIÓN: alcanza con que ≥1 penca lo cubra
                    draw_log.append({
                        "home": home, "away": away, "ts": ts,
                        "actual": f"{ah}-{aa}", "covered": covered_here, "total": len(ours),
                    })

    modal_log.sort(key=lambda d: d["ts"], reverse=True)
    draw_log.sort(key=lambda d: d["ts"], reverse=True)

    modal_rate = round(100 * modal_win / modal_n) if modal_n else None
    modal = {
        "n": modal_n,
        "winner_hits": modal_win,
        "exact_hits": modal_exact,
        "winner_rate": modal_rate,
        "recent": modal_log[:3],
        "status": "info",  # contexto puro: bajo es bueno para nosotros, no es alarma
    }

    # Métrica PRIMARIA: cobertura de UNIÓN — % de empates con ≥1 penca, alineada al objetivo
    # del sistema (P(al menos una penca gana), no E[puntos]). El % de slots (promedio de
    # planillas) queda como dato secundario de robustez/piso.
    union_raw = (100 * draw_union / draw_matches) if draw_matches else None
    union_pct = round(union_raw) if union_raw is not None else None
    cov_pct = round(100 * draw_covered / draw_slots) if draw_slots else None
    avg_slots = round(draw_slots / draw_matches, 1) if draw_matches else None  # N real (hubo transición 5→15)
    draws = {
        "n_draws": draw_matches,
        "union_pct": union_pct,           # primaria
        "coverage_pct": cov_pct,          # secundaria (slots)
        "avg_per_match": round(draw_covered / draw_matches, 1) if draw_matches else None,
        "avg_slots": avg_slots,
        "recent": draw_log[:3],
        # warn solo si se nos escapan empates ENTEROS (sin ninguna penca). Comparación contra
        # el valor sin redondear para que el número mostrado y la regla no se contradigan.
        "status": "warn" if (draw_matches >= 4 and union_raw is not None and union_raw < 75) else "info",
    }
    return {"modal": modal, "draws": draws}


def _publication_signal() -> dict:
    """¿Las picks de los próximos partidos están REALMENTE cargadas en el pool?

    Clave: no alcanza con `published == True`. Bajo DRY_RUN el NullPublisher marca
    published=True sin tocar la API (loguea y nada más). Por eso cruzamos el flag del
    archivo con el MODO real de publicación (DRY_RUN / API configurada). Una pick que no
    llegó al pool = 0 fijo en esa penca.
    """
    import yaml
    dry_run = os.environ.get("DRY_RUN", "true").lower() == "true"
    has_api = bool(os.environ.get("PENCA_API_BASE_URL") and os.environ.get("PENCA_API_KEY"))
    real = (not dry_run) and has_api  # ¿publicamos de verdad?

    now = datetime.now(timezone.utc)
    horizon = now + timedelta(hours=24)
    try:
        fixtures_path = Path(__file__).resolve().parents[2] / "config" / "fixtures.yaml"
        fixtures = yaml.safe_load(fixtures_path.read_text()) or {}
    except Exception:
        fixtures = {}
    all_matches = (fixtures.get("fase_grupos") or []) + (fixtures.get("eliminatorias") or [])

    upcoming = []
    for mm in all_matches:
        ko_raw = mm.get("kickoff_utc")
        if not ko_raw:
            continue
        try:
            ko = datetime.fromisoformat(str(ko_raw).replace("Z", "+00:00"))
        except Exception:
            continue
        if not (now < ko <= horizon):
            continue
        pred = _load_latest_prediction(mm["id"])
        published = pred.get("published") if pred else None
        upcoming.append({
            "match_id": mm["id"],
            "label": f'{mm.get("home_name") or mm.get("home", "?")} vs {mm.get("away_name") or mm.get("away", "?")}',
            "hours_to": (ko - now).total_seconds() / 3600,
            "published": published,
            "real_pub": bool(real and published is True),  # existe DE VERDAD en el pool
        })
    upcoming.sort(key=lambda u: u["hours_to"])
    n = len(upcoming)
    n_real = sum(1 for u in upcoming if u["real_pub"])

    if not real:
        why = "DRY_RUN está activo" if dry_run else "la API del pool no está configurada"
        if n == 0:
            status, reason = "info", (
                f"Modo prueba: {why}, así que las picks no se cargan en el pool (solo se loguean). "
                f"No hay partidos en las próximas 24h, pero ojo cuando los haya.")
        else:
            imminent = any(u["hours_to"] <= 0.5 for u in upcoming)
            status = "alert" if imminent else "warn"
            reason = (
                f"Modo prueba: {why}. Hay {n} partido(s) en 24h y las picks NO están llegando al pool, "
                f"aunque el sistema las marque como «publicadas». "
                f"{'Uno arranca en menos de 30min. ' if imminent else ''}"
                f"Para jugar de verdad: DRY_RUN=false y la API configurada.")
    else:
        blocked = [u for u in upcoming if u["published"] is False]
        missing_late = [u for u in upcoming if u["published"] is not True and u["hours_to"] <= 3]
        imminent_bad = [u for u in upcoming if not u["real_pub"] and u["hours_to"] <= 0.5]
        if imminent_bad or any(u["hours_to"] <= 1.5 for u in blocked):
            status, reason = "alert", (
                "Hay picks sin confirmar en el pool con el partido encima. Reintentá la pasada del "
                "partido afectado ya.")
        elif blocked or missing_late:
            status, reason = "warn", (
                "Falta confirmar la publicación de algún partido próximo (puede que la pasada tardía "
                "no haya corrido). Revisá antes del kickoff.")
        elif n == 0:
            status, reason = "ok", "Publicando de verdad. No hay partidos en las próximas 24h."
        else:
            status, reason = "ok", (
                f"Publicando de verdad: las {n} pick(s) de las próximas 24h están cargadas en el pool.")

    if not real:
        value = "Modo prueba"
    elif status == "ok":
        value = "Al día" if n else "Sin partidos"
    elif status == "alert":
        value = "¡Sin cargar!"
    else:
        value = "Revisar"

    return {
        "status": status, "value": value, "reason": reason,
        "real": real, "dry_run": dry_run, "has_api": has_api,
        "n_upcoming": n, "n_real": n_real,
        "next": upcoming[0] if upcoming else None,
    }


def _pool_fit_signal() -> dict:
    """¿La distribución de puntos que PREDICE el modelo del pool (ranking-inversion) le pega
    a la OBSERVADA en el delta del leaderboard, jornada a jornada?

    Reusa src/meta/calibration con la config calibrada vigente. Por cada observación calcula
    el residuo L2 multicanal (clases de puntos + frac. exactos + frac. ganadores), SIN la
    regularización del loss de calibración. Flag si el último calza bastante peor que lo típico
    (ratio vs la mediana previa). Termómetro del único modelo que dispara la apuesta de varianza.
    """
    try:
        from src.meta.calibration import (
            build_observations, predicted_group_shares, predicted_exact_winner,
            get_pool_config, load_calibration, W_POINTS, W_EXACT, W_WINNER,
        )
    except Exception as e:
        return {"status": "info", "available": False,
                "reason": f"Módulo de calibración no disponible: {e}"}

    try:
        obs = build_observations()
    except Exception as e:
        return {"status": "info", "available": False,
                "reason": f"No se pudieron armar observaciones del pool: {e}"}
    if not obs:
        return {"status": "info", "available": False,
                "reason": "Todavía no hay jornadas con snapshot + resultado para evaluar el modelo del pool."}

    cfg = get_pool_config()
    cal = load_calibration() or {}
    no_show = float(cal.get("no_show_frac", 0.0) or 0.0)

    residuals = []
    for ob in obs:
        pred = predicted_group_shares(ob.grids, ob.actuals, cfg, no_show)
        classes = set(pred) | set(ob.shares)
        r = W_POINTS * sum((pred.get(c, 0.0) - ob.shares.get(c, 0.0)) ** 2 for c in classes)
        if ob.exact_frac is not None or ob.winner_frac is not None:
            pe, pw = predicted_exact_winner(ob.grids, ob.actuals, cfg, no_show)
            if ob.exact_frac is not None:
                r += W_EXACT * (pe - ob.exact_frac) ** 2
            if ob.winner_frac is not None:
                r += W_WINNER * (pw - ob.winner_frac) ** 2
        residuals.append(r)

    n = len(residuals)
    last = residuals[-1]
    median = sorted(residuals)[n // 2]
    # "Típico" = mediana de las jornadas PREVIAS (sin la última) → comparamos la última contra su pasado.
    ref = residuals[:-1] or residuals
    typical = sorted(ref)[len(ref) // 2]
    ratio = (last / typical) if typical > 1e-9 else (2.0 if last > 1e-6 else 1.0)
    rising3 = n >= 3 and residuals[-1] > residuals[-2] > residuals[-3]
    improvement = cal.get("improvement_pct")
    FLAGS_MIN = 5  # con menos jornadas el ratio es ruidoso

    if improvement is not None and improvement <= 0 and n >= 3:
        status = "alert"
        reason = (f"El fit calibrado predice PEOR que el prior (mejora {improvement:.0f}%). Sospechá "
                  f"datos contaminados (snapshots de la regla vieja pre-12/06) o un partido outlier — "
                  f"revisá antes de confiar en la adivinanza del pool.")
    elif n < FLAGS_MIN:
        status = "info"
        reason = (f"Modelo del pool en rodaje: {n} jornada(s) evaluada(s). Con tan pocas conviene no "
                  f"gatillar; el termómetro se activa con {FLAGS_MIN}+. Por ahora: residuo última "
                  f"{last:.3f}, típico {typical:.3f}.")
    elif ratio >= 1.5:
        status = "alert"
        reason = (f"La última jornada calza bastante peor que lo típico (residuo {last:.3f}, ~{ratio:.1f}× "
                  f"lo normal {typical:.3f}). La adivinanza del pool se está desviando → forzá un "
                  f"recalibrado; si no baja, el prior (chalk/bias) está mal especificado.")
    elif ratio >= 1.2 or rising3:
        status = "warn"
        reason = (f"La última jornada calza algo peor que lo típico (residuo {last:.3f}, ~{ratio:.1f}× "
                  f"lo normal {typical:.3f}){' y sube 3 jornadas seguidas' if rising3 else ''}. "
                  f"Para vigilar, no urgente.")
    else:
        status = "ok"
        reason = (f"El modelo del pool le viene pegando: la última jornada calza como de costumbre "
                  f"(residuo {last:.3f}, ~{ratio:.1f}× lo normal {typical:.3f}).")

    return {
        "status": status, "available": True, "reason": reason,
        "n_obs": n, "last": round(last, 4), "median": round(median, 4),
        "typical": round(typical, 4), "ratio": round(ratio, 2),
        "improvement_pct": round(improvement, 1) if improvement is not None else None,
        "chalk": cal.get("chalk_strength"), "bias": cal.get("bias_scale"), "no_show": no_show,
    }


def _strategy_diagnostics(my_ids: set) -> dict:
    """Rendimiento por estrategia y utilidad del ajuste LLM (Capa 4), desde postmortems."""
    from collections import Counter, defaultdict
    pmdir = _data_dir() / "postmortems"
    pts: dict = defaultdict(int)
    npick: dict = defaultdict(int)
    exact: dict = defaultdict(int)
    win: dict = defaultdict(int)
    llm: Counter = Counter()
    n_matches = 0
    if pmdir.exists():
        for f in pmdir.glob("*.json"):
            if ".bak" in f.name:
                continue
            try:
                pm = json.loads(f.read_text())
            except Exception:
                continue
            if pm.get("actual_home") is None:
                continue
            n_matches += 1
            for r in pm.get("pencas_results", []):
                if int(r.get("penca_id", 0)) not in my_ids:
                    continue
                s = r.get("strategy_used", "?")
                pts[s] += r.get("points_earned", 0)
                npick[s] += 1
                if r.get("is_exact"):
                    exact[s] += 1
                if r.get("correct_winner"):
                    win[s] += 1
            lh = pm.get("llm_adjustment_was_helpful")
            if lh:
                llm[lh] += 1
    by_strategy = sorted(
        (
            {
                "strategy": s,
                "picks": npick[s],
                "points": pts[s],
                "ppp": round(pts[s] / npick[s], 2) if npick[s] else 0,
                "win_pct": round(100 * win[s] / npick[s]) if npick[s] else 0,
                "exacts": exact[s],
            }
            for s in npick
        ),
        key=lambda d: -d["ppp"],
    )
    return {
        "n_matches": n_matches,
        "by_strategy": by_strategy,
        "llm": {"yes": llm.get("yes", 0), "no": llm.get("no", 0), "neutral": llm.get("neutral", 0)},
    }


def _capa4_eval() -> dict:
    """Valor MARGINAL del ajuste LLM (Capa 4) con vs sin δ, read-only.

    - Δlog-loss / ΔRPS pareado sobre el 1X2 real: calibración del modelo, señal POR partido
      (mucho más potente que el binario ayudó/perjudicó a n chico).
    - Contrafactual de 15 pencas: re-corre greedy_assignment con λ (con LLM) y con λ−δ (sin LLM),
      MISMOS standings/threshold/premium → el confound del partido se cancela en la resta.
      `picks_changed` = picks que el δ efectivamente movió; `dpoints` = Δpuntos contra el real.
    """
    import math
    pmdir = _data_dir() / "postmortems"
    if not pmdir.exists():
        return {"n_adjusted": 0, "n_matches": 0}
    try:
        import numpy as np
        from src.model.poisson import score_grid, marginals, jmlm_points
        from src.strategy.portfolio import generate_candidates, picks_to_dicts
        from src.strategy.assignment import greedy_assignment
        from src.meta.calibration import get_pool_config
        from src.meta.pool import pool_pick_distribution
        pool_cfg = get_pool_config()
    except Exception:
        return {"n_adjusted": 0, "n_matches": 0, "error": "imports"}

    def _prior_standings(mid, ids):
        # standings del snapshot más reciente previo al partido (el error se cancela en pre/post)
        for prev in range(int(mid) - 1, 0, -1):
            f = _data_dir() / "pool_snapshots" / f"{prev}.json"
            if not f.exists():
                continue
            try:
                snap = json.loads(f.read_text())
            except Exception:
                continue
            byid = {e["penca_id"]: e for e in snap.get("entries", [])}
            return {pid: {"points_total": byid.get(pid, {}).get("points_total", 0)} for pid in ids}
        return {pid: {"points_total": 0} for pid in ids}

    def _wc(h, a):
        return 0 if h > a else (1 if h == a else 2)

    n_matches = n_adj = ll_better = ll_worse = picks_changed = dpoints = 0
    ll_sum = rps_sum = 0.0
    for f in sorted(pmdir.glob("*.json")):
        if ".bak" in f.name:
            continue
        try:
            pm = json.loads(f.read_text())
        except Exception:
            continue
        if pm.get("actual_home") is None:
            continue
        n_matches += 1
        mid = pm["match_id"]
        actual = (pm["actual_home"], pm["actual_away"])
        pred = _load_latest_prediction(mid)
        if not pred:
            continue
        c = pred.get("constraints") or {}
        qa = pred.get("qualitative_adjustment") or {}
        dL = float(qa.get("delta_lambda_L", 0) or 0)
        dV = float(qa.get("delta_lambda_V", 0) or 0)
        if abs(dL) + abs(dV) <= 1e-6:
            continue
        lL, lV = c.get("lambda_L"), c.get("lambda_V")
        if lL is None or lV is None:
            continue
        l12 = c.get("lambda_12", 0.1)
        n_adj += 1
        try:
            g_post = score_grid(lL, lV, l12, max_goals=7)
            g_pre = score_grid(max(0.05, lL - dL), max(0.05, lV - dV), l12, max_goals=7)
            k = _wc(*actual)

            def _p(g):
                m = marginals(g)
                return [float(m.p_home_win), float(m.p_draw), float(m.p_away_win)]

            pp, pr = _p(g_post), _p(g_pre)
            eps = 1e-12
            dll = (-math.log(max(pr[k], eps))) - (-math.log(max(pp[k], eps)))
            ll_sum += dll
            if dll > 1e-9:
                ll_better += 1
            elif dll < -1e-9:
                ll_worse += 1
            cdf_r = np.cumsum([1 if i == k else 0 for i in range(3)])
            rps_sum += 0.5 * (
                float(np.sum((np.cumsum(pr) - cdf_r) ** 2)) - float(np.sum((np.cumsum(pp) - cdf_r) ** 2))
            )
        except Exception:
            continue
        try:
            ids = [a["penca_id"] for a in pred.get("assignment", [])]
            meta = pred.get("assignment_meta") or {}
            st = _prior_standings(mid, ids)
            cp = picks_to_dicts(generate_candidates(g_post, market_p_home=c.get("p_home"), market_p_away=c.get("p_away"), pool_config=pool_cfg, max_candidates=10))
            cr = picks_to_dicts(generate_candidates(g_pre, market_p_home=c.get("p_home"), market_p_away=c.get("p_away"), pool_config=pool_cfg, max_candidates=10))
            rp, _ = greedy_assignment(cp, ids, g_post, st, pool_top_k_threshold=meta.get("threshold"), pool_q=pool_pick_distribution(g_post, pool_cfg), horizon_premium=meta.get("horizon_premium"))
            rr, _ = greedy_assignment(cr, ids, g_pre, st, pool_top_k_threshold=meta.get("threshold"), pool_q=pool_pick_distribution(g_pre, pool_cfg), horizon_premium=meta.get("horizon_premium"))
            post = {pid: tuple(p["score"]) for pid, p, _ in rp}
            pre = {pid: tuple(p["score"]) for pid, p, _ in rr}
            for pid in ids:
                if post.get(pid) != pre.get(pid):
                    picks_changed += 1
                    dpoints += jmlm_points(post[pid], actual) - jmlm_points(pre[pid], actual)
        except Exception:
            pass

    return {
        "n_matches": n_matches,
        "n_adjusted": n_adj,
        "logloss_sum": round(ll_sum, 2),
        "logloss_mean": round(ll_sum / n_adj, 3) if n_adj else 0,
        "ll_better": ll_better,
        "ll_worse": ll_worse,
        "rps_sum": round(rps_sum, 3),
        "picks_changed": picks_changed,
        "dpoints": dpoints,
    }


def load_strategy_metrics() -> dict[str, Any]:
    """Estado de la estrategia para decidir si hay que ajustar: veredicto + palancas + diagnósticos."""
    return _cached("strategy_metrics", 30.0, _load_strategy_metrics_uncached)


def _load_strategy_metrics_uncached() -> dict[str, Any]:
    from src.utils.env import get_int_list

    pl = load_pool_intelligence()
    if pl.get("error"):
        return {"error": pl["error"]}
    sig = pl.get("signals", {})
    diag = _strategy_diagnostics(set(get_int_list("PENCA_IDS")))

    tail = sig.get("tail", {})
    spread = sig.get("spread", {})
    tb = sig.get("tailbet", {})
    draws = sig.get("draws", {})

    # Palancas de ajuste: cada una con su condición de disparo y la acción sugerida.
    levers = [
        {
            "key": "tailbet",
            "name": "Timing del tail-bet (β)",
            "when": 'sigue en "Sumando" con ≤16 partidos restantes',
            "triggered": tb.get("status") == "warn",
            "action": "bajar PENCA_HORIZON_BETA (hoy 2.0) para que arranque antes — con backtest",
        },
        {
            "key": "draws",
            "name": "Cobertura de empates",
            "when": "racha de empates y >25% se escapan sin ninguna planilla (unión <75%)",
            "triggered": draws.get("status") == "warn",
            "action": "ojo: la exposición a empates NO se ajusta en la asignación (no existe esa "
                      "perilla — el voraz solo reparte el menú). Se toca aguas arriba: en el menú de "
                      "candidatos (generate_candidates / pick_upset, p.ej. un draw_floor) o en el prior "
                      "del pool (popular_score_bias). Subir solo si el backtest muestra al modelo "
                      "infra-prediciendo empates vs su frecuencia real — con backtest",
        },
        {
            "key": "tail",
            "name": "¿Perdimos la cola?",
            "when": "la mejor penca cae > #30 de forma sostenida",
            "triggered": tail.get("status") == "alert",
            "action": "revisar modelo y asignación: estaríamos fuera de zona ganadora",
        },
        {
            "key": "spread",
            "name": "¿Converge a chalk?",
            "when": "el spread interno colapsa hacia 0",
            "triggered": spread.get("status") == "alert",
            "action": "revisar el motor de diversificación de la asignación",
        },
    ]

    pub = _publication_signal()
    pf = _pool_fit_signal()
    urgent = "alert" in (pub.get("status"), pf.get("status"),
                         tail.get("status"), spread.get("status"), tb.get("status"))
    triggered = [lv["name"] for lv in levers if lv["triggered"]]
    if pub.get("status") == "warn":
        triggered = ["Publicación de picks"] + triggered
    if pf.get("status") == "warn":
        triggered = triggered + ["Calce del modelo del pool"]
    if urgent:
        verdict = {"level": "alert", "headline": "Hay que ajustar",
                   "sub": "una señal principal está en alerta"}
    elif triggered:
        verdict = {"level": "warn", "headline": "Sin ajustes urgentes — vigilar",
                   "sub": "para mirar: " + ", ".join(triggered)}
    else:
        verdict = {"level": "ok", "headline": "Sin ajustes",
                   "sub": "la estrategia viene sana; dejar correr"}

    # Una sola lista de señales: estado + significado + (si no está verde) qué hacer.
    # Cada fila trae `stats`/`note`/`rule` para el detalle que se abre al clickear.
    mo = sig.get("modal", {})
    ps = pl.get("pool_size")
    action_by_key = {lv["key"]: (lv["action"] if lv["triggered"] else None) for lv in levers}
    _best = tail.get("best_rank")

    def _st(label, value):
        return {"label": label, "value": value}

    def _bar(pct, status):
        """Mini-barra 0–100 para el resumen de la señal (None = no se dibuja)."""
        if pct is None:
            return None
        return {"pct": max(0, min(100, round(pct))), "status": status}

    if pub.get("real"):
        pub_mode = "Publicando de verdad"
    elif pub.get("dry_run"):
        pub_mode = "Prueba (DRY_RUN activo)"
    else:
        pub_mode = "API sin configurar"
    pub_next = pub.get("next")
    pub_action = None
    if pub.get("status") in ("warn", "alert"):
        pub_action = ("Poné DRY_RUN=false y verificá la API del pool en /etc/penca/env."
                      if not pub.get("real")
                      else "Reintentá la pasada del partido afectado antes del kickoff.")

    if not pf.get("available"):
        pf_value = "Sin datos"
    else:
        pf_value = {"ok": "Pega bien", "warn": "Algo flojo", "alert": "Se desvía",
                    "info": "En rodaje"}.get(pf.get("status"), "—")
    pf_action = None
    if pf.get("status") == "alert":
        pf_action = ("Forzá un recalibrado del pool; si el residuo no baja, revisá el prior "
                     "(chalk/bias) o snapshots contaminados de la regla vieja.")
    elif pf.get("status") == "warn":
        pf_action = "Vigilar; si el residuo sigue subiendo, recalibrá el pool."

    health_rows = [
        {
            "key": "pub", "name": "Publicación de picks", "status": pub.get("status", "info"),
            "value": pub.get("value", "—"),
            "meaning": "que las picks lleguen de verdad al pool (no solo guardadas en disco)",
            "action": pub_action,
            "reason": pub.get("reason"),
            "bar": _bar(100 * pub["n_real"] / pub["n_upcoming"] if pub.get("n_upcoming") else None,
                        pub.get("status", "info")),
            "stats": [
                _st("Modo", pub_mode),
                _st("Partidos en 24h", f"{pub.get('n_upcoming', 0)}"),
                _st("Picks cargadas en el pool", f"{pub.get('n_real', 0)} / {pub.get('n_upcoming', 0)}"),
                _st("Próximo", f"{pub_next['label']} (en {pub_next['hours_to']:.0f}h)" if pub_next else "—"),
            ],
            "note": "Una pick que no llega al pool antes del kickoff = 0 fijo en esa penca. Ojo: bajo "
                    "DRY_RUN el sistema marca «publicado» sin cargar nada — por eso acá cruzamos el "
                    "flag con el modo real (DRY_RUN / API).",
            "rule": "OK: publicando de verdad y las picks de 24h cargadas. Vigilar: modo prueba con "
                    "partidos próximos, o falta confirmar alguna. Alerta: partido en <30min sin pick real.",
        },
        {
            "key": "tail", "name": "Cola ganadora", "status": tail.get("status", "info"),
            "value": (f"#{_best} · top-10: {tail.get('in_top10', 0)}" if _best else "—"),
            "meaning": "que al menos una penca pelee el 1° puesto",
            "action": action_by_key.get("tail"),
            "bar": _bar((1 - (_best - 1) / ps) * 100 if (_best and ps) else None, tail.get("status", "info")),
            "stats": [
                _st("Mejor penca", f"#{_best} de {ps}" if _best else "—"),
                _st("Gap al líder", f"{tail.get('gap_to_top', 0)} pts"),
                _st("En top-10 / 50 / 100", f"{tail.get('in_top10', 0)} / {tail.get('in_top50', 0)} / {tail.get('in_top100', 0)}"),
            ],
            "note": "Para ganar el pool alcanza con que una de las 15 termine arriba. Acá mirás en qué puesto va la mejor.",
            "rule": "Sano: al menos 1 en el top-10. Alerta si la mejor cae por encima de #30 de forma sostenida.",
        },
        {
            "key": "spread", "name": "Diversificación", "status": spread.get("status", "info"),
            "value": f"spread {spread.get('value', 0)} pts",
            "meaning": "las 15 bien distintas entre sí (si no, no hay apuesta de varianza)",
            "action": action_by_key.get("spread"),
            "bar": _bar(min(spread.get("value", 0) / 8, 1) * 100, spread.get("status", "info")),
            "stats": [
                _st("Spread interno", f"{spread.get('value', 0)} pts"),
                _st("Mejor / peor penca", f"{spread.get('best', 0)} / {spread.get('worst', 0)} pts"),
            ],
            "note": "Qué tan distintas son las 15 entre sí. Si se parecen demasiado, ganan o pierden todas juntas y perdés la apuesta de varianza.",
            "rule": "Sano: el spread no colapsa hacia 0. Si tiende a 0 = convergencia a chalk.",
        },
        {
            "key": "tailbet", "name": "Tail-bet", "status": tb.get("status", "info"),
            "value": ("A ganar" if tb.get("active") else "Sumando"),
            "meaning": ("persiguiendo el corte" if tb.get("active") else "juntando puntos — esperado al principio"),
            "action": action_by_key.get("tailbet"),
            "reason": tb.get("reason"),
            "stats": [
                _st("Modo", "A ganar (p_top_k)" if tb.get("active") else "Sumando (e_max)"),
                _st("Corte del pool (hoy)", f"{tb.get('threshold')} pts" if tb.get("threshold") is not None else "— sin dato"),
                _st("Premium horizonte (β·√rest.)", f"+{tb.get('premium')}" if tb.get("premium") is not None else "—"),
                _st("Corte proyectado", f"{tb.get('projected_cutoff')} pts" if tb.get("projected_cutoff") is not None else "—"),
                _st("β (PENCA_HORIZON_BETA)", f"{tb.get('beta')}" if tb.get("beta") is not None else "—"),
                _st("Partidos restantes (aprox)", f"~{tb.get('matches_remaining')}" if tb.get("matches_remaining") else "—"),
            ],
            "note": "Si el sistema todavía junta puntos sin arriesgar (Sumando) o ya juega a ganar el 1° puesto (A ganar). El cambio ocurre solo, cerca del final, cuando el corte proyectado se vuelve alcanzable en un partido.",
            "rule": "Esperado: 'Sumando' al principio. Vigilar si sigue en 'Sumando' con ~≤16 partidos (entrando a eliminatorias) → bajar β. Alerta si ~≤8 partidos y sigue 'Sumando', si no hay corte del pool, o si no hay asignación.",
        },
        {
            "key": "modal", "name": "Favoritos del mercado", "status": mo.get("status", "info"),
            "value": (f"{mo.get('winner_rate')}%" if mo.get("winner_rate") is not None else "—"),
            "meaning": "qué seguido gana el favorito (bajo = más impredecible, nos conviene)",
            "action": None,
            "bar": _bar(mo.get("winner_rate"), "info"),
            "recent": mo.get("recent"),
            "stats": [
                _st("Gana el favorito", f"{mo.get('winner_rate')}%" if mo.get("winner_rate") is not None else "—"),
                _st("Acertó ganador", f"{mo.get('winner_hits', 0)}/{mo.get('n', 0)}"),
                _st("Acertó marcador exacto", f"{mo.get('exact_hits', 0)}/{mo.get('n', 0)}"),
            ],
            "note": "Qué tan seguido el resultado más probable del mercado termina ganando. Cuanto más bajo, más impredecible viene el torneo — y eso nos conviene (más varianza).",
            "rule": "Contexto, no pasa/falla.",
        },
        {
            "key": "draws", "name": "Cobertura de empates", "status": draws.get("status", "info"),
            "value": (f"{draws.get('union_pct')}%" if draws.get("union_pct") is not None else "—"),
            "meaning": "de los empates que salieron, en cuántos cubrimos con al menos una planilla",
            "action": action_by_key.get("draws"),
            "bar": _bar(draws.get("union_pct"), draws.get("status", "info")),
            "recent": draws.get("recent"),
            "stats": [
                _st("Empates con cobertura (≥1 planilla)", f"{draws.get('union_pct')}%" if draws.get("union_pct") is not None else "—"),
                _st("Partidos que salieron empate", f"{draws.get('n_draws', 0)}"),
                _st("Planillas al empate (promedio)",
                    f"~{draws.get('avg_per_match', 0)} de ~{draws.get('avg_slots', 0)} por partido ({draws.get('coverage_pct')}%)"
                    if draws.get("coverage_pct") is not None else "—"),
            ],
            "note": "Para ganar la penca alcanza con que AL MENOS UNA de las 15 cubra cada empate (objetivo de "
                    "la unión), no que lo jueguen todas. Que pocas planillas jueguen el empate es normal y buscado "
                    "(diversificación). El % de planillas promedio queda como dato secundario de robustez.",
            "rule": "A vigilar si hay ≥4 empates y en más del 25% no cubrimos ninguno (unión <75%).",
        },
        {
            "key": "poolfit", "name": "Calce del modelo del pool", "status": pf.get("status", "info"),
            "value": (pf_value),
            "meaning": "que le acertemos a cómo juega el resto del pool (motor de la apuesta de varianza)",
            "action": pf_action,
            "reason": pf.get("reason"),
            "stats": ([
                _st("Jornadas evaluadas", f"{pf.get('n_obs', 0)}"),
                _st("Residuo última vs típico", f"{pf.get('last')} vs {pf.get('typical')} (~{pf.get('ratio')}×)"),
                _st("Mediana histórica", f"{pf.get('median')}"),
                _st("Mejora vs prior", f"{pf.get('improvement_pct')}%" if pf.get("improvement_pct") is not None else "—"),
                _st("Calibración", f"chalk {pf.get('chalk')} · bias {pf.get('bias')} · no-show {pf.get('no_show')}"),
            ] if pf.get("available") else []),
            "note": "El sistema no ve los picks ajenos: los ADIVINA con el ranking-inversion. De esa "
                    "adivinanza salen 'qué tan contrarian es un marcador' y el corte que persigue el "
                    "optimizador. Esta señal compara, jornada a jornada, lo que el modelo predijo que "
                    "iba a jugar la gente contra lo que realmente jugó (delta del leaderboard).",
            "rule": "OK: la última jornada calza ≤ mediana histórica. Vigilar: > mediana o subiendo 3 "
                    "seguidas. Alerta: supera el p80, o el fit predice peor que el prior (datos "
                    "contaminados). Se activa con ≥5 jornadas.",
        },
    ]

    return {
        "pool_size": pl.get("pool_size"),
        "median": pl.get("median_points"),
        "us": pl.get("us"),
        "signals": sig,
        "trajectory": pl.get("trajectory"),
        "verdict": verdict,
        "levers": levers,
        "health_rows": health_rows,
        "by_strategy": diag["by_strategy"],
        "llm": diag["llm"],
        "n_matches": diag["n_matches"],
        "capa4": _capa4_eval(),
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

        label = {"ev": "Favorito", "differentiated": "Diferencial", "tail": "Goles",
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
            obj = _display_objective(entry.get("objective"), sc)
            rank = entry.get("rank")
            total_pencas = len(data.get("assignment") or [])
            is_reinf = _reinforcement_flags(data.get("assignment") or []).get(pid, False)
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
                "assignment_reason": _assignment_reason(rank, obj, total_pencas, is_reinf, sc),
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

_STAGE_ORDER = {
    "group": 0, "round_of_32": 1, "round_of_16": 2,
    "quarter": 3, "semi": 4, "third_place": 5, "final": 6,
}
_STAGE_LABEL = {
    "round_of_32": "Ronda de 32", "round_of_16": "Octavos",
    "quarter": "Cuartos", "semi": "Semifinales",
    "third_place": "Tercer puesto", "final": "Final",
}


def _fixture_index() -> dict[str, dict]:
    """{match_id: fixture_meta} desde config/fixtures.yaml."""
    import yaml
    try:
        fixtures_path = Path(__file__).resolve().parents[2] / "config" / "fixtures.yaml"
        fixtures = yaml.safe_load(fixtures_path.read_text()) or {}
    except Exception:
        return {}
    out = {}
    for m in (fixtures.get("fase_grupos") or []) + (fixtures.get("eliminatorias") or []):
        out[str(m.get("id"))] = m
    return out


def _phase_group(match_id: Any, meta: dict) -> tuple[float, str]:
    """(clave de orden, etiqueta) para agrupar el historial por fase/jornada."""
    stage = meta.get("stage", "?")
    if stage == "group":
        # Jornada del grupo desde el sufijo MATCH_<G>_<NN>: 01/02→J1, 03/04→J2, 05/06→J3
        try:
            nn = int(str(match_id).rsplit("_", 1)[-1])
            md = (nn + 1) // 2
            return (md * 0.1, f"Fase de grupos · Jornada {md}")
        except Exception:
            return (0.9, "Fase de grupos")
    return (10 + _STAGE_ORDER.get(stage, 9), _STAGE_LABEL.get(stage, stage))


def _load_all_postmortems_enriched() -> list[dict]:
    """Todos los postmortems, enriquecidos con meta de fixture y ordenados por
    kickoff real (no por mtime — un git pull/rsync reescribe mtimes). Más reciente primero."""
    pdir = _data_dir() / "postmortems"
    if not pdir.exists():
        return []
    fix = _fixture_index()
    pms = []
    for f in pdir.glob("*.json"):
        try:
            pm = json.loads(f.read_text())
        except Exception:
            continue
        meta = fix.get(str(pm.get("match_id")), {})
        pm["_kickoff"] = meta.get("kickoff_utc", "")
        pm["_stage"] = meta.get("stage", "?")
        pm["_group"] = meta.get("group")
        sort_key, label = _phase_group(pm.get("match_id"), meta)
        pm["_group_sort"] = sort_key
        pm["_group_label"] = label
        pm["_kickoff_uy"] = _to_uy(pm["_kickoff"]) if pm["_kickoff"] else ""
        pms.append(pm)
    pms.sort(key=lambda p: (p.get("_kickoff") or "", str(p.get("match_id"))), reverse=True)
    return pms


def load_recent_postmortems(limit: int = 5) -> list[dict]:
    """Los `limit` postmortems más recientes (por kickoff). Para el resumen del home / API."""
    return _load_all_postmortems_enriched()[:limit]


def _aggregate_postmortems(pms: list[dict]) -> dict:
    """Rollup acumulado a lo largo de todos los partidos: el artefacto para recalibrar."""
    from collections import defaultdict
    strat: dict[str, dict] = defaultdict(lambda: {"pts": 0, "exacts": 0, "winners": 0, "n": 0})
    llm = {"yes": 0, "no": 0, "neutral": 0}
    rank_trend: list[dict] = []
    total_pts = 0
    total_exacts = 0
    best_single = 0

    # cronológico (viejo→nuevo) para la tendencia de rank
    for pm in sorted(pms, key=lambda p: (p.get("_kickoff") or "", str(p.get("match_id")))):
        total_pts += pm.get("portfolio_total_points", 0)
        best_single = max(best_single, pm.get("portfolio_max_points", 0))
        for r in pm.get("pencas_results", []):
            s = strat[r.get("strategy_used", "?")]
            s["pts"] += r.get("points_earned", 0)
            s["n"] += 1
            if r.get("is_exact"):
                s["exacts"] += 1
                total_exacts += 1
            if r.get("correct_winner"):
                s["winners"] += 1
        helpful = pm.get("llm_adjustment_was_helpful")
        if helpful in llm:
            llm[helpful] += 1
        if pm.get("our_best_rank_in_pool"):
            rank_trend.append({
                "label": f"{(pm.get('home_team') or '?')[:3]}-{(pm.get('away_team') or '?')[:3]}",
                "rank": pm["our_best_rank_in_pool"],
            })

    strat_rows = []
    for obj, s in strat.items():
        n = s["n"] or 1
        strat_rows.append({
            "objective": obj,
            "pts": s["pts"],
            "avg": round(s["pts"] / n, 2),
            "exacts": s["exacts"],
            "winners": s["winners"],
            "winner_pct": round(s["winners"] / n * 100),
            "n": s["n"],
        })
    strat_rows.sort(key=lambda r: -r["pts"])

    llm_decided = llm["yes"] + llm["no"]
    if llm_decided == 0:
        llm_verdict = "Sin ajustes decisivos del LLM todavía."
    elif llm["yes"] > llm["no"]:
        llm_verdict = f"El ajuste LLM ayuda en neto ({llm['yes']} a favor / {llm['no']} en contra)."
    elif llm["no"] > llm["yes"]:
        llm_verdict = f"⚠️ El ajuste LLM resta en neto ({llm['no']} en contra / {llm['yes']} a favor) — revisar Capa 4."
    else:
        llm_verdict = f"El ajuste LLM va parejo ({llm['yes']} a favor / {llm['no']} en contra)."

    return {
        "n_matches": len(pms),
        "total_pts": total_pts,
        "total_exacts": total_exacts,
        "best_single": best_single,
        "strategies": strat_rows,
        "best_strategy": strat_rows[0]["objective"] if strat_rows else None,
        "llm": llm,
        "llm_verdict": llm_verdict,
        "rank_trend": rank_trend[-12:],
    }


def load_postmortem_history() -> dict:
    """Historial completo de postmortems para la página de historial:
    panel acumulado + partidos agrupados por fase/jornada (sin cap, navegable con +100 partidos)."""
    pms = _load_all_postmortems_enriched()
    if not pms:
        return {"groups": [], "aggregate": None, "total": 0}

    from itertools import groupby
    by_group = sorted(pms, key=lambda p: (p["_group_sort"], p.get("_kickoff") or ""))
    groups = []
    for (_, label), items in groupby(by_group, key=lambda p: (p["_group_sort"], p["_group_label"])):
        items = list(items)
        items.sort(key=lambda p: (p.get("_kickoff") or "", str(p.get("match_id"))), reverse=True)
        groups.append({
            "label": label,
            "matches": items,
            "n": len(items),
            "portfolio_total": sum(p.get("portfolio_total_points", 0) for p in items),
            "exacts": sum(1 for p in items for r in p.get("pencas_results", []) if r.get("is_exact")),
        })
    groups.reverse()  # fase más reciente arriba

    return {"groups": groups, "aggregate": _aggregate_postmortems(pms), "total": len(pms)}


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
