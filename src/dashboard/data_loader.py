"""Carga la data del dashboard desde los JSON del pipeline."""

from __future__ import annotations

import json
import os
import glob
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


def _data_dir() -> Path:
    from src.utils.env import get_str
    return Path(get_str("DATA_DIR", "data"))


UY_TZ = timezone(timedelta(hours=-3))


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
    files = sorted(pdir.glob("v*_*.json"))
    if not files:
        return None
    return json.loads(files[-1].read_text())


def _load_latest_dossier(match_id: Any) -> dict | None:
    ddir = _data_dir() / "dossiers" / str(match_id)
    if not ddir.exists():
        return None
    files = sorted(ddir.glob("v*.json"))
    if not files:
        return None
    return json.loads(files[-1].read_text())


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


def _assignment_reason(penca_rank: int | None, objective: str, assignment_meta: dict) -> str:
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

    pos = f"rank {penca_rank}"
    if penca_rank == 1:
        return (
            f"Penca actualmente {pos} (puntera). "
            f"Recibe {obj_label} porque el optimizer maximiza P(top-3) y, dada su ventaja, "
            "asignarle el marcador más sólido preserva la posición."
        )
    if penca_rank == 5:
        return (
            f"Penca {pos} (última). "
            f"Recibe {obj_label}: el optimizer detectó que necesita una jugada con mayor varianza "
            "para tener chance real de remontar al cutoff de top-3."
        )
    return (
        f"Penca {pos}. Recibe {obj_label} porque la combinación de su score actual + esa estrategia "
        "es la que más aporta al objetivo global de meter a alguna penca en top-3."
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

    files = sorted(pdir.glob("v*_*.json"))
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
    # Construir lista de pencas con razón por estrategia
    portfolio_picks = {p["objective"]: p for p in (latest.get("portfolio") or {}).get("picks", [])}
    current_pencas = []
    for a in latest.get("assignment") or []:
        obj = a["objective"]
        pick_metrics = portfolio_picks.get(obj, {})
        current_pencas.append({
            "penca_id": a["penca_id"],
            "rank": a.get("rank"),
            "objective": obj,
            "score": a["score"],
            "p_scoreline": pick_metrics.get("p_scoreline"),
            "e_points": pick_metrics.get("e_points"),
            "pool_popularity": pick_metrics.get("pool_popularity"),
            "assignment_reason": _assignment_reason(a.get("rank"), obj, latest.get("assignment_meta") or {}),
            "strategy_rationale": _strategy_rationale_for_pick(obj, pick_metrics),
        })

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
        "latest_meta": latest.get("assignment_meta") or {},
        "latest_qualitative": latest.get("qualitative_adjustment") or {},
        "current_pencas": current_pencas,
        "timeline": timeline,
        "diffs": diffs,
    }


# ---------- pencas ranking vs pool ----------

def load_my_pencas_standings() -> dict[str, Any]:
    """Lee leaderboard real de la penca. Filtra mis 5 pencas."""
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
            my_entries.append({
                "penca_id": pid,
                "penca_name": e.get("penca_name") or f"P{pid}",
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
    """Stub de health — para versión completa usar lógica del heartbeat."""
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
