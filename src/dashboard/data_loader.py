"""Carga la data del dashboard desde los JSON del pipeline."""

from __future__ import annotations

import json
import os
import glob
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


def _data_dir() -> Path:
    return Path(os.environ.get("DATA_DIR", "data"))


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
