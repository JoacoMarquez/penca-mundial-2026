"""Scheduler que el systemd timer ejecuta cada 5 min.

Lee fixtures.yaml, calcula qué partido entra en alguna ventana (T-24h, T-3h, T-30min) AHORA,
y dispara la pipeline correspondiente. Idempotente: si ya corrimos esa fase para ese partido,
no la repite.

La ventana es ±2.5 min para emparejar con el timer cada 5 min. Si la ventana T-24h cae a las
14:32 y el timer corre 14:30 y 14:35, agarra una de las dos.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from src.agent.pipeline import Phase, load_fixtures, PREDICTIONS_DIR, run_match_pipeline

log = logging.getLogger(__name__)


PHASE_OFFSETS_MIN = {
    Phase.T_24H: 24 * 60,
    Phase.T_3H: 3 * 60,
    Phase.T_30MIN: 30,
}


def parse_kickoff(iso: str) -> datetime:
    """Parsea ISO 8601 UTC (con Z) → datetime aware UTC."""
    return datetime.fromisoformat(iso.replace("Z", "+00:00"))


def phase_already_ran(match_id: str, phase: Phase, kickoff: datetime | None = None) -> bool:
    """¿Ya corrió esta fase para este partido, en una pasada RECIENTE?

    Solo cuentan corridas dentro de la ventana real del partido (`run_at` ≥ kickoff − 72h).
    Una predicción vieja —de un test o de otra edición con el mismo id— NO puede marcar la
    fase como "ya corrió"; si lo hiciera, el scheduler se saltaría la pasada (y la publicación)
    del partido real. Sin `kickoff` (modo legacy) cuenta cualquier corrida.
    """
    import json
    match_dir = PREDICTIONS_DIR / str(match_id)
    if not match_dir.exists():
        return False
    cutoff = (kickoff - timedelta(hours=72)) if kickoff is not None else None
    for f in match_dir.glob("v*_*.json"):
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        if data.get("phase") != phase.value and data.get("phase") != phase:
            continue
        # T-30min cuya publicación FALLÓ no cuenta como "ya corrió" → el scheduler reintenta.
        if phase == Phase.T_30MIN and data.get("published") is False:
            continue
        if cutoff is not None:
            ra = data.get("run_at")
            try:
                run_at = datetime.fromisoformat(str(ra).replace("Z", "+00:00")) if ra else None
            except Exception:
                run_at = None
            # Sin fecha o más vieja que kickoff−72h → corrida stale, no cuenta.
            if run_at is None or run_at < cutoff:
                continue
        return True
    return False


def matches_in_window(fixtures: dict, now: datetime | None = None) -> list[tuple[str, Phase]]:
    """Devuelve (match_id, phase) que deberían correrse AHORA, con CATCH-UP.

    Para cada partido aún no jugado (now < kickoff), una fase está "pendiente" si su target
    (kickoff − offset) ya pasó y todavía no corrió. Devolvemos la fase pendiente MÁS TARDÍA
    (la más cercana al kickoff): si nos atrasamos o se saltó un slot del timer, igual disparamos
    la pasada relevante en el próximo despertar — sin ventana fija de ±N min que se pueda perder.

    Garantía clave: T-30min (la que PUBLICA) corre sí o sí mientras el scheduler despierte al
    menos una vez entre su target y el kickoff. `phase_already_ran` evita duplicados.
    """
    now = now or datetime.now(timezone.utc)
    out = []

    all_matches = (fixtures.get("fase_grupos") or []) + (fixtures.get("eliminatorias") or [])
    for m in all_matches:
        kickoff = parse_kickoff(m["kickoff_utc"])
        if now >= kickoff:
            continue  # partido ya arrancó (la web bloquea predicciones) → demasiado tarde

        # Sin equipos resueltos (p.ej. un cruce de eliminatorias cuyo bracket todavía no
        # se definió en la API): no se puede modelar ni publicar. Saltar hasta que el sync
        # traiga los equipos. Sin esto, la pasada corría con un dossier "?" y las odds no
        # mapeaban → constraints MOCK (40/30/30) → pick sesgada al "home" del bracket.
        if not (m.get("home") and m.get("away")):
            log.info(
                "match %s sin equipos resueltos todavía → skip (bracket pendiente)", m.get("id")
            )
            continue

        # Fases cuyo target (kickoff − offset) ya pasó
        due = [
            (phase, offset_min)
            for phase, offset_min in PHASE_OFFSETS_MIN.items()
            if kickoff - timedelta(minutes=offset_min) <= now
        ]
        if not due:
            continue
        # La pasada MÁS RELEVANTE = la de target más tardío = menor offset (más cerca del
        # kickoff). Solo la emitimos si todavía no corrió. Si ya corrió, no hay nada que
        # hacer — NO volvemos a fases anteriores (ya publicamos / es agua pasada).
        latest_phase = min(due, key=lambda p: p[1])[0]
        if not phase_already_ran(m["id"], latest_phase, kickoff):
            out.append((m["id"], latest_phase))

    return out


def _latest_prediction_dict(match_id) -> dict | None:
    """Última versión de predicción (por número de versión) como dict, o None."""
    import json
    from src.utils.versions import latest_version
    match_dir = PREDICTIONS_DIR / str(match_id)
    if not match_dir.exists():
        return None
    p = latest_version(match_dir.glob("v*_*.json"))
    if p is None:
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _retry_pending_publications(fixtures: dict) -> None:
    """Reintenta publicar la última asignación de cualquier partido FUTURO que quedó con
    published=False, barato (solo POSTs, sin re-correr research). Cierra el gap donde una
    pasada T-24h/T-3h fallida no se reintentaba hasta la ventana T-30min (~2.5h después)."""
    from src.agent.pipeline import republish_pending
    now = datetime.now(timezone.utc)
    all_matches = (fixtures.get("fase_grupos") or []) + (fixtures.get("eliminatorias") or [])
    for m in all_matches:
        try:
            if now >= parse_kickoff(m["kickoff_utc"]):
                continue  # ya arrancó → la web bloquea, nada que republicar
        except Exception:
            continue
        latest = _latest_prediction_dict(m["id"])
        # Solo un fallo de publicación (False) se reintenta; True=ok, None=MOCK/degradado/n-a.
        if not latest or latest.get("published") is not False:
            continue
        try:
            republish_pending(m, latest)
        except Exception:
            log.exception("republish_pending falló | match=%s", m.get("id"))


def main() -> int:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)sZ | %(levelname)s | %(name)s | %(message)s",
    )
    fixtures = load_fixtures()

    # Reintento BARATO de publicaciones pendientes (solo POSTs, sin re-investigar). Cierra el
    # gap entre una pasada temprana que falló al publicar y la ventana T-30min. Corre ANTES de
    # matches_in_window: si logra publicar, la nueva versión (published=True) evita que el
    # scheduler re-corra la pipeline cara por published=False.
    try:
        _retry_pending_publications(fixtures)
    except Exception:
        log.exception("reintento de publicaciones pendientes falló (no bloqueante)")

    pending = matches_in_window(fixtures)

    # Pipeline para partidos en ventana
    if pending:
        log.info("scheduler tick — %d tarea(s): %s", len(pending), pending)
        for match_id, phase in pending:
            try:
                run_match_pipeline(match_id, phase)
            except Exception:
                log.exception("error corriendo pipeline | match=%s phase=%s", match_id, phase.value)
                _notify_error("pipeline", f"match={match_id} phase={phase.value}")

    # Postmortems para partidos terminados sin reporte aún
    try:
        from src.agent.postmortem import (
            find_finished_matches_pending_postmortem,
            compute_postmortem,
            save_postmortem,
        )
        pm_pending = find_finished_matches_pending_postmortem(fixtures)
        if pm_pending:
            log.info("scheduler tick — %d postmortem(s) pendiente(s): %s", len(pm_pending), pm_pending)
            completed = []
            for mid in pm_pending:
                try:
                    report = compute_postmortem(mid)
                    if not report:
                        continue
                    save_postmortem(report)
                    completed.append(mid)
                    # No se notifica por partido: el resumen va en el digest de jornada.
                    log.info("Postmortem completado para match %s", mid)
                except Exception:
                    log.exception("error en postmortem | match=%s", mid)
                    _notify_error("postmortem", f"match={mid}")
            # UN solo snapshot por tick: con partidos simultáneos el leaderboard ya
            # incluye los puntos de todos, así que un snapshot por partido quedaría
            # contaminado (deltas mezclados atribuidos a un solo partido). El snapshot
            # único con finished=todos no genera observación individual (sin predecesor
            # exacto, build_observations lo saltea) pero mantiene la cadena limpia
            # como predecesor del próximo partido.
            if completed:
                if len(completed) > 1:
                    log.info(
                        "snapshot multi-partido (%d simultáneos: %s) — se pierden las "
                        "observaciones individuales pero la cadena queda limpia",
                        len(completed), completed,
                    )
                _snapshot_and_recalibrate(completed[-1])
    except Exception:
        log.exception("error general en postmortem block")

    # Digest de jornada: un solo mensaje al cierre del día (resultados + banderas).
    try:
        _maybe_send_jornada_digest(fixtures)
    except Exception:
        log.exception("digest de jornada falló (no bloqueante)")

    if not pending and not _has_pending_postmortems(fixtures):
        log.info("scheduler tick — nada que hacer")
    return 0


def _has_pending_postmortems(fixtures: dict) -> bool:
    try:
        from src.agent.postmortem import find_finished_matches_pending_postmortem
        return bool(find_finished_matches_pending_postmortem(fixtures))
    except Exception:
        return False


def _snapshot_and_recalibrate(match_id) -> None:
    """Capa 5: snapshot del leaderboard post-partido + recalibración del modelo del pool.

    Best-effort — un fallo acá no debe frenar la notificación del postmortem.
    `finished_matches` = postmortems ya guardados (incluye este: save_postmortem corre antes).
    """
    try:
        from src.meta.calibration import snapshot_leaderboard, recalibrate_from_disk
        from src.agent.postmortem import _data_dir as _pm_data_dir
        pm_dir = _pm_data_dir() / "postmortems"
        finished = [p.stem for p in pm_dir.glob("*.json")] if pm_dir.exists() else [str(match_id)]
        if snapshot_leaderboard(match_id, finished) is not None:
            fit = recalibrate_from_disk()
            if fit:
                log.info(
                    "pool recalibrado | chalk=%.2f β=%.2f no_show=%.2f | mejora vs prior: %.1f%% | n=%d",
                    fit["chalk_strength"], fit["bias_scale"], fit["no_show_frac"],
                    fit["improvement_pct"], fit["n_observations"],
                )
    except Exception:
        log.exception("snapshot/recalibración falló | match=%s", match_id)


def _strategy_signal_flags() -> list:
    """Banderas post-jornada a partir de las señales del dashboard que estén en alerta/vigilar.

    Reusa load_strategy_metrics() (misma fuente que /metricas) para que el aviso de Telegram y
    el dashboard nunca se contradigan. Best-effort: un fallo no frena el digest.
    """
    try:
        from src.dashboard.data_loader import load_strategy_metrics
        from src.agent.alerts import Flag
        m = load_strategy_metrics()
        if not m or m.get("error"):
            return []
        flags = []
        for r in m.get("health_rows", []):
            st = r.get("status")
            if st not in ("alert", "warn"):
                continue
            sev = "ALERTA" if st == "alert" else "Vigilar"
            detail = r.get("action") or r.get("reason") or r.get("meaning") or ""
            flags.append(Flag(
                level=("warn" if st == "alert" else "info"),
                code=f"signal_{r.get('key')}",
                title=f"{sev} — {r.get('name')}: {r.get('value')}",
                detail=detail,
            ))
        return flags
    except Exception:
        log.exception("no se pudieron evaluar las señales para el digest")
        return []


def _notify_error(context: str, detail: str) -> None:
    """Best-effort Telegram alert on internal errors."""
    try:
        from src.notifier.telegram import TelegramNotifier, TelegramConfig
        TelegramNotifier(TelegramConfig.from_env()).send_error(context, detail)
    except Exception:
        pass


# ---------- digest de jornada ----------

def _maybe_send_jornada_digest(fixtures: dict) -> None:
    """Al cierre de cada día manda UN digest: resultados de la jornada + banderas post-jornada.
    Idempotente vía marcador data/digests/{fecha}.txt (un envío por jornada)."""
    import json
    from datetime import datetime, timezone, timedelta
    from src.agent.postmortem import _data_dir
    from src.agent.alerts import build_digest_text, check_llm_backfired, check_pool_slippage
    from src.utils.env import get_int_list

    now = datetime.now(timezone.utc)
    data_dir = _data_dir()
    digest_dir = data_dir / "digests"
    pm_dir = data_dir / "postmortems"
    digest_dir.mkdir(parents=True, exist_ok=True)

    all_matches = (fixtures.get("fase_grupos") or []) + (fixtures.get("eliminatorias") or [])
    by_date: dict[str, list] = {}
    for m in all_matches:
        ko = m.get("kickoff_utc")
        if not ko:
            continue
        try:
            dt = datetime.fromisoformat(str(ko).replace("Z", "+00:00"))
        except Exception:
            continue
        uy_date = (dt - timedelta(hours=3)).date().isoformat()  # fecha de calendario UY
        by_date.setdefault(uy_date, []).append((m, dt))

    my_ids = set(get_int_list("PENCA_IDS"))
    for uy_date, items in sorted(by_date.items()):
        marker = digest_dir / f"{uy_date}.txt"
        if marker.exists():
            continue
        last_ko = max(dt for _, dt in items)
        if now < last_ko + timedelta(minutes=150):
            continue  # la jornada todavía no terminó

        reports = []
        for m, _dt in items:
            p = pm_dir / f"{m['id']}.json"
            if p.exists():
                try:
                    reports.append(json.loads(p.read_text()))
                except Exception:
                    pass
        if not reports:
            marker.write_text("sin-reportes")  # marcar para no reintentar eternamente
            continue

        summaries = [{
            "label": f"{r.get('home_team','?')} vs {r.get('away_team','?')}",
            "final": r.get("final_score", "?"),
            "best_pts": r.get("portfolio_max_points", 0),
            "best_score": _best_pick_str(r),
        } for r in reports]

        flags = [f for f in (check_llm_backfired(r) for r in reports) if f]
        prev_e, curr_e = _last_two_snapshots(data_dir, json)
        slip = check_pool_slippage(prev_e, curr_e, my_ids)
        if slip:
            flags.append(slip)
        flags.extend(_strategy_signal_flags())  # señales del dashboard en alerta/vigilar
        pool_line = _digest_pool_line(curr_e, my_ids)

        text = build_digest_text(uy_date, summaries, flags, pool_line)
        try:
            from src.notifier.telegram import TelegramNotifier, TelegramConfig
            TelegramNotifier(TelegramConfig.from_env()).send(text)
            marker.write_text(now.isoformat())
            log.info("digest de jornada enviado | fecha=%s | %d partidos %d banderas",
                     uy_date, len(reports), len(flags))
        except Exception:
            log.exception("envío de digest falló | fecha=%s", uy_date)


def _best_pick_str(report: dict) -> str:
    rs = report.get("pencas_results") or []
    if not rs:
        return "?"
    best = max(rs, key=lambda r: r.get("points_earned", 0))
    sc = best.get("predicted_score") or [None, None]
    return f"{sc[0]}-{sc[1]}"


def _last_two_snapshots(data_dir, json):
    """(entries_previo, entries_actual) de los dos snapshots más recientes del pool."""
    sdir = data_dir / "pool_snapshots"
    if not sdir.exists():
        return None, None
    snaps = []
    for f in sdir.glob("*.json"):
        try:
            snaps.append(json.loads(f.read_text()))
        except Exception:
            pass
    snaps.sort(key=lambda s: s.get("taken_at") or "")
    if not snaps:
        return None, None
    curr = snaps[-1].get("entries")
    prev = snaps[-2].get("entries") if len(snaps) >= 2 else None
    return prev, curr


def _digest_pool_line(curr_entries, my_ids):
    if not curr_entries:
        return None
    s = sorted(curr_entries, key=lambda e: -int(e.get("points_total", 0)))
    rank = {int(e.get("penca_id", 0)): i + 1 for i, e in enumerate(s)}
    mine = [e for e in curr_entries if int(e.get("penca_id", 0)) in my_ids]
    if not mine:
        return None
    best = max(mine, key=lambda e: int(e.get("points_total", 0)))
    pos = rank.get(int(best.get("penca_id", 0)))
    return f"🏆 Mejor penca: {pos}°/{len(s)} del pool · {int(best.get('points_total', 0))} pts"


if __name__ == "__main__":
    raise SystemExit(main())
