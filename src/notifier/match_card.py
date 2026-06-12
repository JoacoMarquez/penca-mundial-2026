"""Tarjeta de Telegram por partido: UN mensaje que se edita en cada fase.

Flujo: T-24h crea la tarjeta (única notificación). T-3h, T-30min y el postmortem
la EDITAN en silencio. Todo el ciclo de vida del partido vive en un solo mensaje,
sin entreverarse con otros partidos.

El diff entre pasadas se muestra a nivel PORTFOLIO (qué marcadores entran/salen de
la cobertura), no por penca: los enroques por ranking — que generaban 11 líneas de
ruido — se resumen en una sola nota.

Estado: data/telegram_cards.json  →  {match_id: message_id}
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.notifier.telegram import HUMAN_OBJECTIVE_LABELS, TelegramNotifier, _esc

log = logging.getLogger(__name__)

PHASE_LABELS = {"T_24h": "T-24h", "T_3h": "T-3h", "T_30min": "T-30min"}


def _data_dir() -> Path:
    return Path(os.environ.get("DATA_DIR", "data"))


def _cards_path() -> Path:
    return _data_dir() / "telegram_cards.json"


def _load_cards() -> dict[str, int]:
    try:
        return json.loads(_cards_path().read_text())
    except Exception:
        return {}


def _save_cards(cards: dict[str, int]) -> None:
    try:
        _cards_path().parent.mkdir(parents=True, exist_ok=True)
        _cards_path().write_text(json.dumps(cards))
    except Exception as e:
        log.warning("no pude guardar telegram_cards.json: %s", e)


# ---------- helpers de contenido ----------

def _hhmm_uy(iso: str) -> str:
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return (dt - timedelta(hours=3)).strftime("%H:%M")
    except Exception:
        return "?"


def _exposure_of(version: dict) -> dict[str, int]:
    meta = version.get("assignment_meta") or {}
    exp = meta.get("exposure")
    if exp:
        return dict(exp)
    assignment = version.get("assignment") or []
    return dict(Counter(f'{a["score"][0]}-{a["score"][1]}' for a in assignment))


def _portfolio_diff_lines(versions: list[dict]) -> list[str]:
    """Diff de cobertura entre las dos últimas pasadas, a nivel portfolio."""
    if len(versions) < 2:
        return []
    prev, cur = versions[-2], versions[-1]
    e_prev, e_cur = _exposure_of(prev), _exposure_of(cur)
    removed = sorted(set(e_prev) - set(e_cur))
    added = sorted(set(e_cur) - set(e_prev))
    moved = {
        s: (e_prev[s], e_cur[s])
        for s in set(e_prev) & set(e_cur)
        if e_prev[s] != e_cur[s]
    }

    prev_by_pid = {a["penca_id"]: tuple(a["score"]) for a in (prev.get("assignment") or [])}
    cur_by_pid = {a["penca_id"]: tuple(a["score"]) for a in (cur.get("assignment") or [])}
    n_reshuffled = sum(
        1 for pid in cur_by_pid if pid in prev_by_pid and prev_by_pid[pid] != cur_by_pid[pid]
    )

    lines = []
    parts = []
    if removed:
        parts.append("−" + " −".join(removed))
    if added:
        parts.append("+" + " +".join(added))
    if moved:
        parts.append(" ".join(f"{s} ×{a}→×{b}" for s, (a, b) in sorted(moved.items())))
    label = PHASE_LABELS.get(cur.get("phase", ""), cur.get("phase", "?"))
    if parts:
        lines.append(f"🔁 <b>Cambios {label}:</b> {_esc(' · '.join(parts))}")
        if n_reshuffled:
            lines.append(f"   <i>(+{n_reshuffled} pencas reasignadas por ranking, cobertura igual)</i>")
    elif n_reshuffled:
        lines.append(
            f"🔁 <b>{label}:</b> <i>cobertura sin cambios — {n_reshuffled} pencas "
            "reasignadas por ranking</i>"
        )
    else:
        lines.append(f"🔁 <b>{label}:</b> <i>sin cambios</i>")
    return lines


def build_match_card(
    match_label: str,
    kickoff_local: str,
    versions: list[dict],
    postmortem: dict | None = None,
) -> str:
    """Renderiza la tarjeta completa desde las versiones persistidas (orden cronológico)."""
    latest = versions[-1]
    c = latest.get("constraints") or {}
    lines: list[str] = [f"⚽ <b>{_esc(match_label)}</b>", f"📅 {_esc(kickoff_local)}"]

    # Resultado (cuando hay postmortem) — arriba de todo
    if postmortem:
        ah, av = postmortem.get("actual_home"), postmortem.get("actual_away")
        best = max(
            (postmortem.get("pencas_results") or []),
            key=lambda r: r.get("points_earned", 0),
            default=None,
        )
        lines.append("")
        lines.append(f"🏁 <b>RESULTADO: {ah}-{av}</b>")
        if best:
            star = " ⭐ exacto" if best.get("is_exact") else ""
            lines.append(
                f"   Mejor: penca {best['penca_id']} "
                f"({best['predicted_score'][0]}-{best['predicted_score'][1]}) "
                f"+{best['points_earned']}pts{star}"
            )
        tot = postmortem.get("portfolio_total_points")
        rank = postmortem.get("our_best_rank_in_pool")
        extra = []
        if tot is not None:
            extra.append(f"portfolio {tot}pts")
        if rank is not None:
            extra.append(f"mejor rank pool: #{rank}")
        if extra:
            lines.append(f"   {_esc(' · '.join(extra))}")

    # Mercado + LLM (última pasada)
    if c:
        lines.append("")
        lines.append(
            f"📊 {c.get('p_home', 0)*100:.0f}% / {c.get('p_draw', 0)*100:.0f}% / "
            f"{c.get('p_away', 0)*100:.0f}% · goles {c.get('e_goals_L', 0):.2f}–{c.get('e_goals_V', 0):.2f}"
        )
    qa = latest.get("qualitative_adjustment") or {}
    if qa and (abs(qa.get("delta_lambda_L", 0)) > 1e-6 or abs(qa.get("delta_lambda_V", 0)) > 1e-6):
        lines.append(
            f"🧠 LLM: δL {qa['delta_lambda_L']:+.2f} δV {qa['delta_lambda_V']:+.2f} "
            f"(conf {qa.get('confidence', 0):.2f}) — {_esc((qa.get('reasoning') or '')[:150])}"
        )
    elif qa:
        lines.append("🧠 LLM: sin ajuste (mercado eficiente)")

    # Exposición
    exposure = _exposure_of(latest)
    if exposure:
        total = sum(exposure.values())
        lines.append("")
        lines.append(f"📋 <b>Exposición ({total} pencas):</b>")
        for score, count in sorted(exposure.items(), key=lambda kv: (-kv[1], kv[0])):
            bar = "█" * count
            lines.append(f"  <code>{score}</code> ×{count} {bar}")

    # Diff de la última pasada
    if not postmortem:
        diff_lines = _portfolio_diff_lines(versions)
        if diff_lines:
            lines.append("")
            lines.extend(diff_lines)

    # Timeline de fases
    lines.append("")
    phase_bits = []
    for v in versions:
        label = PHASE_LABELS.get(v.get("phase", ""), v.get("phase", "?"))
        phase_bits.append(f"{label} ✓ {_hhmm_uy(v.get('run_at', ''))}")
    lines.append("⏱ " + " · ".join(phase_bits))
    if latest.get("published"):
        lines.append(f"🔒 Publicado ({sum(_exposure_of(latest).values())} picks)")

    return "\n".join(lines)


# ---------- upsert ----------

def upsert_match_card(notifier: TelegramNotifier, match_id: str | int, text: str) -> None:
    """Crea la tarjeta si no existe (notifica); si existe, la edita (silencioso).

    Si la edición falla (mensaje borrado/expirado), manda una nueva y re-vincula.
    """
    mid = str(match_id)
    cards = _load_cards()
    msg_id = cards.get(mid)
    if msg_id:
        try:
            if notifier.edit(msg_id, text):
                return
        except Exception as e:
            log.warning("edit de tarjeta %s falló: %s — mando nueva", mid, e)
    new_id = notifier.send(text)
    cards[mid] = new_id
    _save_cards(cards)


def load_versions(match_id: str | int) -> list[dict]:
    """Versiones de predicción persistidas, en orden cronológico."""
    from src.utils.versions import sort_versions
    pdir = _data_dir() / "predictions" / str(match_id)
    if not pdir.exists():
        return []
    out = []
    for f in sort_versions(pdir.glob("v*_*.json")):
        try:
            out.append(json.loads(f.read_text()))
        except Exception:
            continue
    return out
