"""Formateo y envío de mensajes Telegram del sistema valuebet.

Usa TelegramNotifier del sistema penca (solo la clase, no sus formatters).
Si VALUEBET_TELEGRAM_CHAT_ID está seteado, va a ese chat; si no, al mismo de penca.
"""

from __future__ import annotations

import html
import os

from src.notifier.telegram import TelegramConfig, TelegramNotifier
from src.valuebet.types import Suggestion


def get_notifier() -> TelegramNotifier:
    cfg = TelegramConfig.from_env()
    vb_chat = os.environ.get("VALUEBET_TELEGRAM_CHAT_ID")
    if vb_chat:
        cfg = TelegramConfig(bot_token=cfg.bot_token, chat_id=vb_chat)
    return TelegramNotifier(cfg)


def _e(s: str) -> str:
    return html.escape(str(s), quote=False)


def format_suggestion(s: Suggestion) -> str:
    mode_tag = "🧪 [PAPER]" if s.mode == "paper" else "💰 [REAL]"
    kind = f"COMBINADA x{len(s.legs)}" if s.is_parlay else "SIMPLE"
    lines = [f"<b>{mode_tag} Value bet — {kind}</b>  <code>{s.id}</code>", ""]

    for leg in s.legs:
        q = leg["quote"]
        lines += [
            f"⚽️ <b>{_e(q['event_name'])}</b>  ({_e(q['league'])}, {q['sport']})",
            f"   {_e(q['market'])} → <b>{_e(q['outcome'])}</b>",
            f"   Cuota SM: <b>{q['decimal_odds']:.2f}</b> · justa: {1.0 / leg['fair_prob']:.2f} "
            f"(p={leg['fair_prob']:.1%}) · edge leg: {leg['edge']:+.1%}",
            f"   🕐 {_e(q['start_utc'])}",
            "",
        ]

    lines += [
        f"📈 Edge total: <b>{s.edge:+.1%}</b>",
        f"💵 Stake sugerido: <b>{s.stake_suggested:.0f} UYU</b> "
        f"({s.stake_suggested / s.bankroll_at_ts:.1%} del bankroll {s.bankroll_at_ts:.0f})",
    ]
    if s.is_parlay:
        lines += [
            f"🎲 Cuota combinada {s.combined_odds:.2f} · P(acierto) = <b>{s.combined_fair_prob:.1%}</b>",
            "⚠️ Varianza alta: lo más probable es PERDER esta apuesta; es +EV a largo plazo.",
        ]
    return "\n".join(lines)


def format_settlement(s: Suggestion) -> str:
    emoji = {"won": "✅", "lost": "❌", "void": "↩️", "push": "↩️"}[s.status]
    return (
        f"{emoji} <code>{s.id}</code> {s.status.upper()} · PnL: <b>{s.pnl:+.0f} UYU</b>"
        + (f" · CLV: {s.clv:+.1%}" if s.clv is not None else "")
    )


def format_report(stats: dict) -> str:
    lines = [
        "<b>📊 Valuebet — reporte</b>",
        f"Bankroll: <b>{stats['bankroll']:.0f} UYU</b>",
        f"Sugerencias: {stats['n_total']} ({stats['n_open']} abiertas) · modo {stats['mode']}",
        f"PnL asentado: <b>{stats['pnl_total']:+.0f} UYU</b> · ROI: {stats['roi']:+.1%}"
        if stats["n_settled"] else "Sin apuestas asentadas todavía",
        f"CLV medio: <b>{stats['clv_mean']:+.2%}</b> sobre {stats['n_clv']} cerradas"
        if stats["n_clv"] else "Sin CLV computado todavía",
        "",
    ]
    for seg, d in sorted(stats.get("segments", {}).items()):
        status = "⛔" if d["multiplier"] == 0 else f"×{d['multiplier']:.2f}"
        lines.append(f"  <code>{seg}</code>: n={d['n']:.0f} clv={d.get('clv_mean', 0):+.2%} {status}")
    if stats.get("promotion_ready"):
        lines += ["", "🚀 <b>Criterio de promoción paper→real CUMPLIDO.</b> "
                      "Cambiar mode: real en config/valuebet.yaml si querés arrancar en serio."]
    return "\n".join(lines)


def format_error(context: str, error: Exception) -> str:
    return f"🛑 valuebet/{context}: {type(error).__name__}: {error}"
