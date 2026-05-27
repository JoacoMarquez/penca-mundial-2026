"""Notifier por Telegram bot — formato conciso, HTML mode.

Tipos de mensajes:
- t24h_picks:  primer mensaje del partido, con 5 picks + contexto breve.
- t3h_diff:    SOLO si las picks cambian en T-3h (silencio si todo igual).
- t30min_lockin: confirmación final cuando se publica.
- error:       alerta texto plano.
- heartbeat:   resumen diario del sistema.

Usa HTML parse mode (solo escapa &, <, >). Menos mess que MarkdownV2.
"""

from __future__ import annotations

import html
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Literal

import httpx


TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


def _esc(text: str | int | float) -> str:
    """Escape de HTML para Telegram (solo & < >)."""
    return html.escape(str(text), quote=False)


@dataclass(frozen=True)
class TelegramConfig:
    bot_token: str
    chat_id: str

    @classmethod
    def from_env(cls) -> "TelegramConfig":
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
        if not token or not chat_id:
            raise RuntimeError("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID no configurados en env")
        return cls(bot_token=token, chat_id=chat_id)


class TelegramNotifier:
    def __init__(self, config: TelegramConfig):
        self.config = config
        self._client = httpx.Client(timeout=10.0)

    def send(self, text: str, parse_mode: Literal["HTML", "MarkdownV2"] = "HTML") -> None:
        url = TELEGRAM_API.format(token=self.config.bot_token)
        resp = self._client.post(url, json={
            "chat_id": self.config.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        })
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Telegram sendMessage falló: {resp.status_code} — body={resp.text} — "
                f"preview={text[:200]!r}"
            )

    # ---------- mensajes por fase del partido ----------

    def send_t24h_picks(
        self,
        match_label: str,
        kickoff_local: str,
        picks: Iterable[dict],
        model_summary: dict,
    ) -> None:
        """Primer aviso del partido, 24h antes."""
        fav_pct, fav_side = _favorite(model_summary)
        match_label_e = _esc(match_label)
        ko_e = _esc(kickoff_local)
        fav_e = f"{fav_side} fav {fav_pct}"

        # Tabla de picks con alineación monospace
        picks_lines = []
        for p in picks:
            gL, gV = p["score"]
            obj = _esc(p["objective"][:8])    # truncar para que no rompa columnas
            picks_lines.append(f"  P{p['penca_index']} [{obj:<8}]  <b>{gL}-{gV}</b>")

        text = (
            f"⚽ <b>{match_label_e}</b>\n"
            f"⏰ {ko_e}  |  {_esc(fav_e)}\n"
            "\n"
            "<pre>" + "\n".join(picks_lines) + "</pre>\n"
            f"E[goles]: {model_summary['e_goals_L']:.1f} — {model_summary['e_goals_V']:.1f}"
        )
        self.send(text)

    def send_diff(
        self,
        match_label: str,
        phase: str,
        changes: list[dict],
    ) -> None:
        """Cambios entre versiones. NO se envía si no hay cambios."""
        if not changes:
            return
        lines = [
            f"🔁 <b>{_esc(phase)}</b> — {_esc(match_label)}",
            "",
        ]
        for c in changes:
            old = f"{c['old_score'][0]}-{c['old_score'][1]}"
            new_ = f"{c['new_score'][0]}-{c['new_score'][1]}"
            reason = _esc(c.get("reason", ""))
            lines.append(f"  P{c['penca_index']}: {old} → <b>{new_}</b>  <i>{reason}</i>")
        self.send("\n".join(lines))

    def send_lockin(self, match_label: str, picks: Iterable[dict]) -> None:
        """Confirmación final cuando se publica."""
        picks_lines = []
        for p in picks:
            gL, gV = p["score"]
            picks_lines.append(f"  P{p['penca_index']}  <b>{gL}-{gV}</b>")
        text = (
            f"🔒 <b>LOCK-IN</b> — {_esc(match_label)}\n"
            "\n" + "\n".join(picks_lines)
        )
        self.send(text)

    def send_error(self, context: str, error: str) -> None:
        """Texto plano (sin parse_mode) para garantizar entrega aun con caracteres raros."""
        text = f"❌ ERROR ({context})\n\n{error[:1500]}"
        url = TELEGRAM_API.format(token=self.config.bot_token)
        self._client.post(url, json={
            "chat_id": self.config.chat_id,
            "text": text,
            "disable_web_page_preview": True,
        }).raise_for_status()

    # ---------- heartbeat ----------

    def send_heartbeat(self, status: dict) -> None:
        """Heartbeat diario con info de todos los componentes.

        status: dict con keys:
            now_uy, next_match, predictions_24h, predictions_total,
            scheduler_status, last_scheduler_run, api_penca_status,
            pinnacle_status, anthropic_status, disk_free, ram_used,
            errors_24h, dry_run.
        """
        ok = "✅"
        warn = "⚠️"
        err = "❌"

        # ícono por componente
        def icon(s: str) -> str:
            s = (s or "").lower()
            if "ok" in s or "200" in s or "active" in s or "success" in s:
                return ok
            if "err" in s or "fail" in s or "404" in s or "500" in s or "inactive" in s:
                return err
            return warn

        dry_run_line = ""
        if status.get("dry_run"):
            dry_run_line = "\n⚠️ <b>DRY_RUN activo</b> — no publica a la penca"

        lines = [
            f"💓 <b>Heartbeat</b> · {_esc(status['now_uy'])}",
            f"📅 Próximo: {_esc(status['next_match'])}",
            "",
            "<b>Componentes:</b>",
            f"  {icon(status['scheduler_status'])} Scheduler: {_esc(status['scheduler_status'])}",
            f"  {icon(status['api_penca_status'])} API Penca: {_esc(status['api_penca_status'])}",
            f"  {icon(status['pinnacle_status'])} Pinnacle: {_esc(status['pinnacle_status'])}",
            f"  {icon(status['anthropic_status'])} Anthropic: {_esc(status['anthropic_status'])}",
            "",
            "<b>VPS:</b>",
            f"  💾 {_esc(status['disk_free'])}  |  🧠 {_esc(status['ram_used'])}",
            f"  📊 Predicciones: {status['predictions_total']} total · {status['predictions_24h']} en 24h",
            f"  🐛 Errores 24h: {status['errors_24h']}",
        ]
        if status.get("last_scheduler_run"):
            lines.append(f"  ⏱ Último scheduler: {_esc(status['last_scheduler_run'])}")
        lines.append(dry_run_line)

        text = "\n".join(l for l in lines if l)
        self.send(text)


def send_hello() -> None:
    """Smoke check."""
    notif = TelegramNotifier(TelegramConfig.from_env())
    notif.send(f"✅ Penca Mundial 2026 — conectado {_esc(datetime.now().strftime('%H:%M:%S'))}")


def _favorite(model_summary: dict) -> tuple[str, str]:
    """Identifica el favorito y devuelve (porcentaje, descripción)."""
    p_h = model_summary["p_home"]
    p_d = model_summary["p_draw"]
    p_a = model_summary["p_away"]
    m = max(p_h, p_d, p_a)
    if m == p_h:
        return f"{p_h:.0%}", "Local"
    if m == p_a:
        return f"{p_a:.0%}", "Visit"
    return f"{p_d:.0%}", "Empate"


if __name__ == "__main__":
    send_hello()
