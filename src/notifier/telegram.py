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
TELEGRAM_PIN = "https://api.telegram.org/bot{token}/pinChatMessage"
TELEGRAM_UNPIN = "https://api.telegram.org/bot{token}/unpinChatMessage"


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

    def send(self, text: str, parse_mode: Literal["HTML", "MarkdownV2"] = "HTML") -> int:
        """Envía un mensaje y devuelve el message_id."""
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
        return int(resp.json()["result"]["message_id"])

    def pin_message(self, message_id: int, disable_notification: bool = True) -> None:
        """Pinea un mensaje en el chat."""
        url = TELEGRAM_PIN.format(token=self.config.bot_token)
        resp = self._client.post(url, json={
            "chat_id": self.config.chat_id,
            "message_id": message_id,
            "disable_notification": disable_notification,
        })
        if resp.status_code >= 400:
            raise RuntimeError(f"Telegram pinChatMessage falló: {resp.status_code} — {resp.text}")

    def unpin_message(self, message_id: int) -> None:
        """Despinea un mensaje específico. No-throws si el mensaje ya no existe."""
        url = TELEGRAM_UNPIN.format(token=self.config.bot_token)
        try:
            self._client.post(url, json={
                "chat_id": self.config.chat_id,
                "message_id": message_id,
            })
        except Exception:
            pass   # best-effort

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

    def send_heartbeat(self, status: dict) -> int:
        """Heartbeat diario con secciones bien separadas.

        status: dict con keys:
            now_uy, next_match, predictions_24h, predictions_total,
            scheduler_status, last_scheduler_run, api_penca_status,
            pinnacle_status, anthropic_status, disk_free, ram_used,
            errors_24h, dry_run, do_mtd, anthropic_total, anthropic_24h,
            anthropic_calls_total.
        """
        ok_em, warn_em, err_em = "✅", "⚠️", "❌"

        def icon(s: str) -> str:
            s = (s or "").lower()
            if "ok" in s or "200" in s or "active" in s or "success" in s:
                return ok_em
            if "err" in s or "fail" in s or "404" in s or "500" in s or "inactive" in s:
                return err_em
            return warn_em

        # Resumen global: si TODO está OK, una sola línea de estado feliz.
        components_icons = [
            icon(status[k]) for k in (
                "scheduler_status", "api_penca_status", "pinnacle_status", "anthropic_status"
            )
        ]
        all_ok = all(i == ok_em for i in components_icons)
        global_emoji = ok_em if all_ok else (err_em if err_em in components_icons else warn_em)
        global_label = "Todo OK" if all_ok else (
            "Algo falla" if err_em in components_icons else "Algo a chequear"
        )

        sep = "━━━━━━━━━━━━━━━"

        # ── Header
        header = (
            f"💓 <b>Heartbeat</b>  ·  {_esc(status['now_uy'])}\n"
            f"{global_emoji} <b>{global_label}</b>  ·  📅 {_esc(status['next_match'])}"
        )

        # ── Componentes
        comps = (
            f"{sep}\n"
            f"<b>🔧 Componentes</b>\n"
            f"  {icon(status['scheduler_status'])}  Scheduler  ·  {_esc(status['scheduler_status'])}\n"
            f"  {icon(status['api_penca_status'])}  API Penca  ·  {_esc(status['api_penca_status'])}\n"
            f"  {icon(status['pinnacle_status'])}  Pinnacle  ·  {_esc(status['pinnacle_status'])}\n"
            f"  {icon(status['anthropic_status'])}  Anthropic  ·  {_esc(status['anthropic_status'])}"
        )

        # ── VPS
        vps_lines = [
            f"{sep}",
            f"<b>💻 VPS</b>",
            f"  💾 Disco  ·  {_esc(status['disk_free'])}",
            f"  🧠 RAM  ·  {_esc(status['ram_used'])}",
            f"  📊 Predicciones  ·  {status['predictions_total']} total  ·  {status['predictions_24h']} en 24h",
            f"  🐛 Errores 24h  ·  {status['errors_24h']}",
        ]
        if status.get("last_scheduler_run"):
            vps_lines.append(f"  ⏱  Último scheduler  ·  {_esc(status['last_scheduler_run'])}")
        vps = "\n".join(vps_lines)

        # ── Gastos
        do_mtd = status.get("do_mtd", "—")
        anth_total = status.get("anthropic_total", "—")
        anth_24h = status.get("anthropic_24h", "—")
        anth_calls = status.get("anthropic_calls_total", 0)
        gastos = (
            f"{sep}\n"
            f"<b>💵 Gastos</b>\n"
            f"  🌊 DigitalOcean (mes)  ·  {_esc(do_mtd)}\n"
            f"  🧠 Anthropic (total)  ·  {_esc(anth_total)}  ({anth_calls} calls)\n"
            f"  🧠 Anthropic (24h)  ·  {_esc(anth_24h)}"
        )

        # ── Footer (dry-run warning si aplica)
        footer = ""
        if status.get("dry_run"):
            footer = f"\n{sep}\n⚠️ <b>DRY_RUN activo</b>  ·  no publica a la penca"

        text = "\n\n".join([header, comps, vps, gastos]) + footer
        return self.send(text)


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
