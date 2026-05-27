"""Notifier por Telegram bot.

Tipos de mensajes:
- t24h_picks: las 5 picks recién generadas, 24h antes del partido.
- t3h_diff: cambios respecto a v1 si hubo updates (alineaciones, lesiones).
- t30min_lockin: confirmación final de las 5 picks publicadas.
- error: alertas (scraper caído, no se pudo publicar, etc).

Usa Markdown V2 para formato — escapar caracteres reservados con `_escape_md`.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Literal

import httpx


TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"

# Caracteres que MarkdownV2 reserva (https://core.telegram.org/bots/api#markdownv2-style)
_MD_RESERVED = re.compile(r"([_*\[\]()~`>#+\-=|{}.!\\])")


def _escape_md(text: str) -> str:
    return _MD_RESERVED.sub(r"\\\1", text)


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

    def send(self, text: str, parse_mode: Literal["MarkdownV2", "HTML"] = "MarkdownV2") -> None:
        url = TELEGRAM_API.format(token=self.config.bot_token)
        resp = self._client.post(url, json={
            "chat_id": self.config.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        })
        resp.raise_for_status()

    # ---------- plantillas ----------

    def send_t24h_picks(
        self,
        match_label: str,
        kickoff_local: str,
        picks: Iterable[dict],
        model_summary: dict,
    ) -> None:
        """Notifica los 5 picks generados a T-24h.

        picks: iterable de dicts con keys (penca_index, objective, score, e_points, uplift, variance, pool_popularity).
        model_summary: dict con (p_home, p_draw, p_away, e_goals_L, e_goals_V, market_chalk_score).
        """
        lines = [
            f"🔮 *T\\-24h* — {_escape_md(match_label)}",
            f"⏰ {_escape_md(kickoff_local)}",
            "",
            "*Modelo:*",
            f"  P\\(local\\)={_escape_md(f'{model_summary[\"p_home\"]:.0%}')}  "
            f"P\\(empate\\)={_escape_md(f'{model_summary[\"p_draw\"]:.0%}')}  "
            f"P\\(visit\\)={_escape_md(f'{model_summary[\"p_away\"]:.0%}')}",
            f"  E\\[goles\\]: {_escape_md(f'{model_summary[\"e_goals_L\"]:.1f}')} \\- {_escape_md(f'{model_summary[\"e_goals_V\"]:.1f}')}",
            "",
            "*Picks:*",
        ]
        for p in picks:
            gL, gV = p["score"]
            lines.append(
                f"  `P{p['penca_index']}` "
                f"\\[{_escape_md(p['objective'])}\\] "
                f"*{gL}\\-{gV}*  "
                f"E\\[pts\\]={_escape_md(f'{p[\"e_points\"]:.2f}')}  "
                f"uplift={_escape_md(f'{p[\"uplift\"]:+.2f}')}"
            )

        self.send("\n".join(lines))

    def send_diff(
        self,
        match_label: str,
        phase: str,
        changes: list[dict],
    ) -> None:
        """Notifica cambios entre versiones (v1 → v2 a T-3h, o v2 → v_final a T-30min).

        changes: lista de {penca_index, old_score, new_score, reason}.
        """
        if not changes:
            return  # silencio si nada cambió
        lines = [
            f"🔁 *{_escape_md(phase)}* — {_escape_md(match_label)}",
            "",
            "*Cambios:*",
        ]
        for c in changes:
            old_g = f"{c['old_score'][0]}\\-{c['old_score'][1]}"
            new_g = f"{c['new_score'][0]}\\-{c['new_score'][1]}"
            lines.append(
                f"  `P{c['penca_index']}`: {old_g} → *{new_g}*  _{_escape_md(c['reason'])}_"
            )
        self.send("\n".join(lines))

    def send_lockin(self, match_label: str, picks: Iterable[dict]) -> None:
        """Notifica el lock-in final a T-30min con confirmación de publicación."""
        lines = [
            f"🔒 *LOCK\\-IN* — {_escape_md(match_label)}",
            "",
            "*Picks publicadas:*",
        ]
        for p in picks:
            gL, gV = p["score"]
            lines.append(f"  `P{p['penca_index']}`  *{gL}\\-{gV}*")
        self.send("\n".join(lines))

    def send_error(self, context: str, error: str) -> None:
        """Alerta de error. Sin formato — texto plano para garantizar entrega."""
        text = f"❌ ERROR ({context})\n\n{error[:1500]}"
        # texto plano: deshabilitar parse_mode
        url = TELEGRAM_API.format(token=self.config.bot_token)
        self._client.post(url, json={
            "chat_id": self.config.chat_id,
            "text": text,
            "disable_web_page_preview": True,
        }).raise_for_status()


def send_hello() -> None:
    """Smoke check para verificar que bot/chat_id están configurados correctamente."""
    notif = TelegramNotifier(TelegramConfig.from_env())
    notif.send(f"✅ Penca Mundial 2026 \\- conectado a las {_escape_md(datetime.now().strftime('%H:%M:%S'))}")


if __name__ == "__main__":
    send_hello()
