"""Notifier por Telegram bot — formato conciso, HTML mode.

Tipos de mensajes:
- alert:       banderas rojas a revisar antes del partido.
- publish_ok:  confirmación mínima de que las picks se publicaron.
- postmortem:  resultado del partido + comparación vs pool.
- error:       alerta texto plano.
- heartbeat:   resumen diario del sistema.

Usa HTML parse mode (solo escapa &, <, >). Menos mess que MarkdownV2.
"""

from __future__ import annotations

import html
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import httpx


TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"
TELEGRAM_PIN = "https://api.telegram.org/bot{token}/pinChatMessage"
TELEGRAM_UNPIN = "https://api.telegram.org/bot{token}/unpinChatMessage"
TELEGRAM_EDIT = "https://api.telegram.org/bot{token}/editMessageText"


# Labels humanos para los 5 objetivos de strategy/portfolio.py
HUMAN_OBJECTIVE_LABELS: dict[str, tuple[str, str]] = {
    "ev":             ("🎯", "Favorito"),
    "differentiated": ("📊", "Diferencial"),
    "tail":           ("⚡", "Goles"),
    "upset":          ("😲", "Sorpresa"),
    "variance":       ("📈", "Varianza"),
}


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

    def edit(self, message_id: int, text: str, parse_mode: Literal["HTML", "MarkdownV2"] = "HTML") -> bool:
        """Edita un mensaje existente (silencioso — no genera notificación).

        Devuelve True si editó. False si el mensaje ya no es editable (muy viejo,
        borrado) — el caller decide si manda uno nuevo. "message is not modified"
        (texto idéntico) cuenta como éxito.
        """
        url = TELEGRAM_EDIT.format(token=self.config.bot_token)
        resp = self._client.post(url, json={
            "chat_id": self.config.chat_id,
            "message_id": message_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        })
        if resp.status_code < 400:
            return True
        if "message is not modified" in resp.text:
            return True
        return False

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

    def send_postmortem(self, report: Any) -> int:
        """Manda el postmortem de un partido finalizado."""
        actual = f"{report.actual_home}-{report.actual_away}"
        label = f"{report.home_team} vs {report.away_team}"

        # Header
        header = (
            f"📋 <b>POSTMORTEM</b>\n"
            f"⚽ {_esc(label)}  ·  <b>{_esc(actual)}</b>"
        )

        # Resultados por penca
        sep = "━━━━━━━━━━━━━━━"
        lines = [f"{sep}", "<b>🎯 Tus pencas</b>"]
        for r in report.pencas_results:
            emoji_pts = "🎉" if r.points_earned == 6 else ("✅" if r.points_earned >= 3 else ("〰️" if r.points_earned > 0 else "❌"))
            human_obj = HUMAN_OBJECTIVE_LABELS.get(r.strategy_used, ("•", r.strategy_used))[1]
            score_str = f"{r.predicted_score[0]}-{r.predicted_score[1]}"
            lines.append(
                f"  {emoji_pts} P{r.penca_id} {_esc(human_obj):<12}  pick {_esc(score_str)}  → <b>{r.points_earned} pts</b>"
            )

        # Pool comparison
        pool_lines = [sep, "<b>📊 vs Pool</b>"]
        if report.pool_top_points is not None:
            pool_lines.append(
                f"  Pool top: {report.pool_top_points}pts  ·  mediana: {report.pool_median_points:.0f}pts"
            )
        if report.our_best_rank_in_pool is not None:
            pool_lines.append(f"  Mejor penca en pos. <b>{report.our_best_rank_in_pool}°</b>")
        pool_lines.append(
            f"  Mejor penca: {report.portfolio_max_points} pts  ·  suma: {report.portfolio_total_points} pts"
        )

        # Insights
        insight_lines = []
        if report.insights:
            insight_lines = [sep, "<b>🧐 Análisis</b>"]
            for ins in report.insights[:5]:
                insight_lines.append(f"  · {ins}")

        text = "\n".join([header] + lines + pool_lines + insight_lines)
        return self.send(text)

    def send_error(self, context: str, error: str) -> None:
        """Texto plano (sin parse_mode) para garantizar entrega aun con caracteres raros."""
        text = f"❌ ERROR ({context})\n\n{error[:1500]}"
        url = TELEGRAM_API.format(token=self.config.bot_token)
        self._client.post(url, json={
            "chat_id": self.config.chat_id,
            "text": text,
            "disable_web_page_preview": True,
        }).raise_for_status()

    # ---------- avisos (banderas + confirmaciones) ----------

    def send_alert(self, match_label: str, flags_text: str) -> int:
        """Aviso de banderas rojas detectadas antes del partido (algo para revisar)."""
        text = f"🚩 <b>REVISAR</b>  ·  {_esc(match_label)}\n\n{flags_text}"
        return self.send(text)

    def send_publish_ok(self, match_label: str, n_picks: int) -> int:
        """Confirmación mínima de que las picks se publicaron (sin mostrar los pronósticos)."""
        return self.send(f"✅ Picks publicadas ({n_picks}) · {_esc(match_label)}")

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

        # ── VPS (recursos)
        vps_lines = [
            f"{sep}",
            f"<b>💻 VPS</b>",
            f"  💾 Disco  ·  {_esc(status['disk_free'])}",
            f"  🧠 RAM  ·  {_esc(status['ram_used'])}",
            f"  🐛 Errores 24h  ·  {status['errors_24h']}",
        ]
        if status.get("last_scheduler_run"):
            vps_lines.append(f"  ⏱  Último scheduler  ·  {_esc(status['last_scheduler_run'])}")
        vps = "\n".join(vps_lines)

        # ── Pipeline (actividad)
        gen_24h = status["predictions_24h"]
        gen_total = status["predictions_total"]
        expected_last = status.get("expected_last_24h", 0)
        expected_next = status.get("expected_next_24h", 0)
        next_pasada = status.get("next_pasada", "—")

        # Badge comparando lo que DEBERÍA haber corrido (último 24h) vs lo que efectivamente corrió
        if expected_last == 0 and gen_24h == 0:
            activity = "⏸  Sin actividad esperada en últimas 24h"
        elif gen_24h >= expected_last and expected_last > 0:
            activity = f"✅  Últimas 24h: {gen_24h}/{expected_last} pasadas ejecutadas"
        elif gen_24h < expected_last:
            activity = f"⚠️  Últimas 24h: solo {gen_24h}/{expected_last} pasadas (faltan {expected_last - gen_24h})"
        else:
            activity = f"ℹ️  Últimas 24h: {gen_24h} pasadas (sin esperadas)"

        # Próximas 24h: solo informativo
        if expected_next == 0:
            next_line = "  📭  Próximas 24h  ·  sin actividad programada"
        else:
            next_line = f"  📥  Próximas 24h  ·  {expected_next} pasadas programadas"

        pipeline_section = (
            f"{sep}\n"
            f"<b>📊 Pipeline</b>\n"
            f"  {activity}\n"
            f"{next_line}\n"
            f"  ⏭  Próxima pasada  ·  {_esc(next_pasada)}\n"
            f"  🗂  Total histórico  ·  {gen_total} pasadas"
        )

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

        text = "\n\n".join([header, comps, vps, pipeline_section, gastos]) + footer
        return self.send(text)


def send_hello() -> None:
    """Smoke check."""
    notif = TelegramNotifier(TelegramConfig.from_env())
    notif.send(f"✅ Penca Mundial 2026 — conectado {_esc(datetime.now().strftime('%H:%M:%S'))}")


if __name__ == "__main__":
    send_hello()
