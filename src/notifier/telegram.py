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

    # ---------- mensajes por fase del partido ----------

    def send_t24h_picks(
        self,
        match_label: str,
        kickoff_local: str,
        picks: Iterable[dict],
        model_summary: dict,
        qualitative: dict | None = None,
        phase_label: str = "T-24h",
        assignment_meta: dict | None = None,
        tipster_consensus: dict | None = None,
        dossier_summary: str | None = None,
    ) -> int:
        """Primer aviso del partido, 24h antes (o etiqueta personalizable)."""
        fav_pct, fav_side = _favorite(model_summary)
        match_label_e = _esc(match_label)
        ko_e = _esc(kickoff_local)
        local_team = match_label.split(" vs ")[0] if " vs " in match_label else "Local"
        away_team = match_label.split(" vs ")[1] if " vs " in match_label else "Visit"
        fav_team = local_team if fav_side == "Local" else (
            away_team if fav_side == "Visit" else "Empate"
        )
        fav_line = f"{_esc(fav_team)} favorito {fav_pct}" if fav_side != "Empate" else f"empate parejo {fav_pct}"

        # Header
        header = (
            f"⚽ <b>{match_label_e}</b>\n"
            f"⏰ {ko_e}  ·  {fav_line}\n"
            f"📈 E[goles]: {model_summary['e_goals_L']:.1f} — {model_summary['e_goals_V']:.1f}"
        )

        # Picks con labels humanos + indicador de rank si hay asignación adaptativa
        rank_emoji = {1: "🥇", 2: "🥈", 3: "🥉", 4: "4°", 5: "5°"}

        picks_lines = []
        has_assignment = any(p.get("assigned_penca_id") is not None for p in picks)
        for p in picks:
            gL, gV = p["score"]
            emoji, label = HUMAN_OBJECTIVE_LABELS.get(p["objective"], ("🔸", p["objective"].title()))
            p_score = p.get("p_scoreline", 0.0) * 100

            if has_assignment and p.get("assigned_penca_id"):
                rank = p.get("assigned_rank") or 0
                rmark = rank_emoji.get(rank, str(rank))
                penca_str = f"P{p['penca_index']}→{p['assigned_penca_id']}"
                picks_lines.append(
                    f"  {rmark} {emoji} {label:<12} <b>{gL}-{gV}</b>  ({p_score:.0f}%)  <i>{_esc(penca_str)}</i>"
                )
            else:
                picks_lines.append(
                    f"  <b>P{p['penca_index']}</b> {emoji} {label:<12} <b>{gL}-{gV}</b>  ({p_score:.0f}%)"
                )

        if assignment_meta and assignment_meta.get("objective") == "p_top_k":
            p_value = assignment_meta.get("p_top_k_value", 0.0)
            threshold = assignment_meta.get("threshold", 0)
            header_title = f"🎯 Picks  <i>(maximizando P(top-3): {p_value:.1%}, cutoff {threshold} pts)</i>"
        elif assignment_meta and assignment_meta.get("objective", "").startswith("e_max"):
            header_title = "🎯 Picks  <i>(maximizando E[max] — sin data de pool aún)</i>"
        elif has_assignment:
            header_title = "🎯 Picks  <i>(asignación adaptativa por rank)</i>"
        else:
            header_title = "🎯 Picks"
        picks_block = f"<b>{header_title}</b>\n" + "\n".join(picks_lines)

        sep = "━━━━━━━━━━━━━━━"
        parts = [header, sep, picks_block]

        # Exposición real por marcador (N pencas con repetición)
        exposure_block = _exposure_block(assignment_meta)
        if exposure_block:
            parts.append(sep)
            parts.append(exposure_block)

        # Ficha del partido (dossier resumido)
        if dossier_summary:
            parts.append(sep)
            parts.append(f"<b>📋 Ficha del partido</b>\n{dossier_summary}")

        # Consenso de tipsters (si encontró alguno)
        if tipster_consensus and tipster_consensus.get("n_tipsters_picked", 0) > 0:
            n = tipster_consensus["n_tipsters_picked"]
            p1 = tipster_consensus.get("consensus_p_1", 0) * 100
            pX = tipster_consensus.get("consensus_p_X", 0) * 100
            p2 = tipster_consensus.get("consensus_p_2", 0) * 100
            tips_lines = [
                f"<b>🗞️ Tipsters ({n} pronosticadores)</b>",
                f"  Local {p1:.0f}%  ·  Empate {pX:.0f}%  ·  Visit {p2:.0f}%",
            ]
            edge = tipster_consensus.get("info_edge_vs_market")
            if edge:
                delta = edge.get("delta_pp", 0)
                direction = edge.get("direction", "")
                tips_lines.append(f"  ⚠️ <i>Info edge: {delta:+.1f}pp vs mercado — {_esc(direction)}</i>")
            parts.append(sep)
            parts.append("\n".join(tips_lines))

        # Análisis LLM (si hay)
        if qualitative and qualitative.get("reasoning"):
            d_l = qualitative.get("delta_lambda_L", 0.0)
            d_v = qualitative.get("delta_lambda_V", 0.0)
            conf = qualitative.get("confidence", 0.0)
            reasoning_e = _esc(qualitative["reasoning"])
            llm_block = (
                f"<b>🧠 Análisis LLM</b>\n"
                f"{reasoning_e}\n"
                f"<i>Ajuste λ: {d_l:+.2f} / {d_v:+.2f}  ·  confianza {conf:.2f}</i>"
            )
            parts.append(sep)
            parts.append(llm_block)

        text = "\n\n".join(parts)
        return self.send(text)

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
            f"🔁 <b>{_esc(phase)}</b>",
            f"⚽ {_esc(match_label)}",
            "",
            "<b>Cambios:</b>",
        ]
        for c in changes:
            old = f"{c['old_score'][0]}-{c['old_score'][1]}"
            new_ = f"{c['new_score'][0]}-{c['new_score'][1]}"
            reason = _esc(c.get("reason", ""))
            lines.append(f"  P{c['penca_index']}: {old} → <b>{new_}</b>  <i>{reason}</i>")
        self.send("\n".join(lines))

    def send_lockin(
        self,
        match_label: str,
        picks: Iterable[dict],
        assignment_meta: dict | None = None,
    ) -> None:
        """Confirmación final cuando se publica.

        Con N pencas muestra la exposición por marcador; si no hay exposición (fallback),
        lista el menú de objetivos canónicos.
        """
        exposure_block = _exposure_block(assignment_meta)
        if exposure_block:
            total = sum((assignment_meta or {}).get("exposure", {}).values())
            text = (
                f"🔒 <b>LOCK-IN — {total} predicciones publicadas</b>\n"
                f"⚽ {_esc(match_label)}\n\n"
                f"{exposure_block}"
            )
            self.send(text)
            return
        picks_lines = []
        for p in picks:
            gL, gV = p["score"]
            emoji, label = HUMAN_OBJECTIVE_LABELS.get(p["objective"], ("🔸", p["objective"].title()))
            picks_lines.append(f"  <b>P{p['penca_index']}</b> {emoji} {label:<12} <b>{gL}-{gV}</b>")
        text = (
            f"🔒 <b>LOCK-IN — Predicciones publicadas</b>\n"
            f"⚽ {_esc(match_label)}\n"
            "\n" + "\n".join(picks_lines)
        )
        self.send(text)

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


def _exposure_block(assignment_meta: dict | None) -> str | None:
    """Renderiza la exposición por marcador (cuántas pencas juegan cada scoreline).

    Es la vista correcta cuando hay N pencas con repetición: en vez de listar N filas,
    muestra cuántas pencas caen en cada marcador, ordenado por frecuencia.
    """
    if not assignment_meta:
        return None
    exposure = assignment_meta.get("exposure")
    if not exposure:
        return None
    total = sum(exposure.values())
    items = sorted(exposure.items(), key=lambda kv: (-kv[1], kv[0]))
    lines = [f"<b>📋 Exposición — {total} pencas</b>"]
    for score, count in items:
        bar = "▰" * min(count, 12)
        lines.append(f"  <b>{_esc(score)}</b>  ×{count}  {bar}")
    return "\n".join(lines)


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
