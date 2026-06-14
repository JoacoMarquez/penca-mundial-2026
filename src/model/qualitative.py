"""Capa 4 del modelo: ajuste cualitativo con LLM.

Toma contexto pre-partido (lesiones, alineación probable, h2h, clima, árbitro, motivación)
y devuelve un ajuste delta a (λ_L, λ_V) acotado a ±0.3 con justificación.

El bound es un guardrail: el LLM no puede mover más de 0.3 goles esperados en cualquier dirección.
Eso impide que un sesgo del LLM domine al mercado (que ya es muy eficiente).

Costos esperados:
    Claude Sonnet 4.6 con ~1500 tokens input + 300 tokens output = ~$0.01 por partido.
    104 partidos máximo = ~$1 total. Trivial.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

log = logging.getLogger(__name__)


MAX_ABS_ADJUSTMENT = 0.30   # goles esperados — límite duro al ajuste del LLM


# ============ schema del input ============

@dataclass(frozen=True)
class MatchContext:
    """Contexto pre-partido que se le pasa al LLM."""
    home_team: str
    away_team: str
    kickoff_local: str       # "Sab 14/06 16:00 UY" (display, no estructural)
    stage: str               # "Fase de grupos" | "Octavos" | "Cuartos" | ...
    market_p_home: float
    market_p_draw: float
    market_p_away: float
    market_e_goals_L: float
    market_e_goals_V: float
    home_recent_form: str | None = None   # "ULLWW" o resumen de 5 últimos
    away_recent_form: str | None = None
    home_injuries: list[str] | None = None
    away_injuries: list[str] | None = None
    home_lineup_change: str | None = None
    away_lineup_change: str | None = None
    home_xi: list[str] | None = None      # once titular (nombres) si hay alineación disponible
    away_xi: list[str] | None = None
    lineup_confirmed: bool = False        # True solo si el XI es el oficial (no probable)
    h2h_recent: str | None = None         # texto libre sobre los últimos 5 enfrentamientos
    weather: str | None = None
    referee: str | None = None
    motivation_notes: str | None = None   # "El local ya clasificó, puede rotar" / "El visit necesita ganar"
    home_news_summary: str | None = None  # titulares + descriptions de Google News
    away_news_summary: str | None = None
    # Override manual (config/player_overrides.yaml): jugadores forzados como DISPONIBLES.
    # Autoridad máxima sobre rumores de prensa — el LLM no debe recortar λ por su "ausencia".
    home_available: list[str] | None = None
    away_available: list[str] | None = None


# ============ schema del output ============

@dataclass(frozen=True)
class QualitativeAdjustment:
    delta_lambda_L: float    # ∈ [-MAX_ABS_ADJUSTMENT, +MAX_ABS_ADJUSTMENT]
    delta_lambda_V: float    # mismo bound
    reasoning: str           # justificación textual breve
    confidence: float        # 0..1, cuán seguro está el LLM
    raw_response: dict       # para debug

    def clipped(self) -> "QualitativeAdjustment":
        """Aplica el guardrail de ±MAX_ABS_ADJUSTMENT."""
        return QualitativeAdjustment(
            delta_lambda_L=max(-MAX_ABS_ADJUSTMENT, min(MAX_ABS_ADJUSTMENT, self.delta_lambda_L)),
            delta_lambda_V=max(-MAX_ABS_ADJUSTMENT, min(MAX_ABS_ADJUSTMENT, self.delta_lambda_V)),
            reasoning=self.reasoning,
            confidence=self.confidence,
            raw_response=self.raw_response,
        )


# ============ prompt builder ============

SYSTEM_PROMPT = """Sos un analista de pronósticos de fútbol especializado en el Mundial 2026.
Tu trabajo es ajustar las predicciones cuantitativas del mercado en base a información cualitativa
que el mercado puede no haber procesado bien: lesiones de último momento, cambios de alineación
táctica, motivación (equipos que ya clasificaron y rotan), arbitraje, clima.

CUANDO RECIBÍS "NOTICIAS RECIENTES":
- Son titulares + descriptions de Google News, no artículos completos. Pueden ser ruidosos.
- Extraé de ahí: nombre de jugadores lesionados/suspendidos, cambios de DT, controversias internas,
  predicciones de alineación que cite el cuerpo técnico o medios oficiales.
- Ignorá noticias genéricas tipo "cómo ver el partido" o "horario y dónde se juega".
- OJO CON EL ECO: varios medios repitiendo la MISMA noticia (mismo cable o fuente original) son UNA
  sola fuente, no varias confirmaciones independientes. No subas la confianza por volumen de titulares
  que dicen lo mismo. Una lesión vista en un amistoso o entrenamiento es una SEÑAL, no un hecho: el
  jugador puede recuperarse y jugar igual. Confianza alta solo con confirmación oficial o el XI.

PRIORIDAD DE LA ALINEACIÓN CONFIRMADA:
- Si recibís "XI CONFIRMADO", esa es la verdad de quién juega y PISA cualquier rumor de noticias.
- Si un jugador reportado como lesionado/duda APARECE en el XI confirmado, está jugando: NO recortes
  el λ de su equipo por esa supuesta ausencia.
- Solo recortá λ por ausencia de un titular clave si NO está en el XI confirmado, o si no hay XI y la
  ausencia está confirmada por fuente oficial (no un rumor de amistoso).
- Si recibís "DISPONIBILIDAD CONFIRMADA (override manual)", tiene la MÁXIMA autoridad: esos jugadores
  juegan sí o sí, por encima de cualquier noticia o rumor. NO recortes λ por su supuesta ausencia y no
  menciones esa ausencia en el reasoning.

RESTRICCIONES IMPORTANTES:
- El mercado es generalmente muy eficiente. Solo ajustá si tenés información concreta y verificable.
- Tus ajustes (delta) están acotados a ±0.30 goles esperados por equipo. Más allá de eso, asumimos
  que tu razonamiento está mal o el mercado ya lo descontó.
- Si no tenés información cualitativa relevante, devolvé delta=0 con confidence=0.
- Si la info dice "Messi se lesionó" → reducí λ_Argentina. Si dice "Ronaldo descansa, ya clasificaron" → reducí λ_Portugal.
- En el reasoning, citá la fuente concreta de las noticias (ej. "según Marca, Pedri no estará").
- Devolvé SIEMPRE JSON válido con el schema exacto."""


def build_user_prompt(ctx: MatchContext) -> str:
    parts = [
        f"PARTIDO: {ctx.home_team} (local) vs {ctx.away_team} (visit)",
        f"FECHA: {ctx.kickoff_local}",
        f"ETAPA: {ctx.stage}",
        "",
        f"MERCADO (probabilidades implícitas):",
        f"  P(local) = {ctx.market_p_home:.0%}",
        f"  P(empate) = {ctx.market_p_draw:.0%}",
        f"  P(visit) = {ctx.market_p_away:.0%}",
        f"  E[goles local] = {ctx.market_e_goals_L:.2f}",
        f"  E[goles visit] = {ctx.market_e_goals_V:.2f}",
        "",
    ]

    if ctx.home_recent_form:
        parts.append(f"FORMA RECIENTE LOCAL: {ctx.home_recent_form}")
    if ctx.away_recent_form:
        parts.append(f"FORMA RECIENTE VISIT: {ctx.away_recent_form}")
    if ctx.home_injuries:
        parts.append(f"LESIONES LOCAL: {', '.join(ctx.home_injuries)}")
    if ctx.away_injuries:
        parts.append(f"LESIONES VISIT: {', '.join(ctx.away_injuries)}")
    if ctx.home_available:
        parts.append(
            f"DISPONIBILIDAD CONFIRMADA LOCAL (override manual — AUTORIDAD MÁXIMA, pisa rumores): "
            f"{', '.join(ctx.home_available)} SÍ juegan. Ignorá cualquier reporte de lesión/ausencia "
            f"sobre ellos y NO recortes λ por eso."
        )
    if ctx.away_available:
        parts.append(
            f"DISPONIBILIDAD CONFIRMADA VISIT (override manual — AUTORIDAD MÁXIMA, pisa rumores): "
            f"{', '.join(ctx.away_available)} SÍ juegan. Ignorá cualquier reporte de lesión/ausencia "
            f"sobre ellos y NO recortes λ por eso."
        )
    if ctx.home_lineup_change:
        parts.append(f"ALINEACIÓN LOCAL: {ctx.home_lineup_change}")
    if ctx.away_lineup_change:
        parts.append(f"ALINEACIÓN VISIT: {ctx.away_lineup_change}")
    xi_label = "XI CONFIRMADO" if ctx.lineup_confirmed else "XI PROBABLE"
    if ctx.home_xi:
        parts.append(f"{xi_label} LOCAL: {', '.join(ctx.home_xi)}")
    if ctx.away_xi:
        parts.append(f"{xi_label} VISIT: {', '.join(ctx.away_xi)}")
    if ctx.h2h_recent:
        parts.append(f"H2H RECIENTE: {ctx.h2h_recent}")
    if ctx.weather:
        parts.append(f"CLIMA: {ctx.weather}")
    if ctx.referee:
        parts.append(f"ÁRBITRO: {ctx.referee}")
    if ctx.motivation_notes:
        parts.append(f"MOTIVACIÓN: {ctx.motivation_notes}")
    if ctx.home_news_summary:
        parts.append(f"\nNOTICIAS RECIENTES — {ctx.home_team}:\n{ctx.home_news_summary}")
    if ctx.away_news_summary:
        parts.append(f"\nNOTICIAS RECIENTES — {ctx.away_team}:\n{ctx.away_news_summary}")

    parts.append("")
    parts.append("Devolvé un JSON con este schema exacto:")
    parts.append("""{
  "delta_lambda_L": <float entre -0.30 y +0.30>,
  "delta_lambda_V": <float entre -0.30 y +0.30>,
  "reasoning": "<1-2 oraciones explicando por qué>",
  "confidence": <float entre 0 y 1>
}""")

    return "\n".join(parts)


# ============ llamada al LLM ============

def adjust_with_llm(
    ctx: MatchContext,
    model: str = "claude-sonnet-4-6",
    api_key: str | None = None,
) -> QualitativeAdjustment:
    """Llama a Claude y obtiene el ajuste cualitativo. Aplica clipping al bound."""
    # Import perezoso para que el módulo no falle si anthropic no está instalado todavía
    import anthropic

    client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    user_prompt = build_user_prompt(ctx)

    log.info("LLM call | match=%s vs %s | tokens_in≈%d", ctx.home_team, ctx.away_team, len(user_prompt) // 4)

    response = client.messages.create(
        model=model,
        max_tokens=500,
        temperature=0.2,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    # Log de uso/costo
    try:
        from src.utils.usage_log import log_anthropic_call
        log_anthropic_call(model=model, usage=response.usage, purpose="qualitative", match_id=None)
    except Exception:
        pass

    raw_text = response.content[0].text  # type: ignore[union-attr]
    parsed = _extract_json(raw_text)

    adj = QualitativeAdjustment(
        delta_lambda_L=float(parsed.get("delta_lambda_L", 0.0)),
        delta_lambda_V=float(parsed.get("delta_lambda_V", 0.0)),
        reasoning=str(parsed.get("reasoning", "")),
        confidence=float(parsed.get("confidence", 0.0)),
        raw_response={"text": raw_text, "model": model},
    ).clipped()

    log.info(
        "LLM adj | ΔλL=%+.2f ΔλV=%+.2f conf=%.2f | %s",
        adj.delta_lambda_L, adj.delta_lambda_V, adj.confidence, adj.reasoning[:100],
    )
    return adj


def _extract_json(text: str) -> dict:
    """Extrae el JSON del response. Tolerante a code fences."""
    text = text.strip()
    if text.startswith("```"):
        # remover ```json ... ```
        lines = text.split("\n")
        text = "\n".join(l for l in lines if not l.startswith("```"))
    # Buscar primer { y último }
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        log.error("No pude extraer JSON del response del LLM: %r", text[:200])
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as e:
        log.error("JSON inválido del LLM: %s", e)
        return {}


# ============ integración con grid Poisson ============

def apply_to_lambdas(
    lam_L: float, lam_V: float,
    adjustment: QualitativeAdjustment,
) -> tuple[float, float]:
    """Aplica el ajuste cualitativo a (λ_L, λ_V) ya fiteadas del mercado.

    El bound ya se aplicó en clipped(). Acá garantizamos no-negatividad:
    los λ resultantes nunca pueden ser ≤ 0 (asegurar mínimo 0.1).
    """
    new_L = max(0.1, lam_L + adjustment.delta_lambda_L)
    new_V = max(0.1, lam_V + adjustment.delta_lambda_V)
    return new_L, new_V


# ============ CLI smoke ============

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    ap = argparse.ArgumentParser(description="Smoke test de Capa 4 LLM qualitative")
    ap.add_argument("--no-llm", action="store_true", help="Solo print del prompt, sin llamar API")
    args = ap.parse_args()

    ctx = MatchContext(
        home_team="Uruguay",
        away_team="Arabia Saudita",
        kickoff_local="Mar 16/06 16:00 UY",
        stage="Fase de grupos",
        market_p_home=0.65, market_p_draw=0.22, market_p_away=0.13,
        market_e_goals_L=1.8, market_e_goals_V=0.7,
        home_injuries=["Federico Valverde (duda, lesión muscular)"],
        away_injuries=[],
        h2h_recent="Uruguay le ganó 1-0 en Mundial 2022 (gol Cavani).",
        motivation_notes="Uruguay necesita ganar para asegurar liderato; Arabia ya está eliminada.",
    )

    print("=== PROMPT ===")
    print(build_user_prompt(ctx))

    if not args.no_llm:
        print("\n=== LLM RESPONSE ===")
        adj = adjust_with_llm(ctx)
        print(f"ΔλL = {adj.delta_lambda_L:+.2f}")
        print(f"ΔλV = {adj.delta_lambda_V:+.2f}")
        print(f"Reasoning: {adj.reasoning}")
        print(f"Confidence: {adj.confidence:.2f}")
