# AGENTS.md — Penca JMLM Mundial 2026

Contexto completo del proyecto: ver `CLAUDE.md` (arquitectura, modelo en 5 capas,
sistema de puntos, pipeline, deploy). Este archivo agrega guías de **revisión de PRs**
para Codex.

## Review guidelines

Codex marca solo P0/P1. Priorizá estos riesgos, propios de este proyecto:

### P0 — bloqueantes
- **Cambios de estrategia o modelo SIN backtest.** Toda modificación en `src/strategy/`,
  `src/model/` o `src/meta/` debe venir con evidencia de backtest (Euro 2024) o tests que
  cubran el cambio. Sin eso, es P0.
- **Sobrescritura de predicciones.** Las predicciones se versionan en
  `data/predictions/{match_id}/vN_*.json` y **nunca** se sobrescriben. Flag cualquier
  código que pise una versión existente.
- **Secrets hardcodeados.** API keys, tokens (`ANTHROPIC_API_KEY`, `TELEGRAM_*`,
  `PENCA_API_*`, `PINNACLE_*`, `BETFAIR_*`) solo vía env. Nunca commiteados ni en logs.
- **Publicación sin guardrail de DRY_RUN.** El publisher real no debe activarse si
  `DRY_RUN=true`. Cuidado con caminos que publiquen a la API saltando ese check.

### P1 — alto
- **Bound del LLM (Capa 4).** El ajuste cualitativo no puede mover `(λ_L, λ_V)` más de
  **±0.3 goles**. Flag cualquier código que afloje o saltee ese límite.
- **Objetivo correcto.** El sistema maximiza **P(al menos una penca gana el pool)**, NO el
  puntaje esperado. Cuidado con cambios que optimicen EV puro y rompan la diversificación
  (exposición/repetición del portfolio de N pencas).
- **N pencas genérico.** El código debe funcionar para N variable (no asumir 5 ni 15).
  Flag `== 5`, `range(5)`, `[:5]` o similar que reintroduzca el cableado.
- **Zonas horarias.** Timers/scheduling en **UTC**; conversión a UY (UTC-3) solo para
  display. Flag mezclas de TZ.
- **Honestidad sobre incertidumbre.** Probabilidades reportadas tal cual el modelo; nada de
  precisión inventada. Vale para outputs y para mensajes al usuario.
- **Logs estructurados** (JSON con timestamp) en los caminos del pipeline.

### Qué NO marcar
- Estilo/formato cubierto por linters.
- Nits de naming sin impacto funcional.
- Comentarios en español (el proyecto es en español rioplatense, es intencional).

## Cómo correr los tests

```bash
python -m pytest -q
```

## Backtest (para validar cambios de estrategia)

```bash
python -m src.backtest.runner euro_2024 --sweep 5,10,15,20 --seq --sims 1500
```
