# Penca JMLM Mundial 2026 — Agente Autónomo

## Contexto del proyecto

Sistema **autónomo** para participar en la penca JMLM del Mundial 2026 con **5 planillas** simultáneas, diseñadas como un portfolio de perturbaciones óptimas alrededor del óptimo (no estrategias diferenciadas baratas) para maximizar la probabilidad de ganar un pool de 150+ participantes.

- **Web de la penca:** https://penca-jmlm-2026.vercel.app
- **Usuario en la web:** joacomarquez
- **Carga de pronósticos:** automática vía API REST (en desarrollo por el admin de la penca; specs disponibles antes del 2026-06-11)
- **Deadline por partido:** inicio del partido (la web bloquea predicciones al kickoff)
- **Marcador cuenta a:** 90 minutos (sin alargue ni penales)
- **Estructura del torneo:** 48 equipos, 12 grupos de 4. Clasifican: 2 primeros de cada grupo + 8 mejores terceros (32 a octavos). Ronda de 32 → octavos → cuartos → semis → final + 3er puesto.

## Sistema de puntos

| Puntos | Condición |
|--------|-----------|
| 5 | Marcador exacto (ambos goles correctos) |
| 4 | Ganador correcto + goles de UN equipo correctos |
| 3 | Solo ganador (o empate) correcto |
| 1 | Goles de UN equipo correctos, sin acertar ganador |
| 0 | No acertó nada |

**Desempate del torneo:** 1) más marcadores exactos, 2) más ganadores correctos, 3) orden alfabético del nombre de la penca.

**Regla del sistema:** una predicción por partido por penca. Múltiples pencas permitidas por link de invitación.

## Objetivo

**Maximizar P(al menos una penca gana el pool)**, NO maximizar puntaje esperado.

Son objetivos distintos en pools grandes. Jugar chalk maximiza puntaje promedio pero te hace converger con 30-50 jugadores. Para ganar entre 150+, necesitamos varianza calculada: que al menos UNA de las 5 pencas tenga una combinación que casi nadie más tenga.

## Las 5 pencas — portfolio de perturbaciones óptimas

Cada partido genera 5 picks. **Todas son óptimas bajo objetivos distintos** — no hay chalk barato:

| # | Objetivo |
|---|----------|
| 1 | **EV puro** — `argmax E[points \| P]` |
| 2 | **Uplift vs pool** — `argmax (E[points] − E[pool_median_points])` |
| 3 | **Tail-max** — `argmax E[points \| top-5% scoring scenarios]` |
| 4 | **Conditional alternative** — `argmax E[points \| ganador opuesto]` si pick #1 es chalk |
| 5 | **Variance-max forzando diversidad** — máxima varianza entre top-K que no coincidan con #1-#4 |

## Arquitectura

```
penca-mundial-2026/
├── config/
│   ├── teams.yaml            # 48 selecciones + grupos + Elo inicial
│   ├── fixtures.yaml         # calendario completo
│   ├── books.yaml            # config de scrapers de casas
│   └── tipsters.yaml         # fuentes + parsers
├── data/
│   ├── raw/{source}/{date}/
│   ├── processed/*.parquet
│   └── predictions/{match_id}/v{N}_{ts}.json   # versionado, nunca sobreescribe
├── src/
│   ├── scrapers/             # odds (Pinnacle/Bet365/Betfair), tipsters, stats, news, lineups
│   ├── model/                # 5 capas probabilísticas
│   ├── strategy/portfolio.py # las 5 objective functions
│   ├── meta/pool.py          # modelo del pool
│   ├── agent/pipeline.py     # orquesta T-24h/T-3h/T-30min
│   ├── publisher/            # API de la penca (interfaz + impl HTTP)
│   ├── notifier/             # Telegram bot
│   └── backtest/             # validación contra Eurocopa 2024, etc.
├── dashboard/app.py          # Streamlit local
├── deploy/                   # systemd timers, setup VPS
└── tests/
```

## Modelo en 5 capas

**Capa 1 — Market consensus.** Pinnacle (sharp, peso 0.5) + Betfair Exchange (0.3) + Bet365 (0.2). De-vig con proportional method para 1X2 y Shin's method para marcador exacto.

**Capa 2 — Elo + xG.** ClubElo + EloRatings.net para diferencial. FBref para xG reciente y forma. λ_local y λ_visit "stat-prior" blendeados 70/30 con Capa 1.

**Capa 3 — Tipster signals.** 5-10 fuentes (The Athletic, Marca, AS, Forebet, SportingPedia, etc.). LLM Claude Sonnet 4.6 extrae JSON estructurado por tipster. Agrega consensus + dispersión.

**Capa 4 — Qualitative adjustment.** LLM lee lesiones / alineación probable / h2h / clima / árbitro / motivación y devuelve ajuste a (λ_L, λ_V) acotado a ±0.3 goles con justificación.

**Capa 5 — Pool model.** Sin acceso a picks individuales (la web solo muestra ranking).
- **Prior:** chalk con sesgo a marcadores populares (1-0, 2-0, 2-1).
- **Calibración online:** ranking-inversion después de cada jornada (mediana de scores → pick mediano implícito).

## Pipeline en tiempo real

Cada partido tiene tres pasadas. Cada una versiona en `data/predictions/{match_id}/v{N}.json`.

- **T-72h:** batch overnight con scraping base + caché.
- **T-24h:** investigación exhaustiva (5 capas + LLM). Genera v1 + notifica los 5 picks por Telegram para revisión.
- **T-3h:** alineaciones probables. Diff vs v1. Si cambia, regenera (v2) y notifica el cambio.
- **T-30min:** alineaciones confirmadas. v_final + publica vía API + notifica.

**Trigger extra:** movimiento brusco de odds Pinnacle (>5%) entre pasadas dispara recomputo.

## Autonomía y notificaciones

**Modo:** 100% autónomo. El sistema publica las picks por la API sin intervención del usuario. Pero:
- Notifica los 5 picks a T-24h (revisión opcional).
- Notifica cuando hay cambios en T-3h o T-30min.
- Notifica errores (scraper caído, API rate-limited, fallo de publicación).

**Canal:** Telegram bot privado.

## Stack

- **Lenguaje:** Python 3.11+
- **Data:** pandas, polars, pyarrow
- **Scraping:** httpx, beautifulsoup4, playwright (Bet365 / sitios JS-heavy)
- **Modelo:** numpy, scipy.stats (bivariate Poisson)
- **LLM:** Anthropic SDK (Claude Sonnet 4.6 default; Opus 4.7 para partidos clave de eliminatorias)
- **Dashboard:** Streamlit (local)
- **Notificaciones:** python-telegram-bot
- **Publicación:** httpx (cliente de la API de la penca, specs TBD)
- **Storage:** parquet/json local en el VPS, sin DB
- **Tests:** pytest

## Deployment

**Plataforma:** DigitalOcean Droplet Basic ($6/mes), Ubuntu 24.04, 1 vCPU / 1GB RAM / 25GB SSD.

**Scheduling:** systemd timers (no cron) para T-24h / T-3h / T-30min por partido. El scheduler lee `config/fixtures.yaml` y crea timers dinámicos.

**Secrets:** variables de entorno en `/etc/penca/env`, leídas vía `python-dotenv`. Nunca commiteadas. Las llaves:
- `ANTHROPIC_API_KEY`
- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`
- `PENCA_API_BASE_URL`, `PENCA_API_KEY` (TBD)
- `PINNACLE_USER`, `PINNACLE_PASS` (si se requiere login)
- `BETFAIR_APP_KEY`, `BETFAIR_SESSION_TOKEN`

**Deploy flow:** push a GitHub → SSH al VPS → `git pull` + `systemctl daemon-reload` + restart timers.

## Reglas de trabajo

1. **Todo cambio en estrategia o modelo va con backtest.** Eurocopa 2024 (51 partidos, similar escala) es el dataset principal.
2. **Versionar predicciones.** Cada pasada escribe un archivo nuevo. Nunca sobreescribir.
3. **Honestidad sobre incertidumbre.** Si el modelo dice 55/30/15, decir eso. No inventar precisión.
4. **Postmortem después de cada jornada.** Qué pegó cada penca, qué inputs movieron más cada pick, recalibrar pesos.
5. **Bound al LLM.** El ajuste cualitativo de Capa 4 no puede mover (λ_L, λ_V) más de ±0.3. Guardrail para no sobrescribir al mercado.
6. **Logs estructurados.** Todo en JSON con timestamp. Postmortem automático al cierre de cada jornada.

## Calendario crítico

- **2026-06-11:** arranca el Mundial (México vs Sudáfrica, 19:00 UTC / 16:00 UY).
- **Fase de grupos:** 11 al ~27 de junio.
- **Eliminatorias:** ~28 de junio al 19 de julio (final).

## Estado actual y próximos pasos

Estado: Fase 0 — skeleton creado, configs en construcción.

**MVP (Días 1-10, listo para kickoff):**
1. ✓ Skeleton del repo
2. `config/teams.yaml`, `config/fixtures.yaml` (en curso)
3. Scrapers Pinnacle + Betfair API
4. Modelo capas 1, 2, 5 (sin LLM aún)
5. Generación de las 5 picks
6. Backtest Eurocopa 2024
7. Pipeline T-24h orquestado
8. Telegram notifier
9. Publisher (interfaz, impl pendiente de specs)
10. Deploy VPS

**Enriquecimiento durante fase de grupos (Días 11-25):**
11. Scrapers de tipsters
12. Capa 3 (LLM extraction)
13. Capa 4 (LLM qualitative)
14. Pipelines T-3h y T-30min
15. Ranking-inversion para Capa 5
16. Dashboard Streamlit

## Contexto adicional

Usuario en Uruguay (UTC-3). Mundial en USA/Canadá/México, muchos partidos serán de tarde/noche en Uruguay. Los timers de systemd deben estar en UTC para evitar problemas de zona horaria.
