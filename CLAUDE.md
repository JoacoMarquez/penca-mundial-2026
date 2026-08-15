# Penca — Agente Autónomo

> **Lo que corre hoy es el CLAUSURA (`src/clausura/`).** El sistema del Mundial
> ganó su penca el 2026-07-16 y está apagado desde el 29/7 (su droplet fue
> destruido). Todo lo que sigue del Mundial queda como referencia de diseño; para
> trabajar sobre producción, la sección de abajo es la que manda.

---

# Sistema activo: Penca Supermatch — Clausura 2026

Misma tesis que el Mundial (portfolio de perturbaciones alrededor del óptimo para
maximizar E[premio], no puntaje esperado), otro torneo y otra plataforma.

- **Penca:** Supermatch, id 46 (paga, $400/participación) + id 47 (gratuita).
- **Nuestras participaciones:** 12, números 899258848-866 (`CLAUSURA_MIS_PARTICIPACIONES`).
- **Pool:** ~737 participaciones. Premios: $350.000 al campeón de la penca,
  $10.000 por fecha, $3.000 grupo amigo.
- **Torneo:** 15 fechas, 16 equipos. Arrancó el 2026-08-07.
- **Puntos:** kernel aditivo de Supermatch (máx 8), partido preferencial ×2.
  Especiales: campeón y goleador, 25 puntos cada uno.
- **Carga:** **MANUAL** por el usuario (decisión operativa del 4/8, T&C). El
  sistema genera la planilla y la manda por Telegram; el dashboard tiene un
  "Modo carga" para copiarla sin errores.
- **Gate del API:** los picks —propios y de los rivales— son públicos recién al
  cierre de CADA partido. No se puede verificar lo cargado antes del cierre, que
  es justo cuando serviría. Nada que diga lo contrario es cierto.

## Módulos (`src/clausura/`)

| Módulo | Qué hace |
|---|---|
| `picks.py` | Pipeline completo: config → ratings → odds → grillas → pool → optimizador → planilla versionada. Es el `main` del sistema. |
| `strategy.py` | Ascenso por coordenadas sobre el portfolio (menú de candidatos K_EV, warm start desde la planilla previa) + `EvaluadorPortfolio` del gate por valor. |
| `economics.py` | `SeasonSimulator`: Monte Carlo de la temporada, liquidación de premios con reparto entre empatados (Art. 7a). |
| `scoring.py` | Kernel de puntos de Supermatch (verificado contra el Art. 6). |
| `rivals.py` | `RivalModel` empírico: picks conocidos, estilo γ por rival, `p_show`, residuo contra el ranking vivo. |
| `pool.py` / `pool_snapshot.py` | Distribución modelada del pool (chalk, sesgos) y su Q empírica desde el escaneo de picks públicos. |
| `ratings.py` / `historical.py` / `intermedio.py` | Ratings ofensivo/defensivo por equipo desde el histórico del penca-api + Intermedio 2026 (Wikipedia). |
| `odds.py` / `market_grid.py` | Cuotas del Elasticsearch público de Supermatch → λ de mercado (blend 70/30 con ratings). |
| `especiales.py` | P(campeón) y P(goleador) por simulación de la temporada. |
| `api.py` | Cliente de LECTURA del penca-api (sin auth): ranking, fechas, eventos, picks públicos. |
| `sync.py` | Regenera `config/clausura2026.yaml` desde el API (el fixture se reprograma seguido). |
| `rerun_cierre.py` | Corrida T-2h por tanda de cierres; avisa SOLO si el cambio vale plata (gate por valor). |
| `drift_audit.py` | Compara lo cargado en la web vs la planilla; adopta post-cierre lo que la web dice (la web es la verdad). |
| `carga_alert.py` | Recordatorios de carga a 6h y 2h del cierre (recordatorio, NO verificación). |
| `gate_watch.py` | Vigía del gate del API cada 10 min: captura snapshot si se abre una ventana. |
| `postmortem.py` | Cierre de fecha: puntos reales vs esperados, exactos, distribución del pool, tripwire de puntos propios vs ranking y PIT del pool. |
| `pool_pit.py` | ¿El modelo genera una cola tan gorda como la real? Simula rivales i.i.d. ∝ Q^γ contra los resultados REALES y ubica los cuantiles del pool observado. Cola corta ⇒ la vara para ganar es más alta que la que cree el optimizador y los rechazos de diferenciación hay que re-medirlos. |
| `cold_check.py` | Control FRÍO semanal del warm start: corre el pipeline desde el ancla EV (sin heredar la planilla previa, sin versionar) y avisa si le gana a la cadena por >2·SE y >$2.000. Es el único observable del trinquete del warm start. |
| `heartbeat.py` | Telegram diario que confirma que timers y servicios viven. |
| `webapp.py` / `dashboard_loader.py` / `verificar_carga.py` | Dashboard local (FastAPI): planilla, modo carga, pool, verificación post-cierre. |
| `backtest.py` | Validación sobre temporadas reales del penca-api. |

## Operación

**VPS:** DigitalOcean 159.203.66.24, 2 GB. Timers systemd (UTC):
`picks` 11:00 diario · `rerun-cierre` 11..23:35 · `carga-alert` 11..23:10 ·
`drift-audit` 13:20/18:20/23:50 · `postmortem` 03:20 · `heartbeat` 12:30 ·
`gate-watch` cada 10 min · `cold-check` martes 04:30 (semanal).

**Deploy:** `bash deploy/safe_pull.sh` en el VPS — NUNCA `git pull` a secas: los
units de systemd son copias en `/etc` y el pull no las aplica ni avisa.

**Producción:** `--sims 19200`. Los niveles de E[premio] NO son creíbles; lo único
que vale son los **deltas pareados sobre los mismos sorteos**.

**Registro de decisiones:** `config/decisiones.yaml` + `tests/test_decisiones_vigentes.py`
declaran cada constante elegida comparando alternativas y bajo qué supuestos se
midió. Si cambiás `--sims` u otra perilla, la suite se pone roja y lista qué hay
que volver a medir. Se apaga re-midiendo o declarando `vencida:`, nunca en silencio.

---

# Referencia histórica: Penca JMLM Mundial 2026 (ganada, apagada)

## Contexto del proyecto

Sistema **autónomo** para participar en la penca JMLM del Mundial 2026 con **5 planillas** simultáneas, diseñadas como un portfolio de perturbaciones óptimas alrededor del óptimo (no estrategias diferenciadas baratas) para maximizar la probabilidad de ganar un pool de 150+ participantes.

- **Web de la penca:** https://penca-jmlm-2026.vercel.app
- **Usuario en la web:** joacomarquez
- **Carga de pronósticos:** automática vía API REST (en desarrollo por el admin de la penca; specs disponibles antes del 2026-06-11)
- **Deadline por partido:** inicio del partido (la web bloquea predicciones al kickoff)
- **Marcador cuenta a:** 90 minutos (sin alargue ni penales)
- **Estructura del torneo:** 48 equipos, 12 grupos de 4. Clasifican: 2 primeros de cada grupo + 8 mejores terceros (32 a octavos). Ronda de 32 → octavos → cuartos → semis → final + 3er puesto.

## Sistema de puntos

**Regla vigente desde 2026-06-12** (cambió tras la jornada 1; el admin recalculó los puntos viejos retroactivamente):

| Puntos | Condición |
|--------|-----------|
| 6 | Marcador exacto (ambos goles correctos) |
| 4 | Ganador correcto + diferencia de gol correcta (solo partidos con ganador) |
| 3 | Ganador correcto (o empate acertado sin marcador exacto) |
| 0 | No acertó nada |

Nota: el empate acertado no exacto paga 3, NO 4 — en empates la diferencia de gol (0) se acierta por definición. No existe más el tier de 1 punto por goles de un equipo.

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
