# Penca JMLM Mundial 2026

Agente autónomo de pronósticos para la penca JMLM del Mundial 2026.

5 planillas simultáneas, cada una óptima bajo un objetivo distinto (no estrategias diferenciadas baratas). Investigación exhaustiva por partido: scraping de Pinnacle + Betfair + Bet365, 8-10 tipsters, lesiones/alineaciones, análisis cualitativo con Claude Sonnet. Tres pasadas por partido (T-24h, T-3h, T-30min). Publicación automática vía API de la penca, notificaciones por Telegram.

Ver [`CLAUDE.md`](CLAUDE.md) para el contexto completo del proyecto y el plan de build en [`~/.claude/plans/fluttering-tickling-comet.md`](~/.claude/plans/fluttering-tickling-comet.md).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium

# Copiar .env.example a .env y completar las credenciales
cp .env.example .env
```

## Estructura

```
config/        teams.yaml, fixtures.yaml, books.yaml, tipsters.yaml
data/          raw/  processed/  predictions/    (gitignored)
src/
  scrapers/    odds + tipsters + stats + lineups + news
  model/       5 capas de probabilidad
  meta/        modelo del pool
  strategy/    portfolio.py — las 5 picks
  publisher/   cliente de la API de la penca
  notifier/    Telegram bot
  agent/       pipeline + scheduler
  backtest/    validación contra torneos pasados
dashboard/     Streamlit local
deploy/        systemd, setup VPS
tests/
```

## Estado

**Fase 0 (en curso):** skeleton + configs.
- ✓ Estructura de carpetas
- ✓ `config/teams.yaml` (48 equipos, 12 grupos, Elo seed)
- ✓ `config/fixtures.yaml` (schema + Grupo A; el resto via scraper)
- ✓ `config/books.yaml`, `config/tipsters.yaml`
- ✓ `requirements.txt`, `.env.example`

**Próximo (Día 2-3):** scrapers de odds (Pinnacle + Betfair) y módulo de-vig.

Ver tasks activas con `TaskList` en la sesión de Claude Code.

## Calendario crítico

- **2026-06-11:** arranca el Mundial. México vs Sudáfrica, Estadio Azteca, 19:00 UTC (16:00 UY).
- **MVP objetivo:** 2026-06-08 (operativo para fase de grupos).
- **Final:** 2026-07-19, MetLife Stadium.
