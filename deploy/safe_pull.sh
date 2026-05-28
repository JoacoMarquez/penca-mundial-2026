#!/usr/bin/env bash
# Safe pull: hace git pull, valida que todos los módulos importen, y SOLO si pasa,
# acepta el cambio. Si falla, hace rollback al commit anterior.
#
# Uso: ssh root@VPS /opt/penca/deploy/safe_pull.sh
#       o en cron: cada 5 min ANTES del scheduler.

set -e
cd /opt/penca

PREVIOUS=$(git rev-parse HEAD)
git fetch -q origin main
INCOMING=$(git rev-parse origin/main)

if [ "$PREVIOUS" = "$INCOMING" ]; then
    # Nada nuevo, no hace falta validar
    exit 0
fi

echo "Pulling $PREVIOUS → $INCOMING"
git pull -q

# Validar que los módulos críticos importan
if .venv/bin/python -c "
import sys
try:
    from src.agent import scheduler
    from src.agent import pipeline
    from src.agent import heartbeat
    from src.model import dossier, qualitative, poisson, anomaly
    from src.strategy import portfolio, assignment
    from src.scrapers import espn, news, pinnacle, weather, football_api
    print('OK')
except Exception as e:
    print(f'IMPORT ERROR: {type(e).__name__}: {e}', file=sys.stderr)
    sys.exit(1)
" 2>&1; then
    echo "✅ Validation passed at $INCOMING"
else
    echo "❌ Validation FAILED — rolling back to $PREVIOUS" >&2
    git reset --hard "$PREVIOUS" -q
    # Notificar via Telegram (sourcing env)
    set -a
    source /etc/penca/env 2>/dev/null
    set +a
    .venv/bin/python -c "
from src.notifier.telegram import TelegramNotifier, TelegramConfig
try:
    n = TelegramNotifier(TelegramConfig.from_env())
    n.send_error('safe_pull rollback', 'Commit $INCOMING falló validación, rollback a $PREVIOUS')
except Exception:
    pass
" 2>/dev/null || true
    exit 1
fi
