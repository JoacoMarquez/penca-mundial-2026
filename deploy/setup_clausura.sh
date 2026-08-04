#!/usr/bin/env bash
# Instala y habilita SOLO el stack del Clausura en el droplet:
#   - clausura-dashboard.service  (uvicorn :8000, token en la URL)
#   - clausura-picks.timer        (jue-dom 12:00 UTC → planilla por Telegram)
#
# Prerequisito: haber corrido setup_droplet.sh (clona /opt/penca, crea .venv con
# requirements) y tener /etc/penca/env con al menos:
#   DASHBOARD_TOKEN=...
#   TELEGRAM_BOT_TOKEN=... / TELEGRAM_CHAT_ID=...
#
# NO habilita nada del Mundial ni del valuebet. Idempotente.

set -euo pipefail

INSTALL_DIR="/opt/penca"
UNITS=(clausura-dashboard.service clausura-picks.service clausura-picks.timer)

echo "==> Pull del repo"
cd "$INSTALL_DIR" && git pull

echo "==> Config del Clausura (ids de fechas/eventos desde el penca-api)"
"$INSTALL_DIR/.venv/bin/python" -m src.clausura.sync

echo "==> Histórico para ratings (si falta)"
if [ ! -f "$INSTALL_DIR/data/processed/primera_uy_historico.json" ]; then
    "$INSTALL_DIR/.venv/bin/python" -m src.clausura.historical
fi

echo "==> Units systemd"
for u in "${UNITS[@]}"; do
    cp "$INSTALL_DIR/deploy/$u" "/etc/systemd/system/$u"
done
systemctl daemon-reload
systemctl enable --now clausura-dashboard.service
systemctl enable --now clausura-picks.timer

echo "==> Estado"
systemctl --no-pager status clausura-dashboard.service | head -5
systemctl list-timers 'clausura-*' --no-pager

IP=$(curl -s -4 ifconfig.me || echo "<IP>")
TOKEN=$(grep '^DASHBOARD_TOKEN=' /etc/penca/env | cut -d= -f2)
echo
echo "Dashboard: http://${IP}:8000/dash/${TOKEN}/"
