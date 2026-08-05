#!/usr/bin/env bash
# Instala y habilita SOLO el stack del Clausura en el droplet:
#   - clausura-dashboard.service   (uvicorn :8000, token en la URL)
#   - clausura-picks.timer         (jue-dom 12:00 UTC → planilla por Telegram)
#   - clausura-carga-alert.timer   (cada hora 11-23 UTC → aviso si falta cargar)
#   - clausura-drift-audit.timer   (3x/día → web vs planilla guardada)
#   - clausura-postmortem.timer    (03:20 UTC → análisis al completarse una fecha)
#   - clausura-rerun-cierre.timer  (T-2h del primer cierre → diff vs planilla de la mañana)
#   - penca-failure-notify@        (OnFailure de los services → Telegram)
#
# Prerequisito: haber corrido setup_droplet.sh (clona /opt/penca, crea .venv con
# requirements) y tener /etc/penca/env con al menos:
#   DASHBOARD_TOKEN=...
#   TELEGRAM_BOT_TOKEN=... / TELEGRAM_CHAT_ID=...
#
# NO habilita nada del Mundial ni del valuebet. Idempotente.

set -euo pipefail

INSTALL_DIR="/opt/penca"
UNITS=(clausura-dashboard.service clausura-picks.service clausura-picks.timer
       clausura-carga-alert.service clausura-carga-alert.timer
       clausura-drift-audit.service clausura-drift-audit.timer
       clausura-postmortem.service clausura-postmortem.timer
       clausura-rerun-cierre.service clausura-rerun-cierre.timer
       penca-failure-notify@.service)

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
systemctl enable --now clausura-carga-alert.timer
systemctl enable --now clausura-drift-audit.timer
systemctl enable --now clausura-postmortem.timer
systemctl enable --now clausura-rerun-cierre.timer

echo "==> Estado"
systemctl --no-pager status clausura-dashboard.service | head -5
systemctl list-timers 'clausura-*' --no-pager

IP=$(curl -s -4 ifconfig.me || echo "<IP>")
TOKEN=$(grep '^DASHBOARD_TOKEN=' /etc/penca/env | cut -d= -f2)
echo
echo "Dashboard: http://${IP}:8000/dash/${TOKEN}/"
