#!/usr/bin/env bash
# Instala y habilita la operación diaria de la penca LUB 26/27 en el droplet del Clausura:
#   - lub-picks.timer          (15:00 UTC → planilla de lo que cierra hoy por Telegram)
#   - lub-carga-alert.timer    (cada hora 11-23 UTC → recordatorio a 6h y 2h del cierre)
#   - lub-goleador-watch.timer (cada hora → aviso cuando aparezcan los menús de especiales)
#   - lub-cuotas-watch.timer   (cada hora → cuotas publicadas / sin matchear / mercado de campeón)
# El modo carga vive en el dashboard del Clausura: /dash/<token>/lub/carga/
#
# Env (/etc/penca/env): LUB_MIS_PARTICIPACIONES=n1,n2,... (ordenados como en la web)
# después de comprar; sin eso la planilla usa LUB_K (default 12) y columnas P1..PK.
# Idempotente. No toca lub-aviso-compra (instalado a mano el 29/9, fuera del repo).

set -euo pipefail

INSTALL_DIR="/opt/penca"
UNITS=(lub-picks.service lub-picks.timer
       lub-carga-alert.service lub-carga-alert.timer
       lub-goleador-watch.service lub-goleador-watch.timer
       lub-cuotas-watch.service lub-cuotas-watch.timer
       penca-failure-notify@.service)

cd "$INSTALL_DIR"
mkdir -p /var/lib/penca/logs

echo "==> Units systemd"
for u in "${UNITS[@]}"; do
    cp "$INSTALL_DIR/deploy/$u" "/etc/systemd/system/$u"
done
systemctl daemon-reload
systemctl enable --now lub-picks.timer lub-carga-alert.timer lub-goleador-watch.timer lub-cuotas-watch.timer

echo "==> Chequeo en seco"
"$INSTALL_DIR/.venv/bin/python" -m src.lub.goleador_watch --dry-run
"$INSTALL_DIR/.venv/bin/python" -m src.lub.carga_alert --dry-run
"$INSTALL_DIR/.venv/bin/python" -m src.lub.cuotas_watch --dry-run

systemctl list-timers 'lub-*' --no-pager
