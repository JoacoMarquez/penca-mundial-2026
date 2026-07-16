#!/usr/bin/env bash
# Setup del sistema valuebet en el droplet — separado de setup_droplet.sh (penca)
# para no tocar nada del sistema penca. Idempotente.
#
# Uso (en el droplet, con el repo ya en /opt/penca):
#   bash /opt/penca/deploy/setup_valuebet.sh
set -euo pipefail

REPO=/opt/penca
ENV_FILE=/etc/penca/valuebet-env

echo "› Directorios de datos"
mkdir -p "$REPO/data/valuebet"

echo "› Env file"
# Los secretos (TELEGRAM_*, ODDS_API_KEY) se REUSAN de /etc/penca/env — las unidades
# cargan ese archivo primero y este después (solo overrides no-secretos). Por eso acá
# NO se escriben placeholders de secretos: pisarían las credenciales reales de la penca.
if [[ ! -f "$ENV_FILE" ]]; then
  cat > "$ENV_FILE" <<'EOF'
# Valuebet — overrides NO-secretos. Los secretos vienen de /etc/penca/env (reuso penca).
# Para mandar las sugerencias a un chat de Telegram distinto al de la penca:
# VALUEBET_TELEGRAM_CHAT_ID=
VALUEBET_DATA_DIR=/opt/penca/data/valuebet
EOF
  chmod 600 "$ENV_FILE"
  echo "  Creado $ENV_FILE (sin secretos; se reusan de /etc/penca/env)."
else
  echo "  $ENV_FILE ya existe, no lo toco."
fi

echo "› Unidades systemd"
for unit in valuebet-scan valuebet-close valuebet-settle valuebet-report; do
  cp "$REPO/deploy/$unit.service" "/etc/systemd/system/$unit.service"
  cp "$REPO/deploy/$unit.timer" "/etc/systemd/system/$unit.timer"
done
systemctl daemon-reload

echo "› Bankroll inicial (si no existe)"
if [[ ! -f "$REPO/data/valuebet/bankroll.json" ]]; then
  echo '  Inicializar con: PYTHONPATH=/opt/penca /opt/penca/.venv/bin/python -m src.valuebet.runner bankroll --set 1000'
fi

echo "› Habilitar timers"
for unit in valuebet-scan valuebet-close valuebet-settle valuebet-report; do
  systemctl enable --now "$unit.timer"
done

echo
echo "✅ Valuebet instalado. Verificar con: systemctl list-timers 'valuebet-*'"
echo "   Smoke sport IDs Pinnacle: PYTHONPATH=/opt/penca /opt/penca/.venv/bin/python -m src.valuebet.books.pinnacle_vb --list-sports"
