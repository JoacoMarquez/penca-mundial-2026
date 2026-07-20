#!/usr/bin/env bash
# teardown_penca.sh — Baja los timers de la PENCA (Mundial 2026 terminado) y
# DEJA valuebet corriendo. Idempotente: se puede correr varias veces.
#
# Uso (desde el VPS, como root):
#   sudo bash /opt/penca/deploy/teardown_penca.sh
#
# No toca: valuebet-scan/close/settle/report, ni penca-failure-notify (OnFailure).
set -euo pipefail

echo "=== ANTES: timers de penca activos ==="
systemctl list-timers --all 'penca-*' || true

echo
echo "=== Deshabilitando timers de la penca ==="
for unit in penca-scheduler.timer penca-heartbeat.timer penca-fixtures-sync.timer; do
  if systemctl list-unit-files --no-legend "$unit" | grep -q .; then
    systemctl disable --now "$unit" && echo "  ✓ $unit deshabilitado"
  else
    echo "  - $unit no existe, salto"
  fi
done

echo
echo "=== Barriendo timers per-match dinámicos (si quedaron) ==="
mapfile -t MATCH_TIMERS < <(systemctl list-unit-files --no-legend 'penca-match-*.timer' 2>/dev/null | awk '{print $1}')
if [ "${#MATCH_TIMERS[@]}" -eq 0 ]; then
  echo "  - no hay timers per-match"
else
  for t in "${MATCH_TIMERS[@]}"; do
    systemctl disable --now "$t" && echo "  ✓ $t deshabilitado"
  done
fi

echo
echo "=== DESPUÉS: penca (deberían quedar todos dead/disabled) ==="
systemctl list-timers --all 'penca-*' || true
echo
echo "=== valuebet (deberían seguir ACTIVOS) ==="
systemctl list-timers 'valuebet-*' || true

echo
echo "Listo. Penca abajo, valuebet intacto."
