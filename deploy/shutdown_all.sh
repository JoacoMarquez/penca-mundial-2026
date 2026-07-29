#!/usr/bin/env bash
# shutdown_all.sh — Apaga TODO el sistema en el VPS: penca + valuebet + dashboard.
# Idempotente y reversible: antes de deshabilitar, guarda la lista de units que
# estaban habilitadas en /root/penca-shutdown-state.txt para poder revertir exacto.
#
# Uso (desde el VPS, como root):
#   sudo bash /opt/penca/deploy/shutdown_all.sh
#
# Para revertir (mismo droplet, sin destruir):
#   sudo bash /opt/penca/deploy/shutdown_all.sh --undo
#
# Si el droplet ya no existe, ver deploy/RESUME.md para levantar de cero.
set -euo pipefail

STATE=/root/penca-shutdown-state.txt

# Todo lo que puede quedar corriendo. Los timers per-match son dinámicos (penca-match-*).
TIMERS=(
  penca-scheduler.timer
  penca-heartbeat.timer
  penca-fixtures-sync.timer
  valuebet-scan.timer
  valuebet-close.timer
  valuebet-settle.timer
  valuebet-report.timer
)
SERVICES=(
  penca-dashboard.service
)

# Ojo: systemctl list-unit-files sale con 1 cuando el patrón no matchea nada, y con
# `set -o pipefail` eso mataría el script. De ahí los `|| true` en los pipes.
unit_exists() {
  { systemctl list-unit-files --no-legend "$1" 2>/dev/null || true; } | grep -q .
}

collect_match_timers() {
  { systemctl list-unit-files --no-legend 'penca-match-*.timer' 2>/dev/null || true; } | awk '{print $1}'
}

do_undo() {
  if [[ ! -f "$STATE" ]]; then
    echo "No existe $STATE — no hay nada que revertir."
    echo "Si el droplet es nuevo, corré setup_droplet.sh y setup_valuebet.sh (ver RESUME.md)."
    exit 1
  fi
  echo "=== Revirtiendo desde $STATE ==="
  while read -r unit; do
    [[ -z "$unit" || "$unit" == \#* ]] && continue
    if unit_exists "$unit"; then
      systemctl enable --now "$unit" && echo "  ✓ $unit re-habilitado"
    else
      echo "  ! $unit ya no existe (¿repo movido?), salto"
    fi
  done < "$STATE"
  echo
  echo "=== Estado ahora ==="
  systemctl list-timers --all 'penca-*' 'valuebet-*' || true
  echo
  echo "Listo. Sistema de vuelta arriba. Verificá el heartbeat de Telegram."
  exit 0
}

if [[ "${1:-}" == "--undo" ]]; then
  do_undo
fi

echo "=== ANTES: timers activos ==="
systemctl list-timers --all 'penca-*' 'valuebet-*' || true

echo
echo "=== Guardando estado actual en $STATE ==="
{
  echo "# Units habilitadas antes del shutdown. Generado por shutdown_all.sh"
  for unit in "${TIMERS[@]}" "${SERVICES[@]}"; do
    if unit_exists "$unit" && systemctl is-enabled --quiet "$unit" 2>/dev/null; then
      echo "$unit"
    fi
  done
  collect_match_timers
} > "$STATE"
echo "  ✓ $(grep -vc '^#' "$STATE" || true) units registradas"

echo
echo "=== Deshabilitando timers ==="
for unit in "${TIMERS[@]}"; do
  if unit_exists "$unit"; then
    systemctl disable --now "$unit" 2>/dev/null && echo "  ✓ $unit" || echo "  - $unit ya estaba abajo"
  else
    echo "  - $unit no existe, salto"
  fi
done

echo
echo "=== Barriendo timers per-match dinámicos ==="
mapfile -t MATCH_TIMERS < <(collect_match_timers)
if [ "${#MATCH_TIMERS[@]}" -eq 0 ]; then
  echo "  - no hay timers per-match"
else
  for t in "${MATCH_TIMERS[@]}"; do
    systemctl disable --now "$t" 2>/dev/null && echo "  ✓ $t" || echo "  - $t ya estaba abajo"
  done
fi

echo
echo "=== Parando servicios de larga vida ==="
for unit in "${SERVICES[@]}"; do
  if unit_exists "$unit"; then
    systemctl disable --now "$unit" 2>/dev/null && echo "  ✓ $unit" || echo "  - $unit ya estaba abajo"
  else
    echo "  - $unit no existe, salto"
  fi
done

echo
echo "=== DESPUÉS: no debería quedar nada activo ==="
systemctl list-timers --all 'penca-*' 'valuebet-*' || true
echo
echo "Procesos vivos del repo (debería estar vacío):"
pgrep -af '/opt/penca' || echo "  (ninguno)"

echo
echo "✅ Sistema apagado. Estado guardado en $STATE"
echo "   Revertir en este mismo droplet: sudo bash $0 --undo"
echo "   Antes de destruir el droplet: correr deploy/backup_vps.sh DESDE TU MAC."
