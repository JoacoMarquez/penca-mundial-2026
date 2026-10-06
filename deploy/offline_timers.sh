#!/usr/bin/env bash
# Pausa o reanuda los timers que dependen de Supermatch (modo offline, ver
# src/clausura/offline.py). La lista sale de offline.TIMERS_PAUSADOS.
#   bash deploy/offline_timers.sh pausar     # + marca activo: true
#   bash deploy/offline_timers.sh reanudar   # + marca activo: false
set -euo pipefail
cd /opt/penca
ACCION="${1:?uso: offline_timers.sh pausar|reanudar}"
TIMERS=$(.venv/bin/python -c "from src.clausura.offline import TIMERS_PAUSADOS as t; print(' '.join(t))")
case "$ACCION" in
  pausar)   for t in $TIMERS; do systemctl disable --now "$t.timer" || true; done; ACTIVO=true ;;
  reanudar) for t in $TIMERS; do systemctl enable --now "$t.timer" || true; done; ACTIVO=false ;;
  *) echo "acción desconocida: $ACCION" >&2; exit 1 ;;
esac
F=data/state/clausura_offline.yaml
[ -f "$F" ] || cp config/clausura_offline.example.yaml "$F"
sed -i "s/^activo: .*/activo: $ACTIVO/" "$F"
systemctl list-timers --all --no-pager | grep -E "clausura|lub|valuebet" || true
echo "offline activo: $ACTIVO"
