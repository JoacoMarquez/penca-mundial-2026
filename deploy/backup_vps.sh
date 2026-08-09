#!/usr/bin/env bash
# backup_vps.sh — Baja del VPS todo lo que NO está en git antes de destruir el droplet.
# Se corre DESDE TU MAC, no desde el VPS.
#
# Uso:
#   bash deploy/backup_vps.sh                    # destino por defecto: ./backups/vps-<fecha>
#   bash deploy/backup_vps.sh /ruta/destino
#
# Qué baja:
#   /opt/penca/data          → predicciones versionadas, ledger de valuebet, aliases, CLV, caches
#   /var/lib/penca           → logs estructurados
#   /etc/penca/env           → secrets (¡NO commitear!)
#   /etc/penca/valuebet-env  → overrides de valuebet
#   /root/penca-shutdown-state.txt → qué units estaban habilitadas (para revertir)
set -euo pipefail

# Droplet del CLAUSURA (creado 2026-08-04). El anterior — root@68.183.52.166, el del
# Mundial — se destruyó el 2026-07-29 y quedó como default acá hasta el 9/8: correr el
# script sin PENCA_VPS_HOST fallaba contra una IP muerta, justo en el momento en que
# uno corre un backup. Su respaldo sigue en ../penca-backups/vps-20260729/.
VPS="${PENCA_VPS_HOST:-root@159.203.66.24}"
DEST="${1:-backups/vps-$(date +%Y%m%d)}"

echo "› Origen : $VPS"
echo "› Destino: $DEST"
echo

if ! ssh -o ConnectTimeout=10 -o BatchMode=yes "$VPS" true 2>/dev/null; then
  echo "✗ No pude conectar a $VPS por SSH."
  echo "  Verificá la IP (PENCA_VPS_HOST=root@<IP>) y que la key esté en el agente."
  exit 1
fi

mkdir -p "$DEST"

echo "=== Tamaño en el origen ==="
ssh "$VPS" 'du -sh /opt/penca/data /var/lib/penca 2>/dev/null; ls -la /etc/penca/ 2>/dev/null' || true

echo
echo "=== Bajando datos ==="
rsync -az --stats "$VPS:/opt/penca/data/" "$DEST/data/"
rsync -az --stats "$VPS:/var/lib/penca/" "$DEST/var-lib-penca/"

echo
echo "=== Bajando config/secrets ==="
mkdir -p "$DEST/etc-penca"
rsync -az "$VPS:/etc/penca/" "$DEST/etc-penca/" 2>/dev/null || echo "  ! /etc/penca no accesible"
chmod -R go-rwx "$DEST/etc-penca" 2>/dev/null || true

echo
echo "=== Bajando estado de systemd ==="
scp -q "$VPS:/root/penca-shutdown-state.txt" "$DEST/" 2>/dev/null \
  && echo "  ✓ penca-shutdown-state.txt" \
  || echo "  - no existe (¿corriste shutdown_all.sh primero?)"
ssh "$VPS" "systemctl list-timers --all 'penca-*' 'valuebet-*'" > "$DEST/timers-snapshot.txt" 2>&1 || true
ssh "$VPS" "systemctl list-unit-files --no-legend 'penca-*' 'valuebet-*'" > "$DEST/units-snapshot.txt" 2>&1 || true

echo
echo "=== Resultado local ==="
du -sh "$DEST"
find "$DEST" -maxdepth 2 -type d | head -30

echo
echo "✅ Backup en $DEST"
echo "   ⚠️  Contiene secrets en $DEST/etc-penca — NO lo commitees (chequeá .gitignore)."
echo "   Recién ahora es seguro correr deploy/destroy_droplet.sh"
