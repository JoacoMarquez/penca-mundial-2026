#!/usr/bin/env bash
# Chequeo rápido de salud del sistema en producción.
# Uso: ./deploy/health_check.sh
#
# Verifica:
#   - SSH conectividad
#   - systemd timers activos
#   - Última ejecución del scheduler
#   - Errores recientes
#   - Disk + memoria
#   - API de la penca responde
#   - Pinnacle responde

set -u
VPS="root@68.183.52.166"

print_header() { printf "\n\033[1;34m═══ %s ═══\033[0m\n" "$1"; }
print_ok()     { printf "  \033[32m✓\033[0m %s\n" "$1"; }
print_warn()   { printf "  \033[33m⚠\033[0m %s\n" "$1"; }
print_err()    { printf "  \033[31m✗\033[0m %s\n" "$1"; }

echo ""
echo "🔍 Chequeo de salud — Penca Mundial 2026"
echo "VPS: $VPS"

# 1. SSH connectivity
print_header "SSH"
if ssh -o ConnectTimeout=5 -o BatchMode=yes "$VPS" "echo ok" >/dev/null 2>&1; then
  print_ok "SSH funciona"
else
  print_err "SSH NO responde — droplet caído?"
  exit 1
fi

# 2. systemd timers
print_header "systemd timers"
ssh "$VPS" "systemctl is-active penca-scheduler.timer 2>/dev/null" | grep -q "^active$" && \
  print_ok "penca-scheduler.timer activo" || print_err "penca-scheduler.timer NO activo"
ssh "$VPS" "systemctl is-active penca-heartbeat.timer 2>/dev/null" | grep -q "^active$" && \
  print_ok "penca-heartbeat.timer activo" || print_warn "penca-heartbeat.timer NO activo (puede que no esté instalado todavía)"

# 3. Última ejecución del scheduler
print_header "Última ejecución scheduler"
LAST_RUN=$(ssh "$VPS" "systemctl show penca-scheduler.service -p ActiveEnterTimestamp --value 2>/dev/null")
LAST_STATUS=$(ssh "$VPS" "systemctl show penca-scheduler.service -p Result --value 2>/dev/null")
echo "  ⏰ Último run: ${LAST_RUN:-?}"
[ "$LAST_STATUS" = "success" ] && print_ok "Resultado: success" || print_err "Resultado: $LAST_STATUS"

# 4. Errores recientes
print_header "Errores recientes (24h)"
ERR_COUNT=$(ssh "$VPS" "journalctl -u penca-scheduler --since '24h ago' -p err --no-pager 2>/dev/null | grep -c ERROR" | tr -d '[:space:]')
ERR_COUNT=${ERR_COUNT:-0}
if [ "$ERR_COUNT" = "0" ]; then
  print_ok "Sin errores en 24h"
else
  print_warn "$ERR_COUNT errores en últimas 24h — ver: ssh $VPS 'journalctl -u penca-scheduler -p err --since 24h ago'"
fi

# 5. Predicciones generadas
print_header "Predicciones generadas"
TOTAL_PRED=$(ssh "$VPS" "find /opt/penca/data/predictions -name 'v*_*.json' 2>/dev/null | wc -l | tr -d ' '")
echo "  Total versiones generadas: $TOTAL_PRED"

# 6. Disk + memoria
print_header "Recursos del VPS"
ssh "$VPS" "df -h /opt /var/lib/penca 2>/dev/null | awk 'NR>1{print \"  \"\$NF\": \"\$5\" usado (\"\$3\" de \"\$2\")\"}' ; echo ; free -h | awk 'NR==2{print \"  RAM: \"\$3\" usado de \"\$2\"\"}'"

# 7. API de la penca (sin auth, solo conectividad)
print_header "API penca"
HTTP_PENCA=$(curl -s -o /dev/null -w "%{http_code}" -m 5 https://penca-jmlm-2026.vercel.app/api/v1/matches 2>/dev/null)
[ "$HTTP_PENCA" = "401" ] && print_ok "API responde (401 sin auth, como esperado)" || \
  ([ "$HTTP_PENCA" = "200" ] && print_ok "API responde (200)" || print_warn "API devolvió $HTTP_PENCA")

# 8. Pinnacle desde VPS
print_header "Pinnacle desde VPS"
HTTP_PIN=$(ssh "$VPS" "curl -s -o /dev/null -w '%{http_code}' -m 10 https://guest.api.arcadia.pinnacle.com/0.1/sports/29/leagues -H 'X-API-Key: CmX2KcMrXuFmNg6YFbmTxE0y9CIrOi0R' -H 'Referer: https://www.pinnacle.com/' 2>/dev/null")
[ "$HTTP_PIN" = "200" ] && print_ok "Pinnacle responde 200" || print_err "Pinnacle devolvió $HTTP_PIN"

echo ""
echo "✅ Chequeo completo"
