#!/usr/bin/env bash
# destroy_droplet.sh — Destruye el droplet de DigitalOcean para dejar de pagar.
# Se corre DESDE TU MAC. IRREVERSIBLE: pide confirmación escrita.
#
# Uso:
#   bash deploy/destroy_droplet.sh              # lista droplets y pide cuál destruir
#   bash deploy/destroy_droplet.sh --snapshot   # snapshot antes de destruir (~$0.20/mes)
#
# Requisitos: DO_API_TOKEN en .env (raíz del repo) o en el entorno.
# PRE-REQUISITO: haber corrido shutdown_all.sh y backup_vps.sh.
set -euo pipefail

DO_API=https://api.digitalocean.com/v2
SNAPSHOT=0
[[ "${1:-}" == "--snapshot" ]] && SNAPSHOT=1

TOKEN="${DO_API_TOKEN:-}"
if [[ -z "$TOKEN" && -f .env ]]; then
  TOKEN=$(sed -n 's/^DO_API_TOKEN=//p' .env | head -1 | tr -d '"\r ' | tr -d "'")
fi
if [[ -z "$TOKEN" ]]; then
  echo "✗ Falta DO_API_TOKEN (ponelo en .env o exportalo)."
  exit 1
fi

api() { curl -sS -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" "$@"; }

list_droplets() {
  python3 <<'PY'
import json, sys
d = json.load(sys.stdin).get("droplets", [])
if not d:
    print("  (ninguno)")
for x in d:
    ip = next((n["ip_address"] for n in x["networks"]["v4"] if n["type"] == "public"), "?")
    print("  id={}  name={}  ip={}  size={}  status={}  created={}".format(
        x["id"], x["name"], ip, x["size_slug"], x["status"], x["created_at"]))
PY
}

echo "=== Droplets en tu cuenta ==="
api "$DO_API/droplets?per_page=200" | list_droplets

echo
read -rp "ID del droplet a destruir (Enter para abortar): " DID
[[ -z "$DID" ]] && { echo "Abortado."; exit 0; }

NAME=$(api "$DO_API/droplets/$DID" | python3 -c 'import json,sys; print(json.load(sys.stdin)["droplet"]["name"])')
echo
echo "⚠️  Vas a DESTRUIR el droplet: id=$DID  name=$NAME"
echo "    Esto borra el disco. Asegurate de haber corrido deploy/backup_vps.sh."
read -rp "Escribí el nombre del droplet para confirmar: " CONFIRM
if [[ "$CONFIRM" != "$NAME" ]]; then
  echo "No coincide. Abortado — no se destruyó nada."
  exit 1
fi

if [[ "$SNAPSHOT" == "1" ]]; then
  SNAP_NAME="penca-final-$(date +%Y%m%d)"
  echo "› Creando snapshot '$SNAP_NAME' (puede tardar varios minutos)…"
  api -X POST -d "{\"type\":\"snapshot\",\"name\":\"$SNAP_NAME\"}" "$DO_API/droplets/$DID/actions" >/dev/null
  echo "  Esperando a que termine…"
  while true; do
    ST=$(api "$DO_API/droplets/$DID/actions?per_page=1" | python3 -c 'import json,sys; print(json.load(sys.stdin)["actions"][0]["status"])')
    [[ "$ST" == "completed" ]] && { echo "  ✓ snapshot listo"; break; }
    [[ "$ST" == "errored" ]] && { echo "  ✗ snapshot falló — NO destruyo"; exit 1; }
    sleep 15
  done
fi

echo "› Destruyendo droplet $DID…"
CODE=$(curl -sS -o /dev/null -w '%{http_code}' -X DELETE \
  -H "Authorization: Bearer $TOKEN" "$DO_API/droplets/$DID")
if [[ "$CODE" == "204" ]]; then
  echo "✅ Droplet destruido. Ya no se factura."
else
  echo "✗ DigitalOcean respondió HTTP $CODE — revisá el panel."
  exit 1
fi

echo
echo "Para levantar de nuevo, seguí deploy/RESUME.md"
