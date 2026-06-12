#!/usr/bin/env bash
# Deploy ejecutado en el VPS por el workflow de GitHub Actions vía:
#   ssh ... 'bash -s' < deploy/remote_deploy.sh
# (se pipea la versión recién mergeada, así corre siempre la lógica más nueva).
#
# Secuencia:
#   1) safe_pull.sh — git pull + validación de imports + rollback automático si falla.
#      El scheduler es un timer→oneshot: el código nuevo toma efecto SOLO en el próximo
#      tick (proceso fresco cada 5 min), así que el pull en sí ya despliega.
#   2) daemon-reload — por si cambiaron los unit files (.service/.timer).
#   3) restart del timer — SOLO si no hay un partido en ventana de publicación. Si lo hay,
#      no se toca nada: el pull ya surte efecto al próximo tick y reiniciar el .timer no
#      mata un .service corriendo, pero evitamos ruido innecesario durante una publicación.
set -euo pipefail
cd /opt/penca

echo "▸ safe_pull"
./deploy/safe_pull.sh

echo "▸ daemon-reload"
systemctl daemon-reload

echo "▸ preflight (ventana de publicación)"
if .venv/bin/python scripts/preflight_deploy.py; then
    systemctl restart penca-scheduler.timer
    echo "✅ Deploy completo — timer reiniciado"
else
    echo "⏳ Partido en ventana de publicación: código ya pulleado (toma efecto al próximo tick); restart del timer diferido al próximo deploy o reinicio."
fi
