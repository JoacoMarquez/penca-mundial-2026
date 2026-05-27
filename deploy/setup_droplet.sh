#!/usr/bin/env bash
# Setup del droplet Ubuntu 24.04 para correr la pipeline de la penca.
# Idempotente: corre seguro múltiples veces.

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/joacomarquez/penca-mundial-2026.git}"  # ajustar al repo real
INSTALL_DIR="/opt/penca"
DATA_DIR="/var/lib/penca"
ENV_DIR="/etc/penca"

echo "==> System update"
apt-get update -y
apt-get upgrade -y

echo "==> Dependencias del sistema"
apt-get install -y \
    python3.12 python3.12-venv python3-pip \
    git curl ca-certificates \
    libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 libxkbcommon0 \
    libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libpango-1.0-0 \
    libcairo2 libasound2t64 \
    fonts-liberation

echo "==> Carpetas"
mkdir -p "$INSTALL_DIR" "$DATA_DIR" "$ENV_DIR"
mkdir -p "$DATA_DIR"/{raw,processed,predictions,logs}

echo "==> Clonar repo (o pull si existe)"
if [ -d "$INSTALL_DIR/.git" ]; then
    cd "$INSTALL_DIR" && git pull
else
    git clone "$REPO_URL" "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi

echo "==> Virtualenv + deps"
if [ ! -d "$INSTALL_DIR/.venv" ]; then
    python3.12 -m venv "$INSTALL_DIR/.venv"
fi
"$INSTALL_DIR/.venv/bin/pip" install --upgrade pip
"$INSTALL_DIR/.venv/bin/pip" install -r "$INSTALL_DIR/requirements.txt"
"$INSTALL_DIR/.venv/bin/playwright" install chromium

echo "==> .env"
if [ ! -f "$ENV_DIR/env" ]; then
    cp "$INSTALL_DIR/.env.example" "$ENV_DIR/env"
    chmod 600 "$ENV_DIR/env"
    echo ""
    echo "!! AHORA edita $ENV_DIR/env con tus secrets reales (TELEGRAM, ANTHROPIC, PENCA, etc.)"
    echo "!! Después correr de nuevo este script o:"
    echo "   systemctl daemon-reload && systemctl enable --now penca-scheduler.timer"
    exit 0
fi

echo "==> systemd units"
cp "$INSTALL_DIR/deploy/penca-scheduler.service" /etc/systemd/system/
cp "$INSTALL_DIR/deploy/penca-scheduler.timer" /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now penca-scheduler.timer

echo ""
echo "✅ Setup completo."
echo "   Logs:  journalctl -u penca-scheduler -f"
echo "   Status: systemctl status penca-scheduler.timer"
