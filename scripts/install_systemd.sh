#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="${SERVICE_NAME:-nsfw-bot}"
INSTALL_DIR="${INSTALL_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SERVICE_USER="${SERVICE_USER:-${SUDO_USER:-$(id -un)}}"
SERVICE_GROUP="${SERVICE_GROUP:-$(id -gn "$SERVICE_USER" 2>/dev/null || id -gn)}"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

if ! command -v systemctl >/dev/null 2>&1; then
  echo "systemctl not found. This installer requires a Linux VPS with systemd." >&2
  exit 1
fi

if [ "$(id -u)" -ne 0 ]; then
  echo "Run this installer with sudo: sudo -E bash scripts/install_systemd.sh" >&2
  exit 1
fi

if [ ! -f "${INSTALL_DIR}/.env" ]; then
  echo "Missing ${INSTALL_DIR}/.env. Create it from .env.example before installing the service." >&2
  exit 1
fi

if [ ! -f "${INSTALL_DIR}/start" ]; then
  echo "Missing ${INSTALL_DIR}/start." >&2
  exit 1
fi

chmod +x "${INSTALL_DIR}/start"
mkdir -p "${INSTALL_DIR}/data" "${INSTALL_DIR}/tmp"
chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "${INSTALL_DIR}/data" "${INSTALL_DIR}/tmp"

cat > "${SERVICE_FILE}" <<SERVICE
[Unit]
Description=Telegram NSFW Detection Bot
Wants=network-online.target
After=network-online.target
StartLimitIntervalSec=60
StartLimitBurst=10

[Service]
Type=simple
User=${SERVICE_USER}
Group=${SERVICE_GROUP}
WorkingDirectory=${INSTALL_DIR}
Environment=PYTHONUNBUFFERED=1
ExecStart=${INSTALL_DIR}/start
Restart=always
RestartSec=5
TimeoutStopSec=30
KillSignal=SIGINT

[Install]
WantedBy=multi-user.target
SERVICE

systemctl daemon-reload
systemctl enable "${SERVICE_NAME}"
systemctl restart "${SERVICE_NAME}"
systemctl --no-pager --full status "${SERVICE_NAME}"
