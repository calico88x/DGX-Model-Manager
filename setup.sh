#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/venv"
CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/dgx-model-manager-v2"
DATA_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/dgx-model-manager-v2"
CONFIG="$CONFIG_DIR/config.json"
CERT_DIR="$CONFIG_DIR/certs"
SERVICE_NAME="model-manager-v2"

printf '\n== DGX Model Manager v2 setup ==\n'
printf 'Project: %s\n' "$ROOT"
printf 'Config : %s\n' "$CONFIG"
printf 'Data   : %s\n\n' "$DATA_DIR"

mkdir -p "$CONFIG_DIR" "$DATA_DIR" "$CERT_DIR"
chmod 700 "$CONFIG_DIR" "$DATA_DIR" "$CERT_DIR" 2>/dev/null || true

if [[ ! -f "$CONFIG" ]]; then
  cp "$ROOT/config.example.json" "$CONFIG"
  chmod 600 "$CONFIG"
  echo "Created independent v2 config. Existing Model Manager config was not modified."
else
  echo "Existing v2 config retained."
fi

if [[ ! -d "$VENV" ]]; then
  python3 -m venv "$VENV"
fi
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install -r "$ROOT/requirements.txt"

if ! command -v docker >/dev/null 2>&1; then
  echo "WARNING: Docker was not found. The UI can run, but Compose deployment lifecycle controls will be unavailable." >&2
elif ! docker compose version >/dev/null 2>&1; then
  echo "WARNING: Docker Compose v2 was not detected. Install the Docker Compose plugin before testing Compose deployments." >&2
fi

# Generate an encrypted transport for the test install if no cert exists.
CERT="$CERT_DIR/server.crt"
KEY="$CERT_DIR/server.key"
if [[ ! -f "$CERT" || ! -f "$KEY" ]]; then
  if ! command -v openssl >/dev/null 2>&1; then
    echo "ERROR: openssl is required to create the default HTTPS certificate." >&2
    exit 1
  fi
  HOSTNAME_FQDN="$(hostname -f 2>/dev/null || hostname)"
  LOCAL_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
  SAN="DNS:$HOSTNAME_FQDN,DNS:localhost,IP:127.0.0.1"
  [[ -n "$LOCAL_IP" ]] && SAN="$SAN,IP:$LOCAL_IP"
  openssl req -x509 -newkey rsa:3072 -sha256 -days 825 -nodes \
    -keyout "$KEY" -out "$CERT" \
    -subj "/CN=$HOSTNAME_FQDN" \
    -addext "subjectAltName=$SAN" >/dev/null 2>&1
  chmod 600 "$KEY"; chmod 644 "$CERT"
  echo "Generated self-signed HTTPS certificate for the v2 test install."
fi

PORT="$($VENV/bin/python - <<PY
import json
print(json.load(open('$CONFIG')).get('app',{}).get('port',8091))
PY
)"

BOOTSTRAP_TOKEN="$($VENV/bin/python "$ROOT/scripts/bootstrap_token.py" --config "$CONFIG")"

echo
read -r -p "Install ${SERVICE_NAME}.service (independent from model-manager.service)? [Y/n]: " INSTALL_SERVICE
INSTALL_SERVICE="${INSTALL_SERVICE:-Y}"
if [[ "$INSTALL_SERVICE" =~ ^[Yy]$ ]]; then
  USER_NAME="$(id -un)"
  sudo tee "/etc/systemd/system/${SERVICE_NAME}.service" >/dev/null <<UNIT
[Unit]
Description=DGX Model Manager v2 (test/coexistence install)
After=network-online.target docker.service
Wants=network-online.target

[Service]
Type=simple
User=$USER_NAME
WorkingDirectory=$ROOT
Environment=DMM_CONFIG=$CONFIG
ExecStart=$VENV/bin/python $ROOT/app.py
Restart=on-failure
RestartSec=5
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=full
ProtectHome=false

[Install]
WantedBy=multi-user.target
UNIT
  sudo systemctl daemon-reload
  sudo systemctl enable --now "$SERVICE_NAME"
  echo "Installed ${SERVICE_NAME}.service. Existing model-manager.service was not changed."
fi

echo
read -r -p "Allow v2 to authenticate to LiteLLM using its existing master key? [y/N]: " ADD_LITELLM_AUTH
ADD_LITELLM_AUTH="${ADD_LITELLM_AUTH:-N}"
if [[ "$INSTALL_SERVICE" =~ ^[Yy]$ && "$ADD_LITELLM_AUTH" =~ ^[Yy]$ ]]; then
  LITELLM_ENV="${LITELLM_ENV:-/etc/litellm/litellm.env}"
  CRED_DIR="/etc/dgx-model-manager-v2"
  CRED_FILE="$CRED_DIR/litellm_master_key"

  if sudo test -r "$LITELLM_ENV"; then
    sudo bash -c '
      set -a
      source "$1"
      : "${LITELLM_MASTER_KEY:?LITELLM_MASTER_KEY is not set}"
      umask 077
      install -d -m 700 -o root -g root "$2"
      printf "%s" "$LITELLM_MASTER_KEY" > "$3"
      chmod 600 "$3"
      chown root:root "$3"
    ' _ "$LITELLM_ENV" "$CRED_DIR" "$CRED_FILE"

    sudo mkdir -p "/etc/systemd/system/${SERVICE_NAME}.service.d"
    sudo tee "/etc/systemd/system/${SERVICE_NAME}.service.d/litellm-auth.conf" >/dev/null <<EOF
[Service]
LoadCredential=litellm_master_key:$CRED_FILE
EOF

    sudo systemctl daemon-reload
    sudo systemctl restart "$SERVICE_NAME"
    echo "Installed isolated LiteLLM master-key credential for ${SERVICE_NAME}.service."
  else
    echo "LiteLLM environment file is unavailable: $LITELLM_ENV"
    echo "Skipping LiteLLM API authentication setup."
  fi
fi

echo
read -r -p "Allow v2 to restart LiteLLM with a restricted passwordless sudo rule? [y/N]: " ADD_SUDO
if [[ "$ADD_SUDO" =~ ^[Yy]$ ]]; then
  USER_NAME="$(id -un)"
  SYSTEMCTL_BIN="$(command -v systemctl)"
  SUDOERS="/etc/sudoers.d/dgx-model-manager-v2-litellm"
  echo "$USER_NAME ALL=(root) NOPASSWD: $SYSTEMCTL_BIN restart litellm, $SYSTEMCTL_BIN restart litellm.service" | sudo tee "$SUDOERS" >/dev/null
  sudo chmod 440 "$SUDOERS"
  sudo visudo -cf "$SUDOERS" >/dev/null
  echo "Restricted LiteLLM restart rule installed."
  # NoNewPrivileges blocks sudo from acquiring privileges. Keep the stronger default
  # unless the operator explicitly opted into the narrowly-scoped restart capability.
  sudo mkdir -p "/etc/systemd/system/${SERVICE_NAME}.service.d"
  sudo tee "/etc/systemd/system/${SERVICE_NAME}.service.d/litellm-restart.conf" >/dev/null <<'OVERRIDE'
[Service]
NoNewPrivileges=false
OVERRIDE
  sudo systemctl daemon-reload
  sudo systemctl restart "$SERVICE_NAME" 2>/dev/null || true
  echo "Installed a service override allowing only the explicitly configured sudo capability to function."
fi

SCHEME="https"
echo
printf 'v2 is ready at: %s://<this-node>:%s\n' "$SCHEME" "$PORT"
echo "The generated certificate is self-signed, so your browser will show a trust warning until you install/replace it."
if [[ -n "$BOOTSTRAP_TOKEN" ]]; then
  echo
  echo "First-run administrator bootstrap token (shown from the local mode-0600 token file):"
  echo "  $BOOTSTRAP_TOKEN"
  echo "Enter this token on the first-run account screen. It is deleted after successful bootstrap."
fi
echo "v1 can continue running on its existing port/service during testing."
echo
printf 'Useful commands:\n'
printf '  sudo systemctl status %s\n' "$SERVICE_NAME"
printf '  sudo journalctl -u %s -f\n' "$SERVICE_NAME"
printf '  %s/scripts/migrate_from_v1.py --help\n' "$ROOT"
