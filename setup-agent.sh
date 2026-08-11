#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/venv"
CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/dgx-model-manager-v2"
AGENT_CONFIG="$CONFIG_DIR/agent.json"
CERT_DIR="$CONFIG_DIR/agent-certs"
mkdir -p "$CONFIG_DIR" "$CERT_DIR"
chmod 700 "$CONFIG_DIR" "$CERT_DIR" 2>/dev/null || true
if ! command -v openssl >/dev/null 2>&1; then
  echo "ERROR: openssl is required to create the node-agent HTTPS certificate." >&2
  exit 1
fi
if ! command -v docker >/dev/null 2>&1; then
  echo "WARNING: Docker was not found. The node agent can start, but deployment lifecycle operations will be unavailable." >&2
elif ! docker compose version >/dev/null 2>&1; then
  echo "WARNING: Docker Compose v2 was not detected. Install the Docker Compose plugin before testing remote Compose deployments." >&2
fi
[[ -d "$VENV" ]] || python3 -m venv "$VENV"
"$VENV/bin/pip" install -r "$ROOT/requirements.txt"
TOKEN="$($VENV/bin/python - <<'PY'
import secrets
print('node_'+secrets.token_urlsafe(36))
PY
)"
HASH="$($VENV/bin/python - <<PY
import hashlib
print(hashlib.sha256('$TOKEN'.encode()).hexdigest())
PY
)"
CERT="$CERT_DIR/server.crt"; KEY="$CERT_DIR/server.key"
HOST="$(hostname -f 2>/dev/null || hostname)"; IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
SAN="DNS:$HOST,DNS:localhost,IP:127.0.0.1"; [[ -n "$IP" ]] && SAN="$SAN,IP:$IP"
openssl req -x509 -newkey rsa:3072 -sha256 -days 825 -nodes -keyout "$KEY" -out "$CERT" -subj "/CN=$HOST" -addext "subjectAltName=$SAN" >/dev/null 2>&1
chmod 600 "$KEY"; chmod 644 "$CERT"
FINGERPRINT="$(openssl x509 -in "$CERT" -noout -fingerprint -sha256 | cut -d= -f2)"
cat > "$AGENT_CONFIG" <<JSON
{
  "name": "$(hostname)",
  "host": "0.0.0.0",
  "port": 8092,
  "token_hash": "$HASH",
  "allow_insecure_http": false,
  "max_request_bytes": 2097152,
  "manager_config": "$CONFIG_DIR/config.json",
  "tls": {
    "enabled": true,
    "cert_file": "$CERT",
    "key_file": "$KEY"
  }
}
JSON
chmod 600 "$AGENT_CONFIG"
USER_NAME="$(id -un)"
sudo tee /etc/systemd/system/model-manager-v2-agent.service >/dev/null <<UNIT
[Unit]
Description=DGX Model Manager v2 Node Agent
After=network-online.target docker.service
Wants=network-online.target
[Service]
Type=simple
User=$USER_NAME
WorkingDirectory=$ROOT
Environment=DMM_AGENT_CONFIG=$AGENT_CONFIG
ExecStart=$VENV/bin/python $ROOT/agent.py
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
sudo systemctl enable --now model-manager-v2-agent
cat <<EOF

Node agent installed.

Enrollment token (shown once):
  $TOKEN

Self-signed certificate SHA-256 fingerprint:
  $FINGERPRINT

In the manager UI add this node using https://$IP:8092.
Because this setup uses a self-signed certificate, disable standard CA verification and paste the fingerprint above.
The manager will pin and verify that fingerprint before sending the enrollment token.
EOF
