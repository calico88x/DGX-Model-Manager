#!/usr/bin/env bash
# DGX Model Manager — setup script
# Run once: bash setup.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/venv"

echo "==> DGX Model Manager Setup"
echo "    Directory: $SCRIPT_DIR"
echo ""

# ── Python venv ───────────────────────────────────────────────────────────────
echo "==> Creating Python venv"
python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip -q
"$VENV/bin/pip" install -r "$SCRIPT_DIR/requirements.txt"
echo "    Done."

# ── UFW (optional) ────────────────────────────────────────────────────────────
if command -v ufw &>/dev/null; then
  PORT=$(python3 -c "import json; d=json.load(open('$SCRIPT_DIR/config.json')); print(d.get('app',{}).get('port',8090))" 2>/dev/null || echo 8090)
  echo ""
  echo "==> UFW detected — adding rule for port $PORT"
  echo "    Edit the subnet below to match your network before confirming."
  read -r -p "    Allow port $PORT from subnet [192.168.1.0/24]: " SUBNET
  SUBNET="${SUBNET:-192.168.1.0/24}"
  sudo ufw allow from "$SUBNET" to any port "$PORT" proto tcp
  echo "    UFW rule added."
fi

# ── Sudoers (optional — for LiteLLM restart button) ───────────────────────────
echo ""
read -r -p "==> Add passwordless sudo for 'systemctl restart litellm'? [y/N]: " ADD_SUDO
if [[ "$ADD_SUDO" =~ ^[Yy]$ ]]; then
  USER_NAME="$(whoami)"
  SUDOERS_LINE="$USER_NAME ALL=(ALL) NOPASSWD: /bin/systemctl restart litellm, /bin/systemctl restart litellm.service"
  echo "$SUDOERS_LINE" | sudo tee /etc/sudoers.d/model-manager-litellm > /dev/null
  sudo chmod 440 /etc/sudoers.d/model-manager-litellm
  echo "    Sudoers entry created."
fi

# ── Systemd service ───────────────────────────────────────────────────────────
echo ""
read -r -p "==> Install as systemd service (starts on boot)? [y/N]: " ADD_SERVICE
if [[ "$ADD_SERVICE" =~ ^[Yy]$ ]]; then
  USER_NAME="$(whoami)"
  cat <<UNIT | sudo tee /etc/systemd/system/model-manager.service > /dev/null
[Unit]
Description=DGX Model Manager
After=network.target ollama.service
Wants=ollama.service

[Service]
Type=simple
User=$USER_NAME
WorkingDirectory=$SCRIPT_DIR
ExecStart=$VENV/bin/python3 $SCRIPT_DIR/app.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT

  sudo systemctl daemon-reload
  sudo systemctl enable model-manager
  sudo systemctl start model-manager
  echo "    Service installed and started."
  echo ""
  sudo systemctl status model-manager --no-pager
else
  echo ""
  echo "==> To run manually:"
  echo "    $VENV/bin/python3 $SCRIPT_DIR/app.py"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
PORT=$(python3 -c "import json; d=json.load(open('$SCRIPT_DIR/config.json')); print(d.get('app',{}).get('port',8090))" 2>/dev/null || echo 8090)
echo ""
echo "==> Setup complete."
echo "    Open: http://$(hostname -I | awk '{print $1}'):$PORT"
echo ""
