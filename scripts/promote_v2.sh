#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET=""
EXECUTE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) [[ $# -ge 2 ]] || { echo "--target requires a path" >&2; exit 2; }; TARGET="$2"; shift 2;;
    --execute) EXECUTE=1; shift;;
    *) echo "Unknown argument: $1" >&2; exit 2;;
  esac
done

if [[ -z "$TARGET" ]]; then
  echo "Usage: $0 --target /path/to/existing/DGX-Model-Manager [--execute]"
  echo "Without --execute this prints the promotion plan and changes nothing."
  exit 0
fi

TARGET="$(realpath -m "$TARGET")"
ROOT="$(realpath -m "$ROOT")"
if [[ "$TARGET" == "/" || "$TARGET" == "$HOME" ]]; then
  echo "Refusing unsafe promotion target: $TARGET" >&2
  exit 2
fi
if [[ "$TARGET" == "$ROOT" || "$TARGET" == "$ROOT/"* || "$ROOT" == "$TARGET/"* ]]; then
  echo "Promotion source and target must be separate directories: source=$ROOT target=$TARGET" >&2
  exit 2
fi
command -v python3 >/dev/null || { echo "python3 is required" >&2; exit 1; }
command -v tar >/dev/null || { echo "tar is required" >&2; exit 1; }

STAMP="$(date +%Y%m%d-%H%M%S)"
BACKUP="${TARGET}.pre-v2-${STAMP}"
STAGING="${TARGET}.v2-staging-${STAMP}"
PRECHECK_VENV="${TARGET}.v2-precheck-${STAMP}"
WHEELHOUSE="${TARGET}.v2-wheels-${STAMP}"
CONFIG="${XDG_CONFIG_HOME:-$HOME/.config}/dgx-model-manager-v2/config.json"
SERVICE_UNIT="/etc/systemd/system/model-manager.service"
SERVICE_BACKUP="${SERVICE_UNIT}.pre-v2-${STAMP}"

cat <<EOF
Promotion plan
--------------
Source v2 : $ROOT
Existing  : $TARGET
Backup    : $BACKUP
Staging   : $STAGING
Config    : $CONFIG
Service   : model-manager.service (replaces old service only after backup)
Port      : current v2 config will be changed to 8090 unless you edit it afterward

Safety sequence:
  1. Validate the v2 release and build an offline dependency wheelhouse while both apps stay running.
  2. Back up the existing service unit.
  3. Stop the coexistence/production services only at the cutover boundary.
  4. Move the existing app to the timestamped backup and install v2.
  5. Roll back the directory, port, service unit, and prior service state automatically if cutover fails.

Model data is not copied or moved. HuggingFace/Ollama paths remain external.
EOF

if [[ "$EXECUTE" -ne 1 ]]; then
  echo
  echo "DRY RUN ONLY. Re-run with --execute after testing v2 and reviewing the backup paths."
  exit 0
fi

read -r -p "Type PROMOTE-V2 to continue: " CONFIRM
[[ "$CONFIRM" == "PROMOTE-V2" ]] || { echo "Cancelled."; exit 1; }
[[ -f "$CONFIG" ]] || { echo "v2 config not found: $CONFIG" >&2; exit 1; }
[[ -d "$TARGET" ]] || { echo "Existing application directory not found: $TARGET" >&2; exit 1; }
[[ ! -e "$BACKUP" ]] || { echo "Backup path already exists: $BACKUP" >&2; exit 1; }
[[ ! -e "$STAGING" ]] || { echo "Staging path already exists: $STAGING" >&2; exit 1; }
[[ ! -e "$PRECHECK_VENV" ]] || { echo "Precheck venv path already exists: $PRECHECK_VENV" >&2; exit 1; }
[[ ! -e "$WHEELHOUSE" ]] || { echo "Wheelhouse path already exists: $WHEELHOUSE" >&2; exit 1; }

# Capture state before changing anything so a failed cutover can restore it.
OLD_PORT="$(python3 - <<PY
import json
p='$CONFIG'
d=json.load(open(p))
print(d.get('app',{}).get('port',8091))
PY
)"
OLD_SERVICE_ACTIVE=0
OLD_V2_ACTIVE=0
OLD_V2_ENABLED=0
sudo systemctl is-active --quiet model-manager 2>/dev/null && OLD_SERVICE_ACTIVE=1 || true
sudo systemctl is-active --quiet model-manager-v2 2>/dev/null && OLD_V2_ACTIVE=1 || true
sudo systemctl is-enabled --quiet model-manager-v2 2>/dev/null && OLD_V2_ENABLED=1 || true
OLD_UNIT_EXISTS=0
[[ -f "$SERVICE_UNIT" ]] && OLD_UNIT_EXISTS=1

# Preflight before downtime: source validation, source staging, and all Python wheels.
echo "==> Validating v2 release before cutover"
python3 "$ROOT/scripts/validate_release.py"
mkdir -p "$STAGING" "$WHEELHOUSE"
tar -C "$ROOT" \
  --exclude='./venv' --exclude='./.venv' \
  --exclude='./__pycache__' --exclude='*/__pycache__' \
  --exclude='./.pytest_cache' --exclude='./.git' \
  --exclude='./MANIFEST.sha256' \
  -cf - . | tar -C "$STAGING" -xf -
python3 -m venv "$PRECHECK_VENV"
"$PRECHECK_VENV/bin/pip" install --upgrade pip >/dev/null
"$PRECHECK_VENV/bin/pip" wheel -r "$STAGING/requirements.txt" --wheel-dir "$WHEELHOUSE"

echo "==> Dependency wheelhouse ready; beginning short cutover"
if [[ "$OLD_UNIT_EXISTS" -eq 1 ]]; then
  sudo cp "$SERVICE_UNIT" "$SERVICE_BACKUP"
fi

CUTOVER_STARTED=0
rollback() {
  local rc="${1:-1}"
  trap - ERR INT TERM
  set +e
  if [[ "$CUTOVER_STARTED" -eq 1 ]]; then
    echo "Promotion failed during cutover; restoring previous application/service state..." >&2
    sudo systemctl stop model-manager 2>/dev/null || true
    if [[ -d "$TARGET" ]]; then rm -rf -- "$TARGET"; fi
    if [[ -d "$BACKUP" ]]; then mv -- "$BACKUP" "$TARGET"; fi
    python3 - <<PY
import json
p='$CONFIG'
d=json.load(open(p))
d.setdefault('app',{})['port']=int('$OLD_PORT')
open(p,'w').write(json.dumps(d,indent=2)+'\\n')
PY
    if [[ "$OLD_UNIT_EXISTS" -eq 1 && -f "$SERVICE_BACKUP" ]]; then
      sudo cp "$SERVICE_BACKUP" "$SERVICE_UNIT"
    else
      sudo rm -f "$SERVICE_UNIT"
    fi
    sudo systemctl daemon-reload 2>/dev/null || true
    if [[ "$OLD_SERVICE_ACTIVE" -eq 1 ]]; then sudo systemctl start model-manager 2>/dev/null || true; fi
    if [[ "$OLD_V2_ENABLED" -eq 1 ]]; then sudo systemctl enable model-manager-v2 2>/dev/null || true; fi
    if [[ "$OLD_V2_ACTIVE" -eq 1 ]]; then sudo systemctl start model-manager-v2 2>/dev/null || true; fi
  fi
  rm -rf -- "$STAGING" "$PRECHECK_VENV" "$WHEELHOUSE"
  exit "$rc"
}
trap 'rollback $?' ERR INT TERM

CUTOVER_STARTED=1
sudo systemctl disable --now model-manager-v2 2>/dev/null || true
sudo systemctl stop model-manager 2>/dev/null || true
mv -- "$TARGET" "$BACKUP"
mv -- "$STAGING" "$TARGET"

python3 - <<PY
import json, os
p='$CONFIG'
d=json.load(open(p))
d.setdefault('app',{})['port']=8090
tmp=p+'.promote.tmp'
open(tmp,'w').write(json.dumps(d,indent=2)+'\\n')
os.chmod(tmp,0o600)
os.replace(tmp,p)
PY

python3 -m venv "$TARGET/venv"
"$TARGET/venv/bin/pip" install --no-index --find-links "$WHEELHOUSE" -r "$TARGET/requirements.txt"

USER_NAME="$(id -un)"
NO_NEW_PRIVILEGES=true
if [[ -f /etc/sudoers.d/dgx-model-manager-v2-litellm ]]; then
  NO_NEW_PRIVILEGES=false
fi
sudo tee "$SERVICE_UNIT" >/dev/null <<UNIT
[Unit]
Description=DGX Model Manager v2
After=network-online.target docker.service
Wants=network-online.target
[Service]
Type=simple
User=$USER_NAME
WorkingDirectory=$TARGET
Environment=DMM_CONFIG=$CONFIG
ExecStart=$TARGET/venv/bin/python $TARGET/app.py
Restart=on-failure
RestartSec=5
NoNewPrivileges=$NO_NEW_PRIVILEGES
PrivateTmp=true
ProtectSystem=full
ProtectHome=false
[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload
sudo systemctl enable --now model-manager
sudo systemctl is-active --quiet model-manager

# The production service is active; automatic rollback is no longer armed.
CUTOVER_STARTED=0
trap - ERR INT TERM
rm -rf -- "$PRECHECK_VENV" "$WHEELHOUSE"

echo "Promotion complete. Backup retained at $BACKUP"
if [[ "$OLD_UNIT_EXISTS" -eq 1 ]]; then
  echo "Previous service unit retained at $SERVICE_BACKUP"
fi
echo "The coexistence service model-manager-v2 has been disabled to prevent a port conflict after reboot."
echo "Rollback: stop model-manager, restore $BACKUP and the backed-up service unit, restore the v2 config port to $OLD_PORT, then daemon-reload/start."
