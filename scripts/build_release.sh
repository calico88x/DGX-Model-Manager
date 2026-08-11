#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST="${1:-$ROOT/dist}"

VERSION="$(PYTHONPATH="$ROOT" python3 - <<'PY'
from dgx_manager import __version__
print(__version__)
PY
)"

NAME="DGX-Model-Manager-v${VERSION}"
STAGE="$(mktemp -d -t dmm2-release-XXXXXX)"

trap 'rm -rf -- "$STAGE"' EXIT

command -v tar >/dev/null || {
  echo "tar is required" >&2
  exit 1
}

command -v sha256sum >/dev/null || {
  echo "sha256sum is required" >&2
  exit 1
}

printf '==> Validating source\n'
python3 "$ROOT/scripts/validate_release.py"

rm -rf -- "$DIST"
mkdir -p "$DIST" "$STAGE/$NAME"

# ---------------------------------------------------------------------------
# Stage public release material
#
# This repository may live beside runtime state, credentials, developer
# environments, acceptance artifacts, and generated release packages.
# Explicitly exclude those classes even when they happen to exist locally.
# ---------------------------------------------------------------------------

tar -C "$ROOT" \
  --exclude='./.git' \
  --exclude='./dist' \
  \
  --exclude='./venv' \
  --exclude='./.venv' \
  --exclude='./node_modules' \
  --exclude='*/node_modules' \
  \
  --exclude='./__pycache__' \
  --exclude='*/__pycache__' \
  --exclude='./.pytest_cache' \
  --exclude='*/.pytest_cache' \
  --exclude='./.mypy_cache' \
  --exclude='*/.mypy_cache' \
  --exclude='./.ruff_cache' \
  --exclude='*/.ruff_cache' \
  --exclude='./.cache' \
  --exclude='*/.cache' \
  --exclude='./.local' \
  --exclude='*/.local' \
  --exclude='*.pyc' \
  --exclude='*.pyo' \
  --exclude='./.coverage' \
  --exclude='./coverage.xml' \
  --exclude='./htmlcov' \
  \
  --exclude='./config.json' \
  --exclude='./config.local.json' \
  --exclude='./config.*.local.json' \
  --exclude='./.env' \
  --exclude='./.env.*' \
  --exclude='*/.env' \
  --exclude='*/.env.*' \
  \
  --exclude='*.key' \
  --exclude='*.pem' \
  --exclude='*.crt' \
  --exclude='*.p12' \
  --exclude='*.pfx' \
  --exclude='*.token' \
  --exclude='./secret.key' \
  --exclude='./certs' \
  \
  --exclude='*.db' \
  --exclude='*.db-shm' \
  --exclude='*.db-wal' \
  --exclude='*.sqlite' \
  --exclude='*.sqlite3' \
  \
  --exclude='./runtime' \
  --exclude='./logs' \
  --exclude='*/logs' \
  --exclude='*.log' \
  \
  --exclude='./acceptance' \
  --exclude='./release-notes-private' \
  --exclude='./docs/TEST_PLAN.md' \
  --exclude='./RELEASE_CHECKLIST.md' \
  \
  --exclude='./.vscode' \
  --exclude='./.idea' \
  --exclude='./.DS_Store' \
  --exclude='*/.DS_Store' \
  --exclude='./Thumbs.db' \
  --exclude='*/Thumbs.db' \
  --exclude='*~' \
  --exclude='*.swp' \
  \
  --exclude='./MANIFEST.sha256' \
  --exclude='./DGX-Model-Manager-v*.tar.gz' \
  --exclude='./DGX-Model-Manager-v*.tar.gz.sha256' \
  --exclude='./DGX-Model-Manager-v*.zip' \
  --exclude='./DGX-Model-Manager-v*.zip.sha256' \
  \
  -cf - . |
  tar -C "$STAGE/$NAME" -xf -

# ---------------------------------------------------------------------------
# Per-file integrity manifest
#
# MANIFEST.sha256 intentionally does not hash itself.
# ---------------------------------------------------------------------------

(
  cd "$STAGE/$NAME"

  find . \
    -type f \
    ! -name MANIFEST.sha256 \
    -print0 |
    sort -z |
    xargs -0 sha256sum > MANIFEST.sha256
)

TARBALL="$DIST/$NAME.tar.gz"
ZIPFILE="$DIST/$NAME.zip"

tar -C "$STAGE" -czf "$TARBALL" "$NAME"

# Build a portable ZIP while preserving Unix permission bits where possible.
python3 - "$STAGE" "$NAME" "$ZIPFILE" <<'PY'
from pathlib import Path
import sys
import zipfile

stage = Path(sys.argv[1])
name = sys.argv[2]
out = Path(sys.argv[3])
root = stage / name

with zipfile.ZipFile(
    out,
    "w",
    compression=zipfile.ZIP_DEFLATED,
    compresslevel=9,
) as zf:
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue

        rel = Path(name) / path.relative_to(root)
        info = zipfile.ZipInfo.from_file(
            path,
            rel.as_posix(),
        )

        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = (
            path.stat().st_mode & 0xFFFF
        ) << 16

        with path.open("rb") as src:
            zf.writestr(
                info,
                src.read(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )
PY

# Archive-level checksums.
(
  cd "$DIST"

  sha256sum "$(basename "$TARBALL")" \
    > "$(basename "$TARBALL").sha256"

  sha256sum "$(basename "$ZIPFILE")" \
    > "$(basename "$ZIPFILE").sha256"
)

printf '==> Release artifacts\n'
printf '    %s\n' \
  "$TARBALL" \
  "$TARBALL.sha256" \
  "$ZIPFILE" \
  "$ZIPFILE.sha256"

printf '    manifest: %s/MANIFEST.sha256 (inside each archive)\n' \
  "$NAME"