#!/usr/bin/env python3
"""Create or rotate the one-time first-admin bootstrap token locally."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dgx_manager.config import Config
from dgx_manager.db import Database


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="v2 config.json path")
    ap.add_argument("--rotate", action="store_true", help="replace an unusable/unseen bootstrap token")
    args = ap.parse_args()
    if args.config:
        os.environ["DMM_CONFIG"] = str(Path(args.config).expanduser().resolve())
    cfg = Config()
    cfg.ensure_directories()
    db = Database(cfg)
    if db.user_count() != 0:
        print("Bootstrap is already complete; an administrator account exists.", file=sys.stderr)
        return 0
    token, path, _ = cfg.ensure_bootstrap_token(rotate=args.rotate)
    if token is None:
        print(f"Bootstrap token hash exists but {path} is unavailable. Re-run with --rotate.")
        return 2
    # stdout intentionally contains only the token so setup.sh can capture it.
    print(token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
