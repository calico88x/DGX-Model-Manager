#!/usr/bin/env python3
"""Import compatible v1 settings into the independent v2 test configuration.

This script never edits the v1 installation.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT))
from dgx_manager.config import Config, default_config_path


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",required=True,help="Path to existing Model Manager directory or its config.json")
    ap.add_argument("--target",default=str(default_config_path()),help="v2 config path")
    ap.add_argument("--force",action="store_true",help="Accepted for compatibility; existing v2 config is always backed up before merge")
    args=ap.parse_args()
    src=Path(os.path.expanduser(args.source)).resolve()
    if src.is_dir(): src=src/"config.json"
    if not src.exists(): raise SystemExit(f"Source config not found: {src}")
    old=json.loads(src.read_text())
    target=Path(os.path.expanduser(args.target)).resolve()
    cfg=Config(target)
    if target.exists():
        backup=target.with_suffix(target.suffix+".pre-migration.bak")
        shutil.copy2(target,backup)
        print(f"Backed up existing v2 config to: {backup}")
    # Preserve v2 coexistence identity: port, TLS, database, compose root and service remain v2-specific.
    for key,val in (old.get("services") or {}).items():
        if key in cfg.data["services"]: cfg.data["services"][key]=val
    old_paths=old.get("paths") or {}
    for key in ("litellm_config","hf_cache"):
        if key in old_paths: cfg.data["paths"][key]=old_paths[key]
    # v1/v2.0 script-directory settings are imported only as legacy compatibility paths.
    legacy_path_map={
        "sglang_scripts":"legacy_sglang_scripts",
        "vllm_scripts":"legacy_vllm_scripts",
        "llamacpp_scripts":"legacy_llamacpp_scripts",
        "localai_scripts":"legacy_localai_scripts",
        "comfyui_scripts":"legacy_comfyui_scripts",
    }
    for old_key,new_key in legacy_path_map.items():
        if old_key in old_paths: cfg.data["paths"][new_key]=old_paths[old_key]
    old_root=src.parent
    old_custom=old_root/"custom_dirs.json"
    if old_custom.exists():
        dst=cfg.path_value("paths.custom_dirs"); dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(old_custom,dst); os.chmod(dst,0o600)
    # Detect legacy scripts without changing them.
    legacy=False
    for key in ("sglang","vllm","llamacpp","localai","comfyui"):
        p=cfg.path_value(f"paths.legacy_{key}_scripts")
        if p.is_dir() and any(p.glob("start_*.sh")): legacy=True
    cfg.data["app"]["legacy_scripts_enabled"]=legacy
    cfg.save()
    print(f"Imported v1 service/model paths into: {target}")
    print("v1 was not modified.")
    print(f"Legacy Script Mode: {'enabled (existing scripts detected)' if legacy else 'disabled'}")
    print(f"v2 remains on port {cfg.get('app.port')} by default so both applications can coexist.")

if __name__=="__main__": main()
