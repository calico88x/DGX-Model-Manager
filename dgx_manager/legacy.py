from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from .config import Config

ENGINE_PATH_KEYS = {
    "sglang": "paths.legacy_sglang_scripts",
    "vllm": "paths.legacy_vllm_scripts",
    "llamacpp": "paths.legacy_llamacpp_scripts",
    "localai": "paths.legacy_localai_scripts",
    "comfyui": "paths.legacy_comfyui_scripts",
}


def parse_meta(script: Path) -> dict:
    name = description = None; vram = None
    try:
        for line in script.read_text(errors="replace").splitlines()[:30]:
            s = line.strip()
            if s.lower().startswith("# name:"): name = s.split(":",1)[1].strip()
            elif s.lower().startswith("# description:"): description = s.split(":",1)[1].strip()
            elif s.lower().startswith("# vram:"):
                try: vram = float(s.split(":",1)[1].strip().split()[0])
                except Exception: pass
    except Exception:
        pass
    stem = script.stem[6:] if script.stem.startswith("start_") else script.stem
    return {
        "id": "legacy:" + script.stem,
        "name": name or stem.replace("_"," ").replace("-"," ").title(),
        "description": description or f"Legacy script: {script.name}",
        "vram_gb": vram,
        "script": str(script),
        "kind": "legacy",
    }


def scan(config: Config, engine: str) -> list[dict]:
    if not config.get("app.legacy_scripts_enabled", False):
        return []
    key = ENGINE_PATH_KEYS.get(engine)
    if not key: return []
    d = config.path_value(key)
    if not d.exists(): return []
    return [parse_meta(p) for p in sorted(d.glob("start_*.sh")) if p.is_file()]


def launch(profile: dict, log_dir: Path) -> Path:
    script = Path(os.path.expanduser(profile["script"])).resolve()
    if not script.exists() or not script.is_file():
        raise FileNotFoundError(script)
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", profile["id"])
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f"legacy_{safe}.log"
    with log.open("w") as f:
        subprocess.Popen(["bash", str(script)], stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    return log
