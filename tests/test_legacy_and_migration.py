from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from dgx_manager.config import Config
from dgx_manager.legacy import scan

ROOT = Path(__file__).resolve().parents[1]


def test_legacy_scan_requires_explicit_enable(tmp_path: Path):
    scripts = tmp_path / "SGLang"
    scripts.mkdir()
    script = scripts / "start_demo.sh"
    script.write_text("#!/bin/sh\n# Name: Demo Legacy\n# Description: Compatibility test\n# VRAM: 42\nexit 0\n")
    cfg = Config(tmp_path / "config.json")
    cfg.data["paths"]["legacy_sglang_scripts"] = str(scripts)
    cfg.data["app"]["legacy_scripts_enabled"] = False
    assert scan(cfg, "sglang") == []
    cfg.data["app"]["legacy_scripts_enabled"] = True
    rows = scan(cfg, "sglang")
    assert len(rows) == 1 and rows[0]["name"] == "Demo Legacy" and rows[0]["vram_gb"] == 42


def test_migration_merges_without_touching_v1_and_keeps_v2_identity(tmp_path: Path):
    old_root = tmp_path / "old-app"
    old_root.mkdir()
    scripts = tmp_path / "old-vllm"
    scripts.mkdir()
    (scripts / "start_existing.sh").write_text("#!/bin/sh\nexit 0\n")
    old = {
        "app": {"port": 8090},
        "services": {"ollama_base": "http://127.0.0.1:11434", "vllm_base": "http://127.0.0.1:8000"},
        "paths": {
            "hf_cache": str(tmp_path / "existing-hf"),
            "litellm_config": str(tmp_path / "existing-litellm.yaml"),
            "vllm_scripts": str(scripts),
        },
    }
    old_config = old_root / "config.json"
    old_text = json.dumps(old, indent=2) + "\n"
    old_config.write_text(old_text)
    (old_root / "custom_dirs.json").write_text(json.dumps([str(tmp_path / "models")]))

    target = tmp_path / "v2" / "config.json"
    cfg = Config(target)
    cfg.data["app"]["port"] = 8091
    cfg.data["paths"]["database"] = str(tmp_path / "v2" / "db.sqlite")
    cfg.data["paths"]["compose_root"] = str(tmp_path / "v2" / "compose")
    cfg.save()
    pre = target.read_text()

    cp = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "migrate_from_v1.py"), "--source", str(old_root), "--target", str(target)],
        text=True, capture_output=True, check=True,
    )
    assert "v1 was not modified" in cp.stdout
    assert old_config.read_text() == old_text
    assert target.with_suffix(".json.pre-migration.bak").read_text() == pre

    merged = json.loads(target.read_text())
    assert merged["app"]["port"] == 8091
    assert merged["paths"]["hf_cache"] == str(tmp_path / "existing-hf")
    assert merged["paths"]["litellm_config"] == str(tmp_path / "existing-litellm.yaml")
    assert merged["paths"]["legacy_vllm_scripts"] == str(scripts)
    assert merged["app"]["legacy_scripts_enabled"] is True
    assert merged["paths"]["compose_root"] == str(tmp_path / "v2" / "compose")

def test_promotion_dry_run_is_non_mutating(tmp_path: Path):
    target = tmp_path / "current-model-manager"
    target.mkdir()
    marker = target / "keep-me.txt"
    marker.write_text("original\n")
    before = sorted(x.name for x in tmp_path.iterdir())

    cp = subprocess.run(
        ["bash", str(ROOT / "scripts" / "promote_v2.sh"), "--target", str(target)],
        text=True, capture_output=True, check=True,
    )

    assert "DRY RUN ONLY" in cp.stdout
    assert "offline dependency wheelhouse" in cp.stdout
    assert marker.read_text() == "original\n"
    assert sorted(x.name for x in tmp_path.iterdir()) == before
