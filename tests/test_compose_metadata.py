from __future__ import annotations

import json
from pathlib import Path

from dgx_manager.compose_manager import ComposeManager
from dgx_manager.config import Config
from dgx_manager.inventory import parse_hf_model, parse_flat_model


def make_config(tmp_path: Path) -> Config:
    cfg = Config(tmp_path / "config.json")
    cfg.data["paths"]["compose_root"] = str(tmp_path / "compose")
    cfg.data["paths"]["hf_cache"] = str(tmp_path / "hf" / "hub")
    cfg.save()
    return cfg


def metrics() -> dict:
    return {"memory_total_gb": 128, "unified_memory": True, "gpu": {"name": "NVIDIA GB10"}}


def test_hf_quantization_metadata_and_sglang_modelopt_flag(tmp_path: Path):
    cfg = make_config(tmp_path)
    repo = tmp_path / "hf" / "hub" / "models--example--Model-NVFP4"
    snap = repo / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text(json.dumps({
        "model_type": "example",
        "torch_dtype": "bfloat16",
        "num_parameters": 35_000_000_000,
        "quantization_config": {"quant_method": "modelopt", "format": "nvfp4", "bits": 4},
    }))
    (snap / "model.safetensors").write_bytes(b"checkpoint")

    model = parse_hf_model(repo)
    assert model["quant_method"] == "modelopt_fp4"
    assert model["quant_bits"] == 4
    assert model["dtype"] == "FP4"
    assert model["params_b"] == 35.0
    assert model["params_estimated"] is False

    plan = ComposeManager(cfg).generate(model=model, engine_key="sglang", node_metrics=metrics())
    command = plan["yaml"]
    assert "--quantization" in command
    assert "modelopt_fp4" in command
    assert plan["quant_method"] == "modelopt_fp4"
    assert any("ModelOpt" in note for note in plan["notes"])


def test_vllm_relies_on_checkpoint_quantization_autodetection(tmp_path: Path):
    cfg = make_config(tmp_path)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    model = {
        "id": "custom:q", "name": "q", "full_name": "example/q-NVFP4",
        "runtime_path": str(model_dir), "dir_path": str(model_dir), "source": "custom_dir",
        "format": "safetensors", "dtype": "FP4", "params_b": 35, "size_gb": 25,
        "quant_method": "modelopt_fp4", "quant_bits": 4,
    }
    plan = ComposeManager(cfg).generate(model=model, engine_key="vllm", node_metrics=metrics())
    # vLLM documents quantization_config auto-detection, so do not force a CLI method.
    assert "--quantization" not in plan["yaml"]
    assert any("auto-detect" in note for note in plan["notes"])


def test_llamacpp_requires_and_mounts_gguf(tmp_path: Path):
    cfg = make_config(tmp_path)
    model_dir = tmp_path / "gguf-model"
    model_dir.mkdir()
    (model_dir / "demo.Q4_K_M.gguf").write_bytes(b"gguf")
    model = parse_flat_model(model_dir)
    plan = ComposeManager(cfg).generate(model=model, engine_key="llamacpp", node_metrics=metrics(), context_length=8192)
    assert "demo.Q4_K_M.gguf" in plan["yaml"]
    assert "--n-gpu-layers" in plan["yaml"]
    assert "127.0.0.1:8080:8080" in plan["yaml"]


def test_localai_selected_hf_repo_mount_preserves_snapshot_symlinks_scope(tmp_path: Path):
    cfg = make_config(tmp_path)
    repo = tmp_path / "hf" / "hub" / "models--example--local"
    snap = repo / "snapshots" / "deadbeef"
    blobs = repo / "blobs"
    snap.mkdir(parents=True); blobs.mkdir()
    (snap / "config.json").write_text("{}")
    model = parse_hf_model(repo)
    plan = ComposeManager(cfg).generate(model=model, engine_key="localai", node_metrics=metrics())
    assert f"{repo}:/models/selected:ro" in plan["yaml"]
    # The broad HF cache mount used by vLLM/SGLang is intentionally replaced for LocalAI.
    assert ":/root/.cache/huggingface:ro" not in plan["yaml"]
    assert "NVIDIA_DRIVER_CAPABILITIES" in plan["yaml"]


def test_generated_plan_validation_rejects_compose_tampering(tmp_path: Path):
    import copy
    import pytest
    import yaml

    cfg = make_config(tmp_path)
    model_dir = tmp_path / "trusted-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    model = {
        "id": "custom:trusted", "name": "trusted", "full_name": "example/trusted",
        "runtime_path": str(model_dir), "dir_path": str(model_dir), "source": "custom_dir",
        "format": "safetensors", "dtype": "BF16", "params_b": 8, "size_gb": 16,
        "quant_method": None, "quant_bits": None,
    }
    manager = ComposeManager(cfg)
    plan = manager.generate(model=model, engine_key="vllm", node_metrics=metrics(), name="trusted-service")

    # An untouched server-generated plan is accepted and rebuilt from trusted inputs.
    validated = manager.validate_generated_plan(plan, model=model, node_metrics=metrics())
    assert yaml.safe_load(validated["yaml"]) == yaml.safe_load(plan["yaml"])

    # A browser/operator cannot smuggle a host mount or arbitrary Compose privilege into save.
    tampered = copy.deepcopy(plan)
    doc = yaml.safe_load(tampered["yaml"])
    doc["services"]["inference"].setdefault("volumes", []).append("/etc:/host-etc:rw")
    doc["services"]["inference"]["privileged"] = True
    tampered["yaml"] = yaml.safe_dump(doc, sort_keys=False)
    with pytest.raises(ValueError, match="modified after generation"):
        manager.validate_generated_plan(tampered, model=model, node_metrics=metrics())


def test_generated_plan_validation_rejects_metadata_tampering(tmp_path: Path):
    import copy
    import pytest

    cfg = make_config(tmp_path)
    model_dir = tmp_path / "trusted-model"
    model_dir.mkdir()
    model = {
        "id": "custom:trusted", "name": "trusted", "full_name": "example/trusted",
        "runtime_path": str(model_dir), "dir_path": str(model_dir), "source": "custom_dir",
        "format": "safetensors", "dtype": "BF16", "params_b": 8, "size_gb": 16,
        "quant_method": None, "quant_bits": None,
    }
    manager = ComposeManager(cfg)
    plan = manager.generate(model=model, engine_key="vllm", node_metrics=metrics(), name="trusted-service")
    tampered = copy.deepcopy(plan)
    tampered["port"] = 22
    with pytest.raises(ValueError, match="metadata mismatch: port"):
        manager.validate_generated_plan(tampered, model=model, node_metrics=metrics())


def test_gb10_vllm_profiles_keep_concurrency_small_and_ptxas_is_scoped(tmp_path: Path):
    cfg = make_config(tmp_path)
    model_dir = tmp_path / "model-profile"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    model = {
        "id": "custom:profile", "name": "profile", "full_name": "example/profile",
        "runtime_path": str(model_dir), "dir_path": str(model_dir), "source": "custom_dir",
        "format": "safetensors", "dtype": "BF16", "params_b": 8, "size_gb": 16,
        "quant_method": None, "quant_bits": None,
    }
    manager = ComposeManager(cfg)
    for profile, expected in (("conservative", "1"), ("balanced", "2"), ("performance", "4")):
        plan = manager.generate(model=model, engine_key="vllm", node_metrics=metrics(), profile=profile)
        doc = __import__("yaml").safe_load(plan["yaml"])
        cmd = doc["services"]["inference"]["command"]
        idx = cmd.index("--max-num-seqs")
        assert str(cmd[idx + 1]) == expected
        assert doc["services"]["inference"]["environment"]["TRITON_PTXAS_PATH"] == "/usr/local/cuda/bin/ptxas"

    localai = manager.generate(model=model, engine_key="localai", node_metrics=metrics())
    local_doc = __import__("yaml").safe_load(localai["yaml"])
    assert "TRITON_PTXAS_PATH" not in local_doc["services"]["inference"]["environment"]
