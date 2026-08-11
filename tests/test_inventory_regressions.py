from __future__ import annotations

import json
import os
import struct
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import dgx_manager.inventory as inventory
from dgx_manager.config import Config


def make_config(tmp_path: Path) -> Config:
    path = tmp_path / "config.json"

    cfg = Config(path)
    cfg.data["paths"]["hf_cache"] = str(tmp_path / "hf" / "hub")
    cfg.data["paths"]["hf_metadata_cache"] = str(
        tmp_path / "state" / "hf-metadata-cache.json"
    )
    cfg.data["paths"]["custom_dirs"] = str(
        tmp_path / "state" / "custom_dirs.json"
    )
    cfg.save()

    return cfg


def write_safetensors_header(
    path: Path,
    tensors: dict[str, dict],
) -> None:
    """Create the minimum file needed for local header inspection tests."""
    header = {}

    for name, meta in tensors.items():
        header[name] = {
            "dtype": meta["dtype"],
            "shape": meta["shape"],
            "data_offsets": [0, 0],
        }

    encoded = json.dumps(header).encode("utf-8")

    with path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(encoded)))
        fh.write(encoded)


def test_compressed_tensors_nvfp4_metadata():
    cfg = {
        "torch_dtype": "bfloat16",
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "format": "nvfp4-pack-quantized",
            "bits": 4,
        },
    }

    quant = inventory._quantization_meta(
        cfg,
        "Qwen3-Coder-Next-NVFP4",
    )

    assert quant["quant_method"] == "compressed-tensors"
    assert quant["quant_format"] == "nvfp4-pack-quantized"
    assert quant["quant_bits"] == 4
    assert quant["quant_bits_all"] == [4]
    assert quant["quantization_mixed"] is False
    assert quant["quantization_declared"] is True

    assert inventory._base_dtype(cfg) == "BF16"
    assert (
        inventory._checkpoint_dtype(
            cfg,
            "Qwen3-Coder-Next-NVFP4",
        )
        == "FP4"
    )


def test_mixed_modelopt_fp4_fp8_is_not_collapsed():
    cfg = {
        "torch_dtype": "bfloat16",
        "quantization_config": {
            "quant_method": "modelopt",
            "config_groups": {
                "fp4_group": {
                    "weights": {
                        "num_bits": 4,
                        "type": "float",
                    }
                },
                "fp8_group": {
                    "weights": {
                        "num_bits": 8,
                        "type": "float",
                    }
                },
            },
        },
    }

    quant = inventory._quantization_meta(
        cfg,
        "Qwen3.6-35B-A3B-NVFP4",
    )

    assert quant["quant_method"] == "modelopt"
    assert quant["quant_bits"] is None
    assert quant["quant_bits_all"] == [4, 8]
    assert quant["quantization_mixed"] is True
    assert quant["quant_types"] == ["float"]

    assert inventory._base_dtype(cfg) == "BF16"
    assert (
        inventory._checkpoint_dtype(
            cfg,
            "Qwen3.6-35B-A3B-NVFP4",
        )
        == "FP4/FP8"
    )


def test_modelopt_format_preserves_base_and_checkpoint_dtype():
    cfg = {
        "torch_dtype": "bfloat16",
        "quantization_config": {
            "quant_method": "modelopt",
            "format": "nvfp4",
            "bits": 4,
        },
    }

    assert inventory._base_dtype(cfg) == "BF16"
    assert inventory._checkpoint_dtype(cfg, "demo") == "FP4"


def test_local_safetensors_exact_parameter_count_and_dtype(
    tmp_path: Path,
):
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    write_safetensors_header(
        model_dir / "model.safetensors",
        {
            "model.weight": {
                "dtype": "F32",
                "shape": [23_000_000],
            }
        },
    )

    assert (
        inventory._safetensors_parameter_count(model_dir)
        == 23_000_000
    )

    assert inventory._local_safetensors_dtype(model_dir) == "FP32"

    params_b, estimated = inventory._params_info(
        {},
        model_dir,
        "demo",
    )

    assert params_b == 0.023
    assert estimated is False


def test_quantized_checkpoint_does_not_count_packed_tensor_shapes(
    tmp_path: Path,
):
    model_dir = tmp_path / "quantized"
    model_dir.mkdir()

    write_safetensors_header(
        model_dir / "model.safetensors",
        {
            "packed.weight": {
                "dtype": "U8",
                "shape": [80_000_000],
            }
        },
    )

    cfg = {
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "format": "nvfp4-pack-quantized",
            "bits": 4,
        }
    }

    params_b, estimated = inventory._params_info(
        cfg,
        model_dir,
        "quantized-demo",
    )

    assert params_b is None
    assert estimated is False


def test_hf_quantized_model_uses_base_model_parameter_count():
    normal = SimpleNamespace(
        pipeline_tag="text-generation",
        card_data=SimpleNamespace(
            base_model="Qwen/Base-Model",
        ),
        downloads=1234,
        likes=56,
        safetensors=None,
    )

    expanded = SimpleNamespace(
        pipeline_tag="text-generation",
        safetensors=SimpleNamespace(
            # This represents packed checkpoint storage and must not be
            # treated as the logical parameter count.
            total=9_000_000_000,
            parameters={
                "U8": 9_000_000_000,
            },
        ),
    )

    base_info = SimpleNamespace(
        safetensors=SimpleNamespace(
            total=35_952_000_000,
        )
    )

    class FakeApi:
        def model_info(self, repo: str, **kwargs):
            if repo == "example/quantized-model":
                if kwargs.get("expand"):
                    return expanded
                return normal

            if repo == "Qwen/Base-Model":
                return base_info

            raise AssertionError(f"Unexpected repo: {repo}")

    model = {
        "full_name": "example/quantized-model",
        "quantization_declared": True,
    }

    meta = inventory._fetch_hf_metadata(
        FakeApi(),
        model,
    )

    assert meta["base_model"] == "Qwen/Base-Model"
    assert meta["params_total"] == 35_952_000_000
    assert meta["params_source"] == "base_model_safetensors"
    assert meta["downloads"] == 1234
    assert meta["likes"] == 56


def test_apply_hf_metadata_preserves_local_vision_classification():
    model = {
        "modalities": ["Text", "Vision"],
        "task_label": "Vision LLM",
        "params_b": None,
        "params_estimated": False,
        "base_dtype": "BF16",
        "checkpoint_dtype": "FP4",
        "dtype": "FP4",
        "quantization_declared": True,
    }

    inventory._apply_hf_metadata(
        model,
        {
            "pipeline_tag": "text-generation",
            "base_model": "example/base",
            "params_total": 35_952_000_000,
            "params_source": "base_model_safetensors",
        },
    )

    assert model["task_label"] == "Vision LLM"
    assert model["pipeline_tag"] == "text-generation"
    assert model["base_model"] == "example/base"
    assert model["params_b"] == 35.952


def test_apply_hf_metadata_preserves_local_audio_task():
    model = {
        "modalities": ["Audio"],
        "task_label": "TTS",
        "params_b": 3.301,
        "params_estimated": False,
        "base_dtype": "BF16",
        "checkpoint_dtype": "BF16",
        "dtype": "BF16",
        "quantization_declared": False,
    }

    inventory._apply_hf_metadata(
        model,
        {
            "pipeline_tag": "feature-extraction",
        },
    )

    assert model["task_label"] == "TTS"


def test_fresh_hf_metadata_cache_is_used_without_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cfg = make_config(tmp_path)
    cache_path = cfg.path_value("paths.hf_metadata_cache")
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cache_path.write_text(
        json.dumps(
            {
                "example/model": {
                    "fetched_at": int(time.time()),
                    "pipeline_tag": "feature-extraction",
                    "base_model": "example/base",
                    "params_total": 23_000_000,
                    "params_source": "repo_safetensors",
                    "checkpoint_dtype_remote": "FP32",
                    "downloads": 100,
                    "likes": 5,
                }
            }
        )
    )

    class FakeApi:
        def model_info(self, *args, **kwargs):
            raise AssertionError(
                "Fresh cached metadata should not trigger network access"
            )

    monkeypatch.setattr(
        inventory,
        "HfApi",
        FakeApi,
    )

    models = [
        {
            "source": "hf_cache",
            "full_name": "example/model",
            "modalities": ["Text"],
            "task_label": "Embedding",
            "params_b": None,
            "params_estimated": False,
            "base_dtype": "Unknown",
            "checkpoint_dtype": "Unknown",
            "dtype": "Unknown",
            "quantization_declared": False,
            "hf_downloads": None,
            "hf_likes": None,
        }
    ]

    result = inventory.enrich_hf_metadata(
        cfg,
        models,
    )

    model = result[0]

    assert model["task_label"] == "Embedding"
    assert model["base_model"] == "example/base"
    assert model["params_b"] == 0.023
    assert model["checkpoint_dtype"] == "FP32"
    assert model["dtype"] == "FP32"
    assert model["base_dtype"] == "FP32"
    assert model["hf_downloads"] == 100
    assert model["hf_likes"] == 5


def test_stale_hf_cache_survives_offline_refresh_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cfg = make_config(tmp_path)
    cache_path = cfg.path_value("paths.hf_metadata_cache")
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cache_path.write_text(
        json.dumps(
            {
                "example/offline-model": {
                    "fetched_at": 1,
                    "pipeline_tag": "text-generation",
                    "base_model": "example/base",
                    "params_total": 7_000_000_000,
                    "params_source": "repo_safetensors",
                    "checkpoint_dtype_remote": "BF16",
                    "downloads": 10,
                    "likes": 2,
                }
            }
        )
    )

    class OfflineApi:
        def model_info(self, *args, **kwargs):
            raise OSError("offline")

    monkeypatch.setattr(
        inventory,
        "HfApi",
        OfflineApi,
    )

    models = [
        {
            "source": "hf_cache",
            "full_name": "example/offline-model",
            "modalities": ["Text"],
            "task_label": "Text Gen",
            "params_b": None,
            "params_estimated": False,
            "base_dtype": "Unknown",
            "checkpoint_dtype": "Unknown",
            "dtype": "Unknown",
            "quantization_declared": False,
            "hf_downloads": None,
            "hf_likes": None,
        }
    ]

    result = inventory.enrich_hf_metadata(
        cfg,
        models,
        ttl_seconds=1,
    )

    model = result[0]

    assert model["base_model"] == "example/base"
    assert model["params_b"] == 7.0
    assert model["checkpoint_dtype"] == "BF16"
    assert model["hf_downloads"] == 10
    assert model["hf_likes"] == 2


def test_hf_metadata_cache_is_written_private(
    tmp_path: Path,
):
    cfg = make_config(tmp_path)

    inventory._save_hf_metadata_cache(
        cfg,
        {
            "example/model": {
                "fetched_at": 123,
            }
        },
    )

    path = cfg.path_value("paths.hf_metadata_cache")

    assert path.exists()
    assert path.stat().st_mode & 0o777 == 0o600


def test_scan_local_skips_metadata_only_hf_cache(
    tmp_path: Path,
):
    cfg = make_config(tmp_path)
    hf = cfg.path_value("paths.hf_cache")
    hf.mkdir(parents=True)

    metadata_only = (
        hf
        / "models--example--metadata-only"
        / "snapshots"
        / "revision-a"
    )
    metadata_only.mkdir(parents=True)
    (metadata_only / "config.json").write_text(
        json.dumps(
            {
                "model_type": "bert",
                "torch_dtype": "float32",
            }
        )
    )

    real_model = (
        hf
        / "models--example--real-model"
        / "snapshots"
        / "revision-b"
    )
    real_model.mkdir(parents=True)
    (real_model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "bert",
                "torch_dtype": "float32",
            }
        )
    )

    write_safetensors_header(
        real_model / "model.safetensors",
        {
            "model.weight": {
                "dtype": "F32",
                "shape": [2_000_000],
            }
        },
    )

    models = inventory.scan_local(cfg)

    names = {
        model["full_name"]
        for model in models
    }

    assert "example/real-model" in names
    assert "example/metadata-only" not in names


def test_size_tree_deduplicates_shared_inode(
    tmp_path: Path,
):
    root = tmp_path / "hf-model"
    root.mkdir()

    blob = root / "blob.bin"

    size = 1_234_567

    with blob.open("wb") as fh:
        fh.truncate(size)

    link_one = root / "weight-1.bin"
    link_two = root / "weight-2.bin"

    link_one.symlink_to(blob)
    link_two.symlink_to(blob)

    assert inventory._size_tree(root) == size


def test_parse_hf_model_preserves_sub_gigabyte_precision(
    tmp_path: Path,
):
    model_dir = tmp_path / "models--example--tiny-model"
    snapshot = model_dir / "snapshots" / "revision"
    snapshot.mkdir(parents=True)

    (snapshot / "config.json").write_text(
        json.dumps(
            {
                "model_type": "bert",
                "torch_dtype": "float32",
            }
        )
    )

    payload = snapshot / "model.safetensors"

    with payload.open("wb") as fh:
        fh.truncate(4_700_000)

    model = inventory.parse_hf_model(model_dir)

    assert model["size_gb"] > 0
    assert model["size_gb"] < 0.01
    assert model["size_gb"] == round(
        inventory._size_tree(model_dir) / 1e9,
        4,
    )


def test_hf_files_uses_authoritative_file_metadata(
    monkeypatch: pytest.MonkeyPatch,
):
    siblings = [
        SimpleNamespace(
            rfilename="config.json",
            size=1234,
        ),
        SimpleNamespace(
            rfilename="model-00001-of-00002.safetensors",
            size=4_000_000_000,
        ),
        SimpleNamespace(
            rfilename=".gitattributes",
            size=987,
        ),
    ]

    class FakeApi:
        def model_info(
            self,
            repo: str,
            *,
            files_metadata: bool = False,
        ):
            assert repo == "example/model"
            assert files_metadata is True

            return SimpleNamespace(
                siblings=siblings,
            )

    monkeypatch.setattr(
        inventory,
        "HfApi",
        FakeApi,
    )

    result = inventory.hf_files(
        "example",
        "model",
    )

    assert result == [
        {
            "name": "config.json",
            "size": 1234,
        },
        {
            "name": "model-00001-of-00002.safetensors",
            "size": 4_000_000_000,
        },
    ]