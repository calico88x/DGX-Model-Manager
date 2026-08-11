from __future__ import annotations

import json
import os
import re
import shutil
import struct
import time
from pathlib import Path
from typing import Any

import httpx
from huggingface_hub import HfApi

from .config import Config

PIPELINE_TO_TASK = {
    "text-generation": "Text Gen", "text2text-generation": "Text Gen", "image-text-to-text": "Vision LLM",
    "visual-question-answering": "Vision LLM", "feature-extraction": "Embedding", "sentence-similarity": "Embedding",
    "automatic-speech-recognition": "STT", "text-to-speech": "TTS", "text-to-image": "Image Gen",
    "image-to-image": "Image Gen", "image-classification": "Vision", "audio-classification": "Audio",
}

DTYPE_ALIASES = {
    "float32":"FP32", "float":"FP32", "float16":"FP16", "half":"FP16", "bfloat16":"BF16",
    "float8_e4m3fn":"FP8", "float8":"FP8", "fp8":"FP8", "fp4":"FP4", "nvfp4":"FP4",
    "int8":"INT8", "int4":"INT4",
}


def _size_tree(path: Path) -> int:
    """Return logical stored bytes without double-counting HF cache symlinks.

    Hugging Face snapshots commonly reference files in blobs/ via symlinks.
    Path.stat() follows those links, so track the underlying device/inode and
    count each physical file only once.
    """
    total = 0
    seen: set[tuple[int, int]] = set()

    try:
        for root, _, files in os.walk(path):
            for name in files:
                try:
                    st = (Path(root) / name).stat()
                    key = (st.st_dev, st.st_ino)

                    if key in seen:
                        continue

                    seen.add(key)
                    total += st.st_size

                except OSError:
                    pass
    except OSError:
        pass

    return total


def _read_json(path: Path) -> dict:
    try: return json.loads(path.read_text())
    except Exception: return {}


def _quantization_meta(cfg: dict, name: str = "") -> dict:
    """Return normalized checkpoint quantization metadata.

    Preserve the declared quantizer separately from the storage precision.
    Mixed per-group quantization is represented explicitly rather than
    collapsed into a misleading single bit width.
    """
    raw = cfg.get("quantization_config")
    qc = raw if isinstance(raw, dict) else {}

    declared = str(
        qc.get("quant_method")
        or qc.get("quantization_method")
        or qc.get("method")
        or ""
    ).strip().lower()

    qformat = str(qc.get("format") or "").strip().lower() or None

    if declared in {"compressed_tensors", "compressed-tensors"}:
        method = "compressed-tensors"
    elif declared == "modelopt":
        if qformat and "fp4" in qformat:
            method = "modelopt_fp4"
        elif qformat and "fp8" in qformat:
            method = "modelopt_fp8"
        else:
            method = "modelopt"
    elif declared:
        method = declared
    else:
        method = None

    haystack = " ".join((
        declared,
        qformat or "",
        name.lower(),
        json.dumps(qc, sort_keys=True, default=str).lower(),
    ))

    if method is None:
        if "compressed-tensors" in haystack or "compressed_tensors" in haystack:
            method = "compressed-tensors"
        elif "modelopt_fp4" in haystack or ("modelopt" in haystack and "fp4" in haystack):
            method = "modelopt_fp4"
        elif "modelopt_fp8" in haystack or ("modelopt" in haystack and "fp8" in haystack):
            method = "modelopt_fp8"
        elif "gptq_marlin" in haystack:
            method = "gptq_marlin"
        elif "awq_marlin" in haystack:
            method = "awq_marlin"
        elif "gptq" in haystack:
            method = "gptq"
        elif "awq" in haystack:
            method = "awq"
        elif "bitsandbytes" in haystack:
            method = "bitsandbytes"

    discovered_bits: set[int] = set()
    discovered_types: set[str] = set()

    # Top-level bit width, when present.
    top_bits = qc.get("bits") or qc.get("weight_bits") or qc.get("num_bits")
    try:
        if top_bits is not None:
            discovered_bits.add(int(top_bits))
    except (TypeError, ValueError):
        pass

    # ModelOpt/compressed-tensors may define different precision groups.
    groups = qc.get("config_groups")
    if isinstance(groups, dict):
        for group in groups.values():
            if not isinstance(group, dict):
                continue

            weights = group.get("weights")
            if not isinstance(weights, dict):
                continue

            candidate = (
                weights.get("num_bits")
                or weights.get("bits")
                or weights.get("weight_bits")
            )
            try:
                if candidate is not None:
                    discovered_bits.add(int(candidate))
            except (TypeError, ValueError):
                pass

            qtype = str(weights.get("type") or "").strip().lower()
            if qtype:
                discovered_types.add(qtype)

    # Some formats encode precision in the format name instead.
    fmt_text = (qformat or "").lower()
    if not discovered_bits:
        if "nvfp4" in fmt_text or "fp4" in fmt_text:
            discovered_bits.add(4)
        elif "fp8" in fmt_text:
            discovered_bits.add(8)

    bits_all = sorted(discovered_bits)
    mixed = len(bits_all) > 1
    bits = bits_all[0] if len(bits_all) == 1 else None

    return {
        "quant_method": method,
        "quant_format": qformat,
        "quant_bits": bits,
        "quant_bits_all": bits_all,
        "quantization_mixed": mixed,
        "quant_types": sorted(discovered_types),
        "quantization_declared": bool(qc or method or qformat),
    }


def _base_dtype(cfg: dict) -> str:
    """Logical/default model dtype from the model configuration."""
    for value in (cfg.get("torch_dtype"), cfg.get("dtype")):
        if value is None:
            continue
        raw = str(value).lower().replace("torch.", "").strip()
        if raw in DTYPE_ALIASES:
            return DTYPE_ALIASES[raw]
        if raw == "bf16":
            return "BF16"
        if raw == "fp16":
            return "FP16"
        if raw == "fp32":
            return "FP32"
    return "Unknown"


def _checkpoint_dtype(cfg: dict, name: str = "") -> str:
    """Effective checkpoint storage/quantization precision."""
    q = _quantization_meta(cfg, name)
    bits_all = q.get("quant_bits_all") or []
    qtypes = set(q.get("quant_types") or [])

    # Mixed ModelOpt checkpoints must not be collapsed to whichever precision
    # happens to appear in the repository name.
    if q.get("quantization_mixed"):
        if bits_all == [4, 8] and (not qtypes or qtypes == {"float"}):
            return "FP4/FP8"
        return "Mixed"

    text = " ".join((
        str(q.get("quant_method") or ""),
        str(q.get("quant_format") or ""),
        name.lower(),
    )).lower()

    if "nvfp4" in text or "modelopt_fp4" in text or "mxfp4" in text or "fp4" in text:
        return "FP4"
    if "modelopt_fp8" in text or "fp8" in text or "float8" in text:
        return "FP8"

    if q.get("quant_bits") == 4:
        return "FP4" if "float" in qtypes else "INT4"
    if q.get("quant_bits") == 8:
        return "FP8" if "float" in qtypes else "INT8"

    if any(x in text for x in ("gptq", "awq")):
        return "INT4"

    return _base_dtype(cfg)


def _infer_dtype(cfg: dict, name: str = "") -> str:
    # Backward-compatible display field.
    return _checkpoint_dtype(cfg, name)



def _local_safetensors_dtype(path: Path) -> str | None:
    """Determine dominant checkpoint dtype from local safetensors headers only."""
    counts: dict[str, int] = {}

    for file in sorted(path.glob("*.safetensors")):
        try:
            with file.open("rb") as fh:
                raw = fh.read(8)
                if len(raw) != 8:
                    continue

                header_len = struct.unpack("<Q", raw)[0]
                if header_len <= 0 or header_len > 256 * 1024 * 1024:
                    continue

                header = json.loads(fh.read(header_len))

            if not isinstance(header, dict):
                continue

            for name, meta in header.items():
                if name == "__metadata__" or not isinstance(meta, dict):
                    continue

                dtype = str(meta.get("dtype") or "").upper()
                shape = meta.get("shape")

                if not dtype or not isinstance(shape, list):
                    continue

                count = 1
                valid = True

                for dim in shape:
                    try:
                        dim = int(dim)
                    except (TypeError, ValueError):
                        valid = False
                        break

                    if dim < 0:
                        valid = False
                        break

                    count *= dim

                if valid:
                    counts[dtype] = counts.get(dtype, 0) + count

        except Exception:
            continue

    aliases = {
        "F64": "FP64",
        "F32": "FP32",
        "F16": "FP16",
        "BF16": "BF16",
        "F8_E4M3": "FP8",
        "F8_E4M3FN": "FP8",
        "F8_E5M2": "FP8",
        "I8": "INT8",
        "U8": "INT8",
    }

    candidates = [
        (count, aliases[dtype])
        for dtype, count in counts.items()
        if dtype in aliases and count > 1000
    ]

    if not candidates:
        return None

    candidates.sort(reverse=True)
    return candidates[0][1]


def _safetensors_parameter_count(path: Path) -> int | None:
    """Count logical tensor elements from local safetensors headers only.

    No tensor data is loaded.  Quantized checkpoints are handled elsewhere,
    because packed quantized tensor shapes are not necessarily the model's
    logical parameter count.
    """
    files = sorted(path.glob("*.safetensors"))
    if not files:
        return None

    total = 0
    found = False

    for file in files:
        try:
            with file.open("rb") as fh:
                raw = fh.read(8)
                if len(raw) != 8:
                    continue

                header_len = struct.unpack("<Q", raw)[0]
                if header_len <= 0 or header_len > 256 * 1024 * 1024:
                    continue

                header = json.loads(fh.read(header_len))

            if not isinstance(header, dict):
                continue

            for name, meta in header.items():
                if name == "__metadata__" or not isinstance(meta, dict):
                    continue

                shape = meta.get("shape")
                if not isinstance(shape, list):
                    continue

                count = 1
                valid = True
                for dim in shape:
                    try:
                        dim = int(dim)
                    except (TypeError, ValueError):
                        valid = False
                        break

                    if dim < 0:
                        valid = False
                        break

                    count *= dim

                if valid:
                    total += count
                    found = True

        except Exception:
            continue

    return total if found else None


def _params_info(cfg: dict, model_path: Path | None = None, name: str = "") -> tuple[float | None, bool]:
    # Explicit model metadata always wins.
    for key in ("num_parameters", "parameter_count", "n_params"):
        v = cfg.get(key)
        if isinstance(v, (int, float)) and v > 1e6:
            return round(v / 1e9, 3), False

    # Packed/quantized checkpoints must not be counted from stored tensor shapes.
    quant = _quantization_meta(cfg, name)
    if quant.get("quantization_declared"):
        return None, False

    # For ordinary safetensors checkpoints, tensor shapes provide an exact local
    # parameter count without loading weights or requiring network access.
    if model_path is not None:
        count = _safetensors_parameter_count(model_path)
        if count is not None and count > 1_000_000:
            return round(count / 1e9, 3), False

    # Do not fabricate parameter counts from architecture dimensions.
    return None, False


def _params_b(cfg: dict) -> float | None:
    # Backward-compatible helper used by older callers/tests.
    return _params_info(cfg)[0]


def _modalities(cfg: dict) -> list[str]:
    arch = " ".join(cfg.get("architectures") or []).lower()
    model_type = str(cfg.get("model_type", "")).lower()

    # Wrapper-style multimodal models frequently keep the language and vision/audio
    # architecture in nested config objects rather than encoding it in model_type.
    nested_parts = []
    for key in ("text_config", "vision_config", "audio_config"):
        nested = cfg.get(key)
        if isinstance(nested, dict):
            nested_parts.append(str(nested.get("model_type", "")))
            nested_parts.extend(nested.get("architectures") or [])

    text = " ".join([arch, model_type, *nested_parts]).lower()

    mods = ["Text"]

    if (
        isinstance(cfg.get("vision_config"), dict)
        or any(x in text for x in (
            "vision", "vl", "image", "llava",
            "qwen2_5_vl", "qwen3_vl", "gemma3",
        ))
    ):
        mods.append("Vision")

    if (
        isinstance(cfg.get("audio_config"), dict)
        or any(x in text for x in (
            "audio", "whisper", "speech", "hubert",
        ))
    ):
        mods.append("Audio")

    return list(dict.fromkeys(mods))


def _task(mods: list[str], name: str = "", model_path: Path | None = None) -> str:
    n = name.lower()

    # Sentence Transformers identify themselves structurally, even when the
    # underlying Transformer architecture is a generic BERT/XLM-R model.
    if model_path is not None:
        if any((model_path / marker).exists() for marker in (
            "modules.json",
            "sentence_bert_config.json",
            "config_sentence_transformers.json",
        )):
            return "Embedding"

    # Name fallbacks are intentionally narrow and are used when repository
    # metadata is absent or too generic to identify the actual workload.
    if "embed" in n or "sentence-transform" in n:
        return "Embedding"
    if "whisper" in n:
        return "STT"
    if any(x in n for x in ("tts", "kokoro", "orpheus")):
        return "TTS"
    if any(x in n for x in ("hubert", "snac")):
        return "Audio"

    if "Vision" in mods:
        return "Vision LLM"
    if "Audio" in mods:
        return "Audio"
    return "Text Gen"


def _has_model_payload(path: Path) -> bool:
    """Return True only when a cache snapshot contains actual model weights.

    Hugging Face may create cache entries containing only config/tokenizer
    metadata. Those are useful cache artifacts but are not installed models
    and should not appear in Model Inventory.
    """
    weight_suffixes = (
        ".safetensors",
        ".gguf",
        ".bin",
        ".pt",
        ".pth",
    )

    try:
        for item in path.rglob("*"):
            if item.is_file() and item.name.lower().endswith(weight_suffixes):
                return True
    except OSError:
        pass

    return False


def _format(path: Path) -> str:
    try:
        names = [p.name.lower() for p in path.rglob("*") if p.is_file()]
    except Exception: return "unknown"
    if any(n.endswith(".gguf") for n in names): return "gguf"
    if any(n.endswith(".safetensors") for n in names): return "safetensors"
    if any(n.endswith((".bin",".pt",".pth")) for n in names): return "pytorch"
    return "unknown"


def _latest_snapshot(model_dir: Path) -> Path:
    snap = model_dir / "snapshots"
    if snap.is_dir():
        dirs = [p for p in snap.iterdir() if p.is_dir()]
        if dirs: return max(dirs, key=lambda p: p.stat().st_mtime)
    return model_dir


def parse_hf_model(model_dir: Path) -> dict:
    raw = model_dir.name[len("models--"):]
    parts = raw.split("--", 1)
    owner, name = (parts[0], parts[1]) if len(parts) == 2 else ("", raw)
    snap = _latest_snapshot(model_dir)
    cfg = _read_json(snap / "config.json")
    mods = _modalities(cfg); params_b, params_estimated = _params_info(cfg, snap, name); quant = _quantization_meta(cfg, name)
    base_dtype = _base_dtype(cfg)
    checkpoint_dtype = _checkpoint_dtype(cfg, name)

    if not quant.get("quantization_declared") and checkpoint_dtype == "Unknown":
        local_dtype = _local_safetensors_dtype(snap)
        if local_dtype:
            checkpoint_dtype = local_dtype
            if base_dtype == "Unknown":
                base_dtype = local_dtype

    return {
        "id": f"hf:{owner}/{name}", "name": name, "owner": owner, "full_name": f"{owner}/{name}" if owner else name,
        "dir_path": str(model_dir), "runtime_path": str(snap),
        "dtype": checkpoint_dtype,
        "base_dtype": base_dtype,
        "checkpoint_dtype": checkpoint_dtype,
        "params_b": params_b,
        "params_estimated": params_estimated, "size_gb": round(_size_tree(model_dir)/1e9,4),
        "modalities": mods, "source": "hf_cache", "format": _format(snap), "pipeline_tag": None,
        "task_label": _task(mods, name, snap), "hf_downloads": None, "hf_likes": None, "model_arch": cfg.get("model_type") or "Unknown",
        **quant,
    }


def parse_flat_model(path: Path) -> dict:
    cfg = _read_json(path / "config.json")
    mods = _modalities(cfg); fmt = _format(path); params_b, params_estimated = _params_info(cfg, path, path.name); quant = _quantization_meta(cfg, path.name)
    if fmt == "gguf" and not quant.get("quant_method"):
        quant = {"quant_method":"gguf", "quant_bits":quant.get("quant_bits"), "quantization_declared":True}

    base_dtype = _base_dtype(cfg)
    checkpoint_dtype = "GGUF" if fmt == "gguf" else _checkpoint_dtype(cfg, path.name)

    if fmt != "gguf" and not quant.get("quantization_declared") and checkpoint_dtype == "Unknown":
        local_dtype = _local_safetensors_dtype(path)
        if local_dtype:
            checkpoint_dtype = local_dtype
            if base_dtype == "Unknown":
                base_dtype = local_dtype

    return {
        "id": f"custom:{path}", "name": path.name, "owner": "", "full_name": path.name, "dir_path": str(path), "runtime_path": str(path),
        "dtype": checkpoint_dtype,
        "base_dtype": base_dtype,
        "checkpoint_dtype": checkpoint_dtype,
        "params_b": params_b, "params_estimated": params_estimated,
        "size_gb": round(_size_tree(path)/1e9,4), "modalities": mods, "source": "custom_dir", "format": fmt,
        "pipeline_tag": None, "task_label": _task(mods,path.name), "hf_downloads": None, "hf_likes": None,
        "model_arch": cfg.get("model_type") or "Unknown", **quant,
    }


def load_custom_dirs(config: Config) -> list[str]:
    p = config.path_value("paths.custom_dirs")
    try: return json.loads(p.read_text()) if p.exists() else []
    except Exception: return []


def save_custom_dirs(config: Config, dirs: list[str]) -> None:
    p = config.path_value("paths.custom_dirs"); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dirs, indent=2)+"\n"); os.chmod(p,0o600)


def validate_custom_dir(path: str) -> Path:
    if "\0" in path or "\n" in path: raise ValueError("Invalid path")
    p = Path(os.path.expanduser(path)).resolve()
    if not p.is_dir(): raise ValueError("Directory does not exist")
    blocked = {Path("/"),Path("/etc"),Path("/usr"),Path("/bin"),Path("/sbin"),Path("/var"),Path("/boot"),Path("/dev"),Path("/proc"),Path("/sys"),Path("/root")}
    if p in blocked: raise ValueError("System root directories cannot be scanned")
    return p


def scan_local(config: Config) -> list[dict]:
    models: list[dict] = []
    hf = config.path_value("paths.hf_cache")
    if hf.is_dir():
        for d in sorted(hf.iterdir()):
            if d.is_dir() and d.name.startswith("models--"):
                try:
                    snap = _latest_snapshot(d)
                    if not _has_model_payload(snap):
                        continue
                    models.append(parse_hf_model(d))
                except Exception:
                    pass
    for raw in load_custom_dirs(config):
        try: base = Path(os.path.expanduser(raw)).resolve()
        except Exception: continue
        if not base.is_dir() or base == hf: continue
        # A custom dir may itself be one model or a parent containing models.
        candidates = [base] if (base/"config.json").exists() or list(base.glob("*.gguf")) else [d for d in base.iterdir() if d.is_dir()]
        for d in sorted(candidates):
            if d.name.startswith("models--"):
                try:
                    m=parse_hf_model(d); m["source"]="custom_dir"; models.append(m)
                except Exception: pass
            elif (d/"config.json").exists() or list(d.glob("*.gguf")):
                try: models.append(parse_flat_model(d))
                except Exception: pass
    return models


async def ollama_models(client: httpx.AsyncClient, base: str) -> list[dict]:
    try:
        r = await client.get(base.rstrip("/")+"/api/tags", timeout=5); r.raise_for_status()
    except Exception: return []
    out=[]
    for m in r.json().get("models",[]):
        details=m.get("details",{}); name=m.get("name",""); ps=details.get("parameter_size","")
        try: pb=float(str(ps).upper().replace("B","").strip())
        except Exception: pb=None
        out.append({
            "id":"ollama:"+name,"name":name.split(":")[0],"owner":"","full_name":name,"dir_path":"","runtime_path":"",
            "dtype":str(details.get("quantization_level") or "Unknown").upper(),"params_b":pb,"params_estimated":False,
            "size_gb":round((m.get("size") or 0)/1e9,1),"modalities":["Text"],"source":"ollama","format":"ollama",
            "pipeline_tag":None,"task_label":"Text Gen","hf_downloads":None,"hf_likes":None,"model_arch":details.get("family") or "Ollama",
            "quant_method":str(details.get("quantization_level") or "").lower() or None,"quant_bits":None,"quantization_declared":bool(details.get("quantization_level")),
        })
    return out


HF_METADATA_TTL_SECONDS = 7 * 24 * 60 * 60


def _load_hf_metadata_cache(config: Config) -> dict:
    path = config.path_value("paths.hf_metadata_cache")
    try:
        if not path.exists():
            return {}
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_hf_metadata_cache(config: Config, cache: dict) -> None:
    path = config.path_value("paths.hf_metadata_cache")
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")
    os.chmod(tmp, 0o600)
    tmp.replace(path)
    os.chmod(path, 0o600)


def _normalize_base_model(value: Any) -> str | None:
    if isinstance(value, str):
        value = value.strip()
        return value or None

    if isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, str) and item.strip():
                return item.strip()

    return None


def _dtype_from_safetensors_info(info: Any) -> str | None:
    safe = getattr(info, "safetensors", None)
    parameters = getattr(safe, "parameters", None)

    if not isinstance(parameters, dict) or not parameters:
        return None

    # Ignore tiny integer metadata tensors when determining checkpoint precision.
    weighted = []
    for dtype, count in parameters.items():
        try:
            count = int(count)
        except (TypeError, ValueError):
            continue

        raw = str(dtype).lower().replace("torch.", "").strip()

        safetensors_aliases = {
            "f64": "FP64",
            "f32": "FP32",
            "f16": "FP16",
            "bf16": "BF16",
            "f8_e4m3": "FP8",
            "f8_e4m3fn": "FP8",
            "f8_e5m2": "FP8",
            "i8": "INT8",
            "u8": "INT8",
        }

        mapped = safetensors_aliases.get(raw) or DTYPE_ALIASES.get(raw)

        if mapped and count > 1000:
            weighted.append((count, mapped))

    if not weighted:
        return None

    weighted.sort(reverse=True)
    return weighted[0][1]


def _fetch_hf_metadata(api: HfApi, model: dict) -> dict:
    repo = str(model.get("full_name") or "").strip()
    if not repo or "/" not in repo:
        return {}

    # Normal model_info retains card_data/base_model.  Expanded responses may
    # omit card metadata, so fetch both views and merge only the fields we need.
    normal = api.model_info(repo)

    try:
        expanded = api.model_info(
            repo,
            expand=["safetensors", "pipeline_tag"],
        )
    except Exception:
        expanded = normal

    pipeline_tag = (
        getattr(normal, "pipeline_tag", None)
        or getattr(expanded, "pipeline_tag", None)
    )

    card_data = getattr(normal, "card_data", None)
    base_model = _normalize_base_model(
        getattr(card_data, "base_model", None) if card_data else None
    )

    total = None
    params_source = None

    if model.get("quantization_declared"):
        # Never treat packed quantized checkpoint tensor counts as the logical
        # model parameter count.  Resolve the declared base model instead.
        if base_model:
            try:
                base_info = api.model_info(
                    base_model,
                    expand=["safetensors"],
                )
                base_safe = getattr(base_info, "safetensors", None)
                base_total = getattr(base_safe, "total", None)

                if base_total is not None:
                    total = int(base_total)
                    params_source = "base_model_safetensors"
            except Exception:
                pass
    else:
        safe = (
            getattr(expanded, "safetensors", None)
            or getattr(normal, "safetensors", None)
        )
        repo_total = getattr(safe, "total", None) if safe else None

        try:
            if repo_total is not None:
                total = int(repo_total)
                params_source = "repo_safetensors"
        except (TypeError, ValueError):
            total = None
            params_source = None

    return {
        "fetched_at": int(time.time()),
        "pipeline_tag": pipeline_tag,
        "base_model": base_model,
        "params_total": total,
        "params_source": params_source,
        "checkpoint_dtype_remote": _dtype_from_safetensors_info(expanded),
        "downloads": getattr(normal, "downloads", None),
        "likes": getattr(normal, "likes", None),
    }


def _apply_hf_metadata(model: dict, meta: dict) -> None:
    pipeline_tag = meta.get("pipeline_tag")

    if pipeline_tag:
        model["pipeline_tag"] = pipeline_tag

        modalities = set(model.get("modalities") or [])
        local_task = model.get("task_label") or "Unknown"

        # Local architecture is stronger evidence than a broad/mistagged HF
        # pipeline tag.  In particular, multimodal wrappers are frequently
        # published under text-generation and audio encoders under
        # feature-extraction.
        if "Vision" in modalities:
            model["task_label"] = "Vision LLM"

        elif "Audio" in modalities:
            if pipeline_tag == "automatic-speech-recognition":
                model["task_label"] = "STT"
            elif pipeline_tag == "text-to-speech":
                model["task_label"] = "TTS"
            elif local_task in {"STT", "TTS"}:
                model["task_label"] = local_task
            else:
                model["task_label"] = "Audio"

        elif pipeline_tag == "feature-extraction":
            # Sentence Transformer structure is strong evidence of an
            # embedding model; otherwise feature-extraction is ambiguous.
            if local_task == "Embedding":
                model["task_label"] = "Embedding"

        else:
            model["task_label"] = PIPELINE_TO_TASK.get(
                pipeline_tag,
                local_task,
            )

    base_model = meta.get("base_model")
    if base_model:
        model["base_model"] = base_model

    total = meta.get("params_total")
    if model.get("params_b") is None and isinstance(total, int) and total > 1_000_000:
        model["params_b"] = round(total / 1e9, 3)
        model["params_estimated"] = False
        model["params_source"] = meta.get("params_source") or "hf_metadata"

    remote_dtype = meta.get("checkpoint_dtype_remote")
    if (
        remote_dtype
        and model.get("checkpoint_dtype") in {None, "", "Unknown"}
        and not model.get("quantization_declared")
    ):
        model["checkpoint_dtype"] = remote_dtype
        model["dtype"] = remote_dtype

        if model.get("base_dtype") in {None, "", "Unknown"}:
            model["base_dtype"] = remote_dtype

    if meta.get("downloads") is not None:
        model["hf_downloads"] = meta["downloads"]

    if meta.get("likes") is not None:
        model["hf_likes"] = meta["likes"]


def enrich_hf_metadata(
    config: Config,
    models: list[dict],
    *,
    ttl_seconds: int = HF_METADATA_TTL_SECONDS,
) -> list[dict]:
    """Enrich locally discovered HF models using a private persistent cache.

    Inventory remains usable offline.  Cached metadata is applied first; stale
    entries are refreshed when Hugging Face is reachable.  Network failures do
    not discard existing local or cached metadata.
    """
    cache = _load_hf_metadata_cache(config)
    api = HfApi()
    now = int(time.time())
    dirty = False

    for model in models:
        if model.get("source") not in {"hf_cache", "custom_dir"}:
            continue

        repo = str(model.get("full_name") or "").strip()
        if not repo or "/" not in repo:
            continue

        entry = cache.get(repo)
        if not isinstance(entry, dict):
            entry = None

        if entry:
            _apply_hf_metadata(model, entry)

        fetched_at = int((entry or {}).get("fetched_at") or 0)
        stale = not entry or now - fetched_at >= ttl_seconds

        if not stale:
            continue

        try:
            refreshed = _fetch_hf_metadata(api, model)
            if refreshed:
                cache[repo] = refreshed
                _apply_hf_metadata(model, refreshed)
                dirty = True
        except Exception:
            # Offline/private/unavailable repositories must not break Inventory.
            continue

    if dirty:
        _save_hf_metadata_cache(config, cache)

    return models


async def hf_search(client: httpx.AsyncClient, q: str, sort: str="downloads", limit:int=20, pipeline_tag:str|None=None) -> list[dict]:
    params={"search":q,"sort":sort,"limit":min(max(limit,1),50),"full":"true"}
    if pipeline_tag: params["filter"]=pipeline_tag
    r=await client.get("https://huggingface.co/api/models",params=params,timeout=15); r.raise_for_status()
    out=[]
    for m in r.json():
        tags=m.get("tags") or []; pt=m.get("pipeline_tag") or ""
        out.append({"id":m.get("modelId") or m.get("id", ""),"pipeline_tag":pt,"task_label":PIPELINE_TO_TASK.get(pt,pt or "Unknown"),
                    "downloads":m.get("downloads",0),"likes":m.get("likes",0),"tags":tags[:18],"library_name":m.get("library_name"),
                    "has_safetensors":"safetensors" in tags})
    return out


async def hf_variants(client: httpx.AsyncClient, model_id: str) -> list[dict]:
    base=model_id.split("/",1)[-1]
    variants=[]
    for tag in ("gguf","gptq","awq"):
        try:
            r=await client.get("https://huggingface.co/api/models",params={"search":base,"filter":tag,"sort":"downloads","limit":"6"},timeout=15)
            if r.status_code==200:
                for m in r.json():
                    mid=m.get("modelId") or m.get("id","")
                    if mid and mid!=model_id: variants.append({"id":mid,"type":tag.upper(),"downloads":m.get("downloads",0)})
        except Exception: pass
    seen=set(); out=[]
    for v in variants:
        if v["id"] not in seen: seen.add(v["id"]); out.append(v)
    return out[:20]


def hf_files(owner: str, name: str) -> list[dict]:
    """Return repository files with authoritative byte sizes from Hugging Face."""
    info = HfApi().model_info(
        f"{owner}/{name}",
        files_metadata=True,
    )

    return [
        {
            "name": sibling.rfilename,
            "size": sibling.size,
        }
        for sibling in (info.siblings or [])
        if sibling.rfilename and not sibling.rfilename.startswith(".")
    ]


def safe_delete_model(config: Config, path: str) -> str:
    target=Path(os.path.expanduser(path)).resolve()
    roots=[config.path_value("paths.hf_cache")]+[Path(os.path.expanduser(d)).resolve() for d in load_custom_dirs(config)]
    if not any(_is_relative_to(target,r) and target != r for r in roots): raise ValueError("Path is not under a configured model directory")
    if not target.is_dir(): raise ValueError("Model directory not found")
    shutil.rmtree(target); return str(target)


def _is_relative_to(p:Path, root:Path)->bool:
    try: p.relative_to(root); return True
    except ValueError: return False
