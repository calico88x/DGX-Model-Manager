from __future__ import annotations

import copy
import hashlib
import hmac
import json
import os
import secrets
from pathlib import Path
from typing import Any

APP_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CONFIG: dict[str, Any] = {
    "app": {
        "host": "0.0.0.0",
        "port": 8091,
        "display_name": "DGX Model Manager v2",
        "session_hours": 12,
        "allow_registration": False,
        "legacy_scripts_enabled": False,
        "demo_mode": False,
    },
    "security": {
        "cookie_secure": True,
        "require_https": True,
        "allowed_hosts": ["*"],
        "allow_public_service_targets": False,
        "max_request_bytes": 2097152,
        "bootstrap_token_hash": "",
    },
    "tls": {
        "enabled": True,
        "cert_file": "~/.config/dgx-model-manager-v2/certs/server.crt",
        "key_file": "~/.config/dgx-model-manager-v2/certs/server.key",
    },
    "services": {
        "ollama_base": "http://127.0.0.1:11434",
        "litellm_base": "http://127.0.0.1:4000",
        "sglang_base": "http://127.0.0.1:30000",
        "vllm_base": "http://127.0.0.1:8000",
        "llamacpp_base": "http://127.0.0.1:8080",
        "localai_base": "http://127.0.0.1:9090",
        "comfyui_base": "http://127.0.0.1:8188",
    },
    "paths": {
        # Existing model/storage locations are intentionally preserved.
        "litellm_config": "~/litellm/litellm_config.yaml",
        "hf_cache": "~/.cache/huggingface/hub",
        "hf_metadata_cache": "~/.local/share/dgx-model-manager-v2/hf-metadata-cache.json",
        "compose_root": "~/.local/share/dgx-model-manager-v2/compose",
        "database": "~/.local/share/dgx-model-manager-v2/model-manager.db",
        "secret_key": "~/.config/dgx-model-manager-v2/secret.key",
        "audit_log": "~/.local/share/dgx-model-manager-v2/audit.jsonl",
        "custom_dirs": "~/.config/dgx-model-manager-v2/custom_dirs.json",
        "bootstrap_token": "~/.config/dgx-model-manager-v2/bootstrap.token",
        "legacy_sglang_scripts": "~/SGLang",
        "legacy_vllm_scripts": "~/vLLM",
        "legacy_llamacpp_scripts": "~/llama.cpp",
        "legacy_localai_scripts": "~/LocalAI",
        "legacy_comfyui_scripts": "~/ComfyUI",
    },
    "compose": {
        "bind_host": "127.0.0.1",
        "default_memory_reserve_gb": 24,
        "default_context_length": 32768,
        "default_profile": "balanced",
        "images": {
            "vllm": "vllm/vllm-openai:cu130-nightly",
            "sglang": "lmsysorg/sglang:v0.5.12.post1",
            "llamacpp": "ghcr.io/ggml-org/llama.cpp:server-cuda13",
            "localai": "localai/localai:latest-nvidia-l4t-arm64-cuda-13",
            "comfyui": "",
        },
    },
    "nodes": {
        "allow_insecure_http": False,
        "request_timeout_seconds": 15,
    },
}


def _deep_merge(base: dict, incoming: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def default_config_path() -> Path:
    env = os.environ.get("DMM_CONFIG")
    if env:
        return Path(os.path.expanduser(env)).resolve()
    return Path(os.path.expanduser("~/.config/dgx-model-manager-v2/config.json"))


class Config:
    def __init__(self, path: Path | None = None):
        self.path = path or default_config_path()
        self.data = copy.deepcopy(DEFAULT_CONFIG)
        self.reload()

    def reload(self) -> None:
        if self.path.exists():
            try:
                incoming = json.loads(self.path.read_text())
                self.data = _deep_merge(DEFAULT_CONFIG, incoming)
            except Exception as exc:
                raise RuntimeError(f"Could not parse config file {self.path}: {exc}") from exc
        else:
            self.data = copy.deepcopy(DEFAULT_CONFIG)

        # Environment-only screenshot/demo mode. Never persisted by default.
        if os.environ.get("DMM_DEMO_MODE") == "1":
            self.data["app"]["demo_mode"] = True
            self.data["security"]["require_https"] = False
            self.data["security"]["cookie_secure"] = False
            self.data["tls"]["enabled"] = False

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2) + "\n")
        os.chmod(tmp, 0o600)
        tmp.replace(self.path)
        os.chmod(self.path, 0o600)

    def get(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self.data
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def set(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        cur = self.data
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = value

    def path_value(self, dotted: str) -> Path:
        return Path(os.path.expanduser(str(self.get(dotted)))).resolve()

    def public_dict(self) -> dict:
        """Return only fields that are safe to expose to authenticated users."""
        return {
            "app": {
                "port": self.get("app.port"),
                "display_name": self.get("app.display_name"),
                "allow_registration": bool(self.get("app.allow_registration")),
                "legacy_scripts_enabled": bool(self.get("app.legacy_scripts_enabled")),
                "demo_mode": bool(self.get("app.demo_mode")),
            },
            "security": {
                "require_https": bool(self.get("security.require_https")),
                "cookie_secure": bool(self.get("security.cookie_secure")),
                "allow_public_service_targets": bool(self.get("security.allow_public_service_targets")),
            },
            "tls": {
                "enabled": bool(self.get("tls.enabled")),
            },
            "services": copy.deepcopy(self.get("services", {})),
            "paths": {
                "litellm_config": str(self.path_value("paths.litellm_config")),
                "hf_cache": str(self.path_value("paths.hf_cache")),
                "compose_root": str(self.path_value("paths.compose_root")),
            },
            "compose": copy.deepcopy(self.get("compose", {})),
        }

    def ensure_directories(self) -> None:
        for key in (
            "paths.compose_root",
            "paths.database",
            "paths.hf_metadata_cache",
            "paths.secret_key",
            "paths.audit_log",
            "paths.custom_dirs",
            "paths.bootstrap_token",
        ):
            p = self.path_value(key)
            target = p if p.suffix == "" and key.endswith("compose_root") else p.parent
            target.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _bootstrap_hash(token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()

    def ensure_bootstrap_token(self, *, rotate: bool = False) -> tuple[str | None, Path, bool]:
        """Ensure a one-time first-admin bootstrap token exists.

        The plaintext token is kept only in a mode-0600 local file so setup can show
        it to the machine operator.  Only its SHA-256 hash is retained in config.
        Successful bootstrap deletes the token file and clears the hash.
        """
        path = self.path_value("paths.bootstrap_token")
        current_hash = str(self.get("security.bootstrap_token_hash", "") or "")
        if current_hash and not rotate:
            if path.exists():
                try:
                    token = path.read_text().strip()
                    if token and hmac.compare_digest(self._bootstrap_hash(token), current_hash):
                        return token, path, False
                except OSError:
                    pass
            return None, path, False

        token = "dmm2_bootstrap_" + secrets.token_urlsafe(32)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(token + "\n")
        os.chmod(path, 0o600)
        self.set("security.bootstrap_token_hash", self._bootstrap_hash(token))
        self.save()
        return token, path, True

    def verify_bootstrap_token(self, token: str) -> bool:
        expected = str(self.get("security.bootstrap_token_hash", "") or "")
        if not expected or not token:
            return False
        return hmac.compare_digest(self._bootstrap_hash(token), expected)

    def clear_bootstrap_token(self) -> None:
        self.set("security.bootstrap_token_hash", "")
        self.save()
        try:
            self.path_value("paths.bootstrap_token").unlink(missing_ok=True)
        except OSError:
            pass
