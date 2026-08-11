from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi import HTTPException
from starlette.requests import Request

import app as appmod
import dgx_manager.compose_manager as compose_mod
from dgx_manager.compose_manager import ComposeManager
from dgx_manager.config import Config
from dgx_manager.system import service_check


class FakeResponse:
    def __init__(self, status_code: int = 200):
        self.status_code = status_code


class FakeClient:
    def __init__(self, status_code: int = 200):
        self.status_code = status_code
        self.calls: list[dict] = []

    async def get(self, url: str, **kwargs):
        self.calls.append(
            {
                "url": url,
                **kwargs,
            }
        )
        return FakeResponse(self.status_code)


def make_config(tmp_path: Path) -> Config:
    path = tmp_path / "config.json"
    cfg = Config(path)
    cfg.data["paths"]["compose_root"] = str(tmp_path / "compose")
    cfg.save()
    return cfg


def test_service_check_forwards_optional_headers():
    client = FakeClient()

    result = asyncio.run(
        service_check(
            client,
            "http://127.0.0.1:4000",
            "/health",
            5,
            headers={"Authorization": "Bearer unit-test-key"},
        )
    )

    assert result["ok"] is True
    assert result["status_code"] == 200

    assert len(client.calls) == 1
    assert client.calls[0]["url"] == "http://127.0.0.1:4000/health"
    assert client.calls[0]["timeout"] == 5
    assert client.calls[0]["headers"] == {
        "Authorization": "Bearer unit-test-key",
    }


def test_systemd_service_credential_and_litellm_header(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    credential_dir = tmp_path / "credentials"
    credential_dir.mkdir()

    credential = credential_dir / "litellm_master_key"
    credential.write_text("unit-test-key\n")

    monkeypatch.setenv(
        "CREDENTIALS_DIRECTORY",
        str(credential_dir),
    )

    assert (
        appmod.service_credential("litellm_master_key")
        == "unit-test-key"
    )

    assert appmod.litellm_auth_headers() == {
        "Authorization": "Bearer unit-test-key",
    }


def test_missing_systemd_service_credential_returns_none(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    credential_dir = tmp_path / "credentials"
    credential_dir.mkdir()

    monkeypatch.setenv(
        "CREDENTIALS_DIRECTORY",
        str(credential_dir),
    )

    assert appmod.service_credential("missing") is None


def test_litellm_headers_empty_without_credential(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv(
        "CREDENTIALS_DIRECTORY",
        raising=False,
    )

    assert appmod.litellm_auth_headers() == {}


def test_litellm_401_is_reachable_but_auth_required(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict = {}

    async def fake_service_check(
        client,
        base,
        path,
        timeout=3.0,
        headers=None,
    ):
        captured["base"] = base
        captured["path"] = path
        captured["headers"] = headers

        return {
            "ok": False,
            "status_code": 401,
            "latency_ms": 3,
        }

    monkeypatch.setattr(
        appmod,
        "service_check",
        fake_service_check,
    )

    monkeypatch.setattr(
        appmod,
        "litellm_auth_headers",
        lambda: {
            "Authorization": "Bearer unit-test-key",
        },
    )

    monkeypatch.setattr(
        appmod.app.state,
        "http",
        object(),
        raising=False,
    )

    result = asyncio.run(appmod.litellm_status())

    assert result["ok"] is True
    assert result["auth_required"] is True
    assert result["status_code"] == 401

    assert captured["path"] == "/health"
    assert captured["headers"] == {
        "Authorization": "Bearer unit-test-key",
    }


def test_litellm_403_is_reachable_but_auth_required(
    monkeypatch: pytest.MonkeyPatch,
):
    async def fake_service_check(
        client,
        base,
        path,
        timeout=3.0,
        headers=None,
    ):
        return {
            "ok": False,
            "status_code": 403,
            "latency_ms": 4,
        }

    monkeypatch.setattr(
        appmod,
        "service_check",
        fake_service_check,
    )

    monkeypatch.setattr(
        appmod.app.state,
        "http",
        object(),
        raising=False,
    )

    result = asyncio.run(appmod.litellm_status())

    assert result["ok"] is True
    assert result["auth_required"] is True
    assert result["status_code"] == 403


def test_redaction_hides_secrets_but_not_max_tokens():
    source = {
        "api_key": "hidden",
        "master_key": "hidden",
        "password": "hidden",
        "access_token": "hidden",
        "authorization": "hidden",
        "max_tokens": 4096,
        "tokenizer": "qwen",
        "nested": {
            "database_password": "hidden",
            "client_secret": "hidden",
            "max_tokens": 8192,
        },
    }

    result = appmod.redact(source)

    assert result["api_key"] == "***REDACTED***"
    assert result["master_key"] == "***REDACTED***"
    assert result["password"] == "***REDACTED***"
    assert result["access_token"] == "***REDACTED***"
    assert result["authorization"] == "***REDACTED***"

    assert result["nested"]["database_password"] == "***REDACTED***"
    assert result["nested"]["client_secret"] == "***REDACTED***"

    assert result["max_tokens"] == 4096
    assert result["nested"]["max_tokens"] == 8192
    assert result["tokenizer"] == "qwen"


def test_port_allocator_keeps_preferred_port_when_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cfg = make_config(tmp_path)
    manager = ComposeManager(cfg)

    monkeypatch.setattr(
        manager,
        "list_deployments",
        lambda: [],
    )

    monkeypatch.setattr(
        compose_mod,
        "_port_is_available",
        lambda bind_host, port: True,
    )

    port, changed = manager._allocate_host_port(
        8000,
        "127.0.0.1",
    )

    assert port == 8000
    assert changed is False


def test_port_allocator_skips_reserved_and_unavailable_ports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cfg = make_config(tmp_path)
    manager = ComposeManager(cfg)

    monkeypatch.setattr(
        manager,
        "list_deployments",
        lambda: [
            {
                "engine": "vllm",
                "slug": "existing",
                "port": 8000,
            }
        ],
    )

    def available(bind_host: str, port: int) -> bool:
        # 8000 is reserved by saved deployment metadata.
        # 8001 represents another live host workload.
        # 8002 is the first available port.
        return port == 8002

    monkeypatch.setattr(
        compose_mod,
        "_port_is_available",
        available,
    )

    port, changed = manager._allocate_host_port(
        8000,
        "127.0.0.1",
    )

    assert port == 8002
    assert changed is True


def test_external_engine_can_be_online_without_being_managed(
    monkeypatch: pytest.MonkeyPatch,
):
    async def fake_service_check(
        client,
        base,
        path,
        timeout=3.0,
        headers=None,
    ):
        return {
            "ok": True,
            "status_code": 200,
            "latency_ms": 2,
        }

    monkeypatch.setattr(
        appmod,
        "service_check",
        fake_service_check,
    )

    monkeypatch.setattr(
        appmod,
        "service_base",
        lambda key: "http://127.0.0.1:8000",
    )

    monkeypatch.setattr(
        appmod,
        "engine_health_path",
        lambda key: "/health",
    )

    monkeypatch.setattr(
        appmod.compose,
        "engine",
        lambda key: {},
    )

    monkeypatch.setattr(
        appmod.compose,
        "profiles_for_engine",
        lambda key: [],
    )

    monkeypatch.setattr(
        appmod.app.state,
        "http",
        object(),
        raising=False,
    )

    result = asyncio.run(
        appmod.engine_status("vllm")
    )

    assert result["running"] is True
    assert result["managed_running"] is False
    assert result["base"] == "http://127.0.0.1:8000"


def test_unmanaged_engine_stop_is_refused(
    monkeypatch: pytest.MonkeyPatch,
):
    original_get = appmod.config.get

    def fake_config_get(key, default=None):
        if key == "app.demo_mode":
            return False
        return original_get(key, default)

    monkeypatch.setattr(
        appmod.config,
        "get",
        fake_config_get,
    )

    monkeypatch.setattr(
        appmod.compose,
        "profiles_for_engine",
        lambda engine: [],
    )

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/vllm/stop",
            "headers": [],
            "query_string": b"",
            "scheme": "https",
            "server": ("test", 8091),
            "client": ("127.0.0.1", 12345),
        }
    )

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            appmod.generic_engine_stop(
                "vllm",
                request,
                user={
                    "id": 1,
                    "username": "test-admin",
                    "role": "admin",
                },
            )
        )

    assert exc.value.status_code == 409
    assert "not managed" in str(exc.value.detail)
    assert "Refusing to stop" in str(exc.value.detail)