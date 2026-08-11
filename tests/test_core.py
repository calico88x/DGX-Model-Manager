from __future__ import annotations

import json
import os
from pathlib import Path

from dgx_manager.compose_manager import ComposeManager
from dgx_manager.config import Config
from dgx_manager.db import Database
from dgx_manager.system import is_allowed_service_url


def make_config(tmp_path: Path) -> Config:
    p=tmp_path/"config.json"
    cfg=Config(p)
    cfg.data["security"]["require_https"]=False
    cfg.data["security"]["cookie_secure"]=False
    cfg.data["tls"]["enabled"]=False
    cfg.data["paths"]["database"]=str(tmp_path/"db.sqlite")
    cfg.data["paths"]["secret_key"]=str(tmp_path/"secret.key")
    cfg.data["paths"]["compose_root"]=str(tmp_path/"compose")
    cfg.data["paths"]["hf_cache"]=str(tmp_path/"hf"/"hub")
    cfg.data["paths"]["custom_dirs"]=str(tmp_path/"custom_dirs.json")
    cfg.save(); return cfg


def test_password_sessions_and_roles(tmp_path):
    cfg=make_config(tmp_path); db=Database(cfg)
    user=db.create_user("admin","correct horse battery staple","Administrator","admin")
    assert db.verify_password("admin","wrong") is None
    assert db.verify_password("admin","correct horse battery staple")["role"]=="admin"
    token,csrf=db.create_session(user["id"],12)
    resolved=db.get_session_user(token)
    assert resolved and resolved[0]["username"]=="admin" and resolved[1]==csrf
    db.delete_session(token); assert db.get_session_user(token) is None


def test_service_url_policy():
    assert is_allowed_service_url("http://127.0.0.1:8000")[0]
    assert is_allowed_service_url("http://10.0.0.12:8000")[0]
    assert not is_allowed_service_url("http://169.254.169.254/latest/meta-data")[0]
    assert not is_allowed_service_url("https://example.com")[0]
    assert is_allowed_service_url("https://example.com",allow_public=True)[0]


def test_compose_generator_preserves_model_path(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "dgx_manager.compose_manager._port_is_available",
        lambda bind_host, port: True,
    )
    cfg=make_config(tmp_path); cm=ComposeManager(cfg)
    model_dir=tmp_path/"models"/"demo"; model_dir.mkdir(parents=True); (model_dir/"config.json").write_text("{}")
    model={"id":"custom:demo","name":"demo","full_name":"example/demo","runtime_path":str(model_dir),"dir_path":str(model_dir),"source":"custom_dir","format":"safetensors","dtype":"FP4","params_b":35,"size_gb":25.0}
    metrics={"memory_total_gb":128,"unified_memory":True,"gpu":{"name":"NVIDIA GB10"}}
    plan=cm.generate(model=model,engine_key="vllm",node_metrics=metrics,context_length=32768,memory_reserve_gb=24)
    assert "vllm/vllm-openai" in plan["yaml"]
    assert str(model_dir) in plan["yaml"]
    assert "TRITON_PTXAS_PATH" in plan["yaml"]
    assert "--max-num-seqs" in plan["yaml"]
    assert "'2'" in plan["yaml"] or '- 2' in plan["yaml"]
    assert "127.0.0.1:8000:8000" in plan["yaml"]
    saved=cm.save_generated(plan)
    assert Path(saved["path"],"compose.yaml").exists()
    assert cm.profiles_for_engine("vllm")[0]["kind"]=="compose"
