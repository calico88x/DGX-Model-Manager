from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

_tmp=Path(tempfile.mkdtemp(prefix="dmm2-test-"))
_cfg=_tmp/"config.json"
base=json.loads((Path(__file__).parents[1]/"config.example.json").read_text())
base["security"]["require_https"]=False; base["security"]["cookie_secure"]=False; base["tls"]["enabled"]=False
base["paths"]["database"]=str(_tmp/"db.sqlite"); base["paths"]["secret_key"]=str(_tmp/"secret.key"); base["paths"]["compose_root"]=str(_tmp/"compose"); base["paths"]["custom_dirs"]=str(_tmp/"dirs.json")
_cfg.write_text(json.dumps(base))
os.environ["DMM_CONFIG"]=str(_cfg); os.environ["DMM_DEMO_MODE"]="1"

from fastapi.testclient import TestClient
import app as app_module


def test_demo_endpoints():
    with TestClient(app_module.app) as c:
        assert c.get("/api/auth/status").json()["bootstrap_required"] is False
        assert c.get("/api/auth/me").status_code==200
        d=c.get("/api/dashboard").json(); assert d["metrics"]["platform_class"]=="DGX Spark / GB10"
        inv=c.get("/api/inventory").json()["models"]; assert len(inv)>=4
        model=next(x for x in inv if x["source"]=="hf_cache" and x["format"]=="safetensors")
        plan=c.post("/api/compose/generate",json={"model_id":model["id"],"engine":"vllm","context_length":32768,"memory_reserve_gb":24}).json()
        assert "services:" in plan["yaml"] and plan["fit_status"] in {"good","tight","risk"}
        assert c.get("/api/litellm/config").status_code==200
        assert c.get("/api/debug/system").status_code==200
