#!/usr/bin/env python3
"""Optional DGX Model Manager v2 node agent for multi-node deployments."""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import socket

import httpx
from pathlib import Path

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from dgx_manager.compose_manager import ComposeManager
from dgx_manager.config import Config
from dgx_manager.inventory import scan_local
from dgx_manager.system import docker_compose_version, service_check, system_metrics

AGENT_CONFIG = Path(os.path.expanduser(os.environ.get("DMM_AGENT_CONFIG","~/.config/dgx-model-manager-v2/agent.json")))
if not AGENT_CONFIG.exists():
    raise SystemExit(f"Agent config not found: {AGENT_CONFIG}. Run setup-agent.sh first.")
agent_cfg=json.loads(AGENT_CONFIG.read_text())
expected_hash=agent_cfg.get("token_hash","")

config=Config(Path(os.path.expanduser(agent_cfg.get("manager_config","~/.config/dgx-model-manager-v2/config.json"))))
compose=ComposeManager(config)
app=FastAPI(title="DGX Model Manager v2 Node Agent",docs_url=None,redoc_url=None,openapi_url=None)

async def auth(request:Request):
    auth=request.headers.get("authorization","")
    if not auth.startswith("Bearer ") or not expected_hash:
        raise HTTPException(401,"Authentication required")
    got=hashlib.sha256(auth[7:].encode()).hexdigest()
    if not hmac.compare_digest(got,expected_hash): raise HTTPException(401,"Invalid token")

class GenerateReq(BaseModel):
    model_id:str; engine:str; name:str|None=None; context_length:int|None=None; memory_reserve_gb:float|None=None
    profile:str="balanced"; bind_host:str|None=None; expose_litellm:bool=True

class Plan(BaseModel):
    name:str; slug:str; engine:str; yaml:str
    model_id:str|None=None; model_name:str|None=None; model_source:str|None=None; model_path:str|None=None
    context_length:int|None=None; memory_reserve_gb:float|None=None; estimated_runtime_gb:float|None=None; memory_budget_gb:float|None=None
    fit_status:str|None=None; fit_ratio:float|None=None; profile:str|None=None; bind_host:str|None=None; port:int|None=None; expose_litellm:bool=True
    served_model_name:str|None=None; engine_memory_fraction:float|None=None; quant_method:str|None=None; quant_bits:int|None=None
    notes:list[str]=Field(default_factory=list); generated_at:str|None=None

@app.middleware("http")
async def headers(request:Request,call_next):
    max_bytes=int(agent_cfg.get("max_request_bytes",2097152))
    cl=request.headers.get("content-length")
    if cl and cl.isdigit() and int(cl)>max_bytes:
        from fastapi.responses import JSONResponse
        return JSONResponse({"detail":"Request too large"},status_code=413)
    response=await call_next(request)
    response.headers["X-Content-Type-Options"]="nosniff"
    response.headers["X-Frame-Options"]="DENY"
    response.headers["Referrer-Policy"]="no-referrer"
    response.headers["Permissions-Policy"]="camera=(), microphone=(), geolocation=(), payment=()"
    response.headers["Cache-Control"]="no-store"
    if request.url.scheme=="https": response.headers["Strict-Transport-Security"]="max-age=31536000"
    return response

@app.get("/health")
async def health(): return {"ok":True}

@app.get("/v1/info",dependencies=[Depends(auth)])
async def info():
    m=system_metrics(config.path_value("paths.hf_cache"))
    return {"name":agent_cfg.get("name") or socket.gethostname(),"hostname":m["hostname"],"architecture":m["architecture"],"platform_class":m["platform_class"],"compose_version":await docker_compose_version()}

@app.get("/v1/metrics",dependencies=[Depends(auth)])
async def metrics(): return system_metrics(config.path_value("paths.hf_cache"))

@app.get("/v1/inventory",dependencies=[Depends(auth)])
async def inventory(): return {"models":scan_local(config)}


def _service_base(key: str) -> str:
    return str(config.get(f"services.{key}_base", ""))


async def _agent_status(client: httpx.AsyncClient) -> dict:
    async def check_engine(key: str) -> tuple[str, dict]:
        engine = compose.engine(key)
        result = await service_check(client, _service_base(key), engine.get("health_path", "/health"))
        models_path = engine.get("models_path")
        if result.get("ok") and models_path:
            try:
                r = await client.get(_service_base(key).rstrip("/") + models_path, timeout=3)
                data = r.json().get("data", [])
                if data:
                    result["model"] = data[0].get("id")
            except Exception:
                pass
        return key, result

    pairs = await asyncio.gather(
        asyncio.create_task(_named_check(client, "ollama", "/api/tags")),
        asyncio.create_task(_named_check(client, "litellm", "/health")),
        *(asyncio.create_task(check_engine(key)) for key in compose.catalog),
    )
    return dict(pairs)


async def _named_check(client: httpx.AsyncClient, key: str, path: str) -> tuple[str, dict]:
    return key, await service_check(client, _service_base(key), path)


@app.get("/v1/dashboard", dependencies=[Depends(auth)])
async def dashboard():
    metrics = system_metrics(config.path_value("paths.hf_cache"))
    models = scan_local(config)
    ollama_size = 0.0
    ollama_count = 0
    async with httpx.AsyncClient(timeout=6.0, follow_redirects=False) as client:
        status = await _agent_status(client)
        try:
            r = await client.get(_service_base("ollama").rstrip("/") + "/api/tags", timeout=5)
            if r.status_code < 400:
                ollama = r.json().get("models", [])
                ollama_count = len(ollama)
                ollama_size = sum(float(item.get("size") or 0) for item in ollama) / 1e9
        except Exception:
            pass
    return {
        "metrics": metrics,
        "status": status,
        "model_count": len(models) + ollama_count,
        "model_size_gb": round(sum(float(m.get("size_gb") or 0) for m in models) + ollama_size, 1),
        "deployments": compose.list_deployments(),
        "compose_version": await docker_compose_version(),
        "node": {"name": agent_cfg.get("name") or socket.gethostname()},
    }

@app.post("/v1/compose/generate",dependencies=[Depends(auth)])
async def generate(req:GenerateReq):
    models=scan_local(config)
    model=next((m for m in models if m.get("id")==req.model_id),None)
    if not model: raise HTTPException(404,"Model not found on this node")
    try:
        plan=compose.generate(
            model=model,engine_key=req.engine,node_metrics=system_metrics(config.path_value("paths.hf_cache")),
            name=req.name,context_length=req.context_length,memory_reserve_gb=req.memory_reserve_gb,
            profile=req.profile,bind_host=req.bind_host,expose_litellm=req.expose_litellm,
        )
    except ValueError as exc:
        raise HTTPException(400,str(exc))
    plan["node"]=agent_cfg.get("name") or socket.gethostname()
    return plan

@app.get("/v1/compose",dependencies=[Depends(auth)])
async def deployments(): return {"deployments":compose.list_deployments()}

@app.post("/v1/compose",dependencies=[Depends(auth)])
async def save(plan:Plan):
    models=scan_local(config)
    model=next((m for m in models if m.get("id")==plan.model_id),None)
    if not model: raise HTTPException(404,"Model not found on this node")
    try:
        safe=compose.validate_generated_plan(plan.model_dump(),model=model,node_metrics=system_metrics(config.path_value("paths.hf_cache")))
        safe["node"]=agent_cfg.get("name") or socket.gethostname()
        return compose.save_generated(safe,overwrite=False)
    except ValueError as exc:
        raise HTTPException(400,str(exc))

@app.post("/v1/compose/{engine}/{slug}/up",dependencies=[Depends(auth)])
async def up(engine:str,slug:str): return await compose.up(engine,slug)
@app.post("/v1/compose/{engine}/{slug}/down",dependencies=[Depends(auth)])
async def down(engine:str,slug:str): return await compose.down(engine,slug)
@app.get("/v1/compose/{engine}/{slug}/logs",dependencies=[Depends(auth)])
async def logs(engine:str,slug:str,lines:int=200): return await compose.logs(engine,slug,lines)
@app.get("/v1/compose/{engine}/{slug}/status",dependencies=[Depends(auth)])
async def status(engine:str,slug:str): return await compose.status(engine,slug)
@app.delete("/v1/compose/{engine}/{slug}",dependencies=[Depends(auth)])
async def remove(engine:str,slug:str): return await compose.remove(engine,slug,True)

if __name__=="__main__":
    host=agent_cfg.get("host","0.0.0.0"); port=int(agent_cfg.get("port",8092))
    tls=agent_cfg.get("tls",{})
    kwargs={}
    if tls.get("enabled"):
        cert=Path(os.path.expanduser(tls.get("cert_file",""))); key=Path(os.path.expanduser(tls.get("key_file","")))
        if not cert.exists() or not key.exists():
            raise SystemExit("Node-agent TLS is enabled but the certificate/key is missing")
        kwargs={"ssl_certfile":str(cert),"ssl_keyfile":str(key)}
    elif host not in {"127.0.0.1","::1","localhost"} and not agent_cfg.get("allow_insecure_http",False):
        raise SystemExit("Refusing to expose the node agent without TLS. Set allow_insecure_http=true only for an explicitly accepted test environment.")
    uvicorn.run(app,host=host,port=port,**kwargs)
