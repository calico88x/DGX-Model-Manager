#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator, Optional

import httpx
import uvicorn
import yaml
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pydantic import BaseModel, Field

from dgx_manager import __version__
from dgx_manager.auth import COOKIE_NAME, check_login_rate, clear_login_rate, current_user, enforce_csrf, require_role, role_and_csrf
from dgx_manager.compose_manager import ComposeManager
from dgx_manager.config import APP_ROOT, Config
from dgx_manager.db import Database
from dgx_manager.inventory import (
    enrich_hf_metadata, hf_files, hf_search, hf_variants, load_custom_dirs,
    ollama_models, safe_delete_model, save_custom_dirs, scan_local,
    validate_custom_dir,
)
from dgx_manager.legacy import scan as scan_legacy, launch as launch_legacy
from dgx_manager.nodes import NodeClient
from dgx_manager.system import (
    docker_available, docker_compose_version, docker_containers, is_allowed_service_url,
    local_ip, run_cmd, service_check, system_metrics,
)

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
class RingHandler(logging.Handler):
    def __init__(self, maxlen:int=1000):
        super().__init__(); self.buffer:deque[dict]=deque(maxlen=maxlen)
    def emit(self, record:logging.LogRecord):
        self.buffer.append({"ts":datetime.fromtimestamp(record.created,tz=timezone.utc).isoformat(),"level":record.levelname,"logger":record.name,"msg":self.format(record)})
    def entries(self,level:str|None=None,search:str|None=None,limit:int=250)->list[dict]:
        ranks={"DEBUG":10,"INFO":20,"WARNING":30,"ERROR":40,"CRITICAL":50}; minr=ranks.get(level or "",0)
        rows=list(self.buffer)
        if minr: rows=[x for x in rows if ranks.get(x["level"],0)>=minr]
        if search:
            s=search.lower(); rows=[x for x in rows if s in x["msg"].lower() or s in x["logger"].lower()]
        return rows[-max(1,min(limit,1000)):]

ring=RingHandler(); ring.setFormatter(logging.Formatter("%(message)s"))
log=logging.getLogger("dmm2"); log.setLevel(logging.INFO); log.addHandler(ring)
START_MONO=time.monotonic(); START_UTC=datetime.now(timezone.utc).isoformat()

config=Config(); config.ensure_directories(); db=Database(config); compose=ComposeManager(config)
if not config.get("app.demo_mode") and db.user_count() == 0:
    _bootstrap_token, _bootstrap_path, _bootstrap_new = config.ensure_bootstrap_token()
    if _bootstrap_token:
        log.warning("First-run administrator bootstrap token is stored at %s", _bootstrap_path)
    else:
        log.warning("First-run bootstrap token hash exists but plaintext token file is unavailable. Rotate it locally with scripts/bootstrap_token.py --rotate.")
node_client=NodeClient(db,float(config.get("nodes.request_timeout_seconds",15)))

ENGINE_KEYS=("sglang","vllm","llamacpp","localai","comfyui")
ENGINE_NAMES={k:compose.engine(k)["name"] for k in ENGINE_KEYS}

TEMPLATES=Environment(loader=FileSystemLoader(str(APP_ROOT/"templates")),autoescape=select_autoescape(["html","xml"]))

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def client_ip(request:Request)->str:
    return request.client.host if request.client else ""

def audit(request:Request,user:dict|None,action:str,target:str="",detail=None):
    try: db.audit(action,target,detail,user,client_ip(request))
    except Exception: pass


def service_base(key:str)->str:
    if key=="ollama": return str(config.get("services.ollama_base"))
    if key=="litellm": return str(config.get("services.litellm_base"))
    return str(config.get(f"services.{key}_base"))


def service_credential(name:str)->str|None:
    """Read a systemd service credential without exposing it through app config."""
    cred_dir=os.environ.get("CREDENTIALS_DIRECTORY")
    if not cred_dir:
        return None
    try:
        value=(Path(cred_dir)/name).read_text().strip()
        return value or None
    except (OSError, UnicodeError):
        return None


def litellm_auth_headers()->dict[str,str]:
    key=service_credential("litellm_master_key")
    return {"Authorization":f"Bearer {key}"} if key else {}


def engine_health_path(key:str)->str:
    return str(compose.engine(key).get("health_path") or "/health")


def _sensitive_key(key:str)->bool:
    k=str(key).strip().lower().replace("-", "_")
    exact={
        "api_key","apikey",
        "master_key",
        "password","passwd",
        "secret",
        "token",
        "authorization",
        "credential","credentials",
    }
    if k in exact:
        return True
    return k.endswith((
        "_api_key",
        "_password",
        "_secret",
        "_token",
        "_credential",
        "_credentials",
    ))


def redact(obj):
    if isinstance(obj,dict):
        return {
            k:("***REDACTED***" if _sensitive_key(k) else redact(v))
            for k,v in obj.items()
        }
    if isinstance(obj,list):
        return [redact(x) for x in obj]
    return obj


def load_litellm()->dict:
    p=config.path_value("paths.litellm_config")
    if not p.exists(): return {}
    try: return yaml.safe_load(p.read_text()) or {}
    except Exception as exc: raise HTTPException(500,f"Could not parse LiteLLM configuration: {exc}")


def save_litellm(cfg:dict)->None:
    p=config.path_value("paths.litellm_config"); p.parent.mkdir(parents=True,exist_ok=True)
    # Keep a rollback copy before every manager-authored change. The backup is local runtime
    # state and is not part of the repository.
    if p.exists():
        backup=p.with_suffix(p.suffix+".dmm2.bak")
        shutil.copy2(p,backup)
    temp=p.with_suffix(p.suffix+".dmm2.tmp")
    temp.write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True))
    temp.replace(p)


async def restart_litellm()->dict:
    for cmd in (("sudo","-n","systemctl","restart","litellm"),("systemctl","--user","restart","litellm")):
        r=await run_cmd(*cmd,timeout=30)
        if r.returncode==0: return {"ok":True,"output":(r.stdout+r.stderr).strip()}
    return {"ok":False,"output":"LiteLLM restart failed. Configure the documented restricted sudoers rule or a user service."}


def demo_inventory()->list[dict]:
    return [
        {"id":"hf:example/Mistral-Small-4-119B-NVFP4","name":"Mistral-Small-4-119B-NVFP4","owner":"example","full_name":"example/Mistral-Small-4-119B-NVFP4","dir_path":"/demo/hf/mistral","runtime_path":"/demo/hf/mistral/snapshots/main","dtype":"FP4","params_b":119.0,"params_estimated":False,"size_gb":70.8,"modalities":["Text"],"source":"hf_cache","format":"safetensors","task_label":"Text Gen","model_arch":"mistral","hf_downloads":None,"hf_likes":None,"quant_method":"modelopt_fp4","quant_bits":4,"quantization_declared":True},
        {"id":"hf:example/Qwen3.6-35B-A3B-NVFP4","name":"Qwen3.6-35B-A3B-NVFP4","owner":"example","full_name":"example/Qwen3.6-35B-A3B-NVFP4","dir_path":"/demo/hf/qwen","runtime_path":"/demo/hf/qwen/snapshots/main","dtype":"FP4","params_b":35.0,"params_estimated":False,"size_gb":25.1,"modalities":["Text","Vision"],"source":"hf_cache","format":"safetensors","task_label":"Vision LLM","model_arch":"qwen","hf_downloads":None,"hf_likes":None,"quant_method":"modelopt_fp4","quant_bits":4,"quantization_declared":True},
        {"id":"hf:sentence-transformers/all-MiniLM-L6-v2","name":"all-MiniLM-L6-v2","owner":"sentence-transformers","full_name":"sentence-transformers/all-MiniLM-L6-v2","dir_path":"/demo/hf/minilm","runtime_path":"/demo/hf/minilm/snapshots/main","dtype":"FP32","params_b":0.02,"params_estimated":False,"size_gb":0.9,"modalities":["Text"],"source":"hf_cache","format":"safetensors","task_label":"Embedding","model_arch":"bert","hf_downloads":None,"hf_likes":None},
        {"id":"ollama:glm-4.7-flash","name":"glm-4.7-flash","owner":"","full_name":"glm-4.7-flash","dir_path":"","runtime_path":"","dtype":"Q4_K_M","params_b":29.9,"params_estimated":False,"size_gb":19.0,"modalities":["Text"],"source":"ollama","format":"ollama","task_label":"Text Gen","model_arch":"Ollama","hf_downloads":None,"hf_likes":None},
    ]


def demo_metrics()->dict:
    return {"hostname":"spark-a","ip":"192.0.2.21","architecture":"aarch64","platform":"Linux demo","cpu_percent":18.0,"cpu_count":20,"memory_total_gb":128.0,"memory_used_gb":67.4,"memory_available_gb":60.6,"memory_percent":52.7,"disk_total_gb":1000.0,"disk_used_gb":446.2,"disk_free_gb":553.8,"disk_percent":44.6,"gpu":{"available":True,"name":"NVIDIA GB10","utilization_pct":37.0,"temperature_c":51.0,"power_w":72.0,"memory_used_mb":None,"memory_total_mb":None},"unified_memory":True,"platform_class":"DGX Spark / GB10"}


async def inventory_all()->list[dict]:
    if config.get("app.demo_mode"): return demo_inventory()
    local=await asyncio.to_thread(scan_local,config)
    local=await asyncio.to_thread(enrich_hf_metadata,config,local)
    oll=await ollama_models(app.state.http,service_base("ollama"))
    return local+oll


async def engine_status(key:str)->dict:
    base=service_base(key); check=await service_check(app.state.http,base,engine_health_path(key))
    model=None
    if check["ok"] and compose.engine(key).get("models_path"):
        try:
            r=await app.state.http.get(base.rstrip("/")+compose.engine(key)["models_path"],timeout=3)
            d=r.json().get("data",[]); model=d[0].get("id") if d else None
        except Exception: pass

    managed_running=False
    for profile in compose.profiles_for_engine(key):
        try:
            st=await compose.status(key,profile["slug"])
            if st.get("running"):
                managed_running=True
                break
        except Exception:
            pass

    return {
        "running":bool(check["ok"]),
        "managed_running":managed_running,
        "model":model,
        "latency_ms":check.get("latency_ms"),
        "base":base,
    }


async def litellm_status()->dict:
    check=await service_check(
        app.state.http,
        service_base("litellm"),
        "/health",
        headers=litellm_auth_headers(),
    )
    # Preserve graceful reachability detection on installations where
    # no LiteLLM credential has been configured yet.
    if check.get("status_code") in {401,403}:
        check["ok"]=True
        check["auth_required"]=True
    return check


async def all_status()->dict:
    if config.get("app.demo_mode"):
        return {"ollama":{"ok":True,"latency_ms":4},"litellm":{"ok":True,"latency_ms":7},"sglang":{"ok":False},"vllm":{"ok":True,"latency_ms":5,"model":"example/Qwen3.6-35B-A3B-NVFP4"},"llamacpp":{"ok":False},"localai":{"ok":False},"comfyui":{"ok":False}}
    tasks={"ollama":service_check(app.state.http,service_base("ollama"),"/api/tags"),"litellm":litellm_status()}
    for k in ENGINE_KEYS: tasks[k]=service_check(app.state.http,service_base(k),engine_health_path(k))
    vals=await asyncio.gather(*tasks.values()); return dict(zip(tasks.keys(),vals))


async def find_model(model_id:str)->dict:
    for m in await inventory_all():
        if m.get("id")==model_id: return m
    raise HTTPException(404,"Model not found")


# -----------------------------------------------------------------------------
# Lifespan / app
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(application:FastAPI):
    application.state.http=httpx.AsyncClient(timeout=10.0,follow_redirects=False,headers={"User-Agent":f"DGX-Model-Manager/{__version__}"})
    log.info("DGX Model Manager v2 started")
    yield
    await application.state.http.aclose(); log.info("DGX Model Manager v2 stopped")

app=FastAPI(title="DGX Model Manager v2",version=__version__,lifespan=lifespan,docs_url=None,redoc_url=None,openapi_url=None)
app.state.config=config; app.state.db=db; app.state.compose=compose
app.mount("/static",StaticFiles(directory=str(APP_ROOT/"static")),name="static")

@app.middleware("http")
async def security_middleware(request:Request,call_next):
    max_bytes=int(config.get("security.max_request_bytes",2097152))
    cl=request.headers.get("content-length")
    if cl and cl.isdigit() and int(cl)>max_bytes:
        return JSONResponse({"detail":"Request too large"},status_code=413)
    # Enforce transport for LAN traffic once TLS is configured. Loopback remains usable for local health checks.
    if config.get("security.require_https") and not config.get("app.demo_mode") and request.url.scheme!="https" and request.client and request.client.host not in {"127.0.0.1","::1"}:
        return JSONResponse({"detail":"HTTPS is required"},status_code=426)
    response=await call_next(request)
    response.headers["X-Content-Type-Options"]="nosniff"
    response.headers["X-Frame-Options"]="DENY"
    response.headers["Referrer-Policy"]="no-referrer"
    response.headers["Permissions-Policy"]="camera=(), microphone=(), geolocation=(), payment=()"
    response.headers["Content-Security-Policy"]="default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
    if request.url.scheme=="https": response.headers["Strict-Transport-Security"]="max-age=31536000"
    if request.url.path.startswith("/api/"): response.headers["Cache-Control"]="no-store"
    return response

# -----------------------------------------------------------------------------
# Page routes
# -----------------------------------------------------------------------------
@app.get("/",response_class=HTMLResponse)
async def index():
    tpl=TEMPLATES.get_template("index.html"); return tpl.render(version=__version__)

@app.get("/help",response_class=HTMLResponse)
async def help_page():
    return FileResponse(APP_ROOT/"docs.html")

@app.get("/favicon.svg")
async def favicon(): return FileResponse(APP_ROOT/"static"/"favicon.svg",media_type="image/svg+xml")

# -----------------------------------------------------------------------------
# Authentication / accounts
# -----------------------------------------------------------------------------
class LoginReq(BaseModel): username:str=Field(min_length=1,max_length=64); password:str=Field(min_length=1,max_length=256)
class BootstrapReq(BaseModel): username:str; display_name:str="Administrator"; password:str; bootstrap_token:str=Field(min_length=1,max_length=256)
class RegisterReq(BaseModel): username:str; display_name:str=""; password:str
class UserCreateReq(BaseModel): username:str; display_name:str=""; password:str; role:str="viewer"
class UserUpdateReq(BaseModel): role:Optional[str]=None; is_active:Optional[bool]=None; password:Optional[str]=None; display_name:Optional[str]=None
class ApiTokenReq(BaseModel): name:str; role:str="viewer"

@app.get("/api/auth/status")
async def auth_status(): return {"bootstrap_required":False if config.get("app.demo_mode") else db.user_count()==0,"bootstrap_token_required":bool(not config.get("app.demo_mode") and db.user_count()==0),"registration_enabled":bool(config.get("app.allow_registration")),"demo_mode":bool(config.get("app.demo_mode")),"version":__version__}

@app.post("/api/auth/bootstrap")
async def bootstrap(req:BootstrapReq,request:Request):
    if db.user_count()!=0: raise HTTPException(409,"Bootstrap has already been completed")
    check_login_rate(f"bootstrap-ip:{client_ip(request)}", limit=8, window_seconds=600)
    if not config.verify_bootstrap_token(req.bootstrap_token):
        audit(request,None,"auth.bootstrap_failed","user",{"username":req.username})
        raise HTTPException(403,"Invalid bootstrap token")
    try: user=db.create_first_admin(req.username,req.password,req.display_name)
    except ValueError as exc:
        code=409 if "already" in str(exc).lower() else 400
        raise HTTPException(code,str(exc))
    config.clear_bootstrap_token(); clear_login_rate(f"bootstrap-ip:{client_ip(request)}")
    token,csrf=db.create_session(user["id"],int(config.get("app.session_hours",12))); audit(request,user,"auth.bootstrap","user",{"username":user["username"]})
    resp=JSONResponse({"ok":True,"user":user,"csrf_token":csrf})
    resp.set_cookie(COOKIE_NAME,token,httponly=True,secure=bool(config.get("security.cookie_secure")),samesite="strict",max_age=int(config.get("app.session_hours",12))*3600,path="/")
    return resp

@app.post("/api/auth/login")
async def login(req:LoginReq,request:Request):
    ip=client_ip(request)
    account_key=f"login-account:{ip}:{req.username.lower()}"
    ip_key=f"login-ip:{ip}"
    check_login_rate(ip_key, limit=30, window_seconds=300)
    check_login_rate(account_key, limit=8, window_seconds=300)
    user=db.verify_password(req.username,req.password)
    if not user:
        audit(request,None,"auth.login_failed","user",{"username":req.username}); raise HTTPException(401,"Invalid username or password")
    clear_login_rate(account_key); clear_login_rate(ip_key); token,csrf=db.create_session(user["id"],int(config.get("app.session_hours",12))); audit(request,user,"auth.login","user")
    resp=JSONResponse({"ok":True,"user":user,"csrf_token":csrf})
    resp.set_cookie(COOKIE_NAME,token,httponly=True,secure=bool(config.get("security.cookie_secure")),samesite="strict",max_age=int(config.get("app.session_hours",12))*3600,path="/")
    return resp

@app.post("/api/auth/register")
async def register(req:RegisterReq,request:Request):
    if not config.get("app.allow_registration"): raise HTTPException(403,"Self-registration is disabled")
    check_login_rate(f"register-ip:{client_ip(request)}", limit=5, window_seconds=600)
    try: user=db.create_user(req.username,req.password,req.display_name,"viewer")
    except ValueError as exc: raise HTTPException(400,str(exc))
    audit(request,user,"auth.register","user"); return {"ok":True}

@app.get("/api/auth/me")
async def me(user:dict=Depends(current_user)):
    return {"user":{k:v for k,v in user.items() if k not in {"csrf_token","auth_kind"}},"csrf_token":user.get("csrf_token","demo"),"auth_kind":user.get("auth_kind")}

@app.post("/api/auth/logout")
async def logout(request:Request,user:dict=Depends(enforce_csrf)):
    db.delete_session(request.cookies.get(COOKIE_NAME,"")); audit(request,user,"auth.logout")
    resp=JSONResponse({"ok":True}); resp.delete_cookie(COOKIE_NAME,path="/"); return resp

@app.get("/api/users")
async def users(user:dict=Depends(require_role("admin"))): return {"users":db.list_users()}
@app.post("/api/users")
async def create_user(req:UserCreateReq,request:Request,user:dict=Depends(role_and_csrf("admin"))):
    try: created=db.create_user(req.username,req.password,req.display_name,req.role)
    except ValueError as exc: raise HTTPException(400,str(exc))
    audit(request,user,"user.create",created["username"],{"role":created["role"]}); return created
@app.patch("/api/users/{user_id}")
async def update_user(user_id:int,req:UserUpdateReq,request:Request,user:dict=Depends(role_and_csrf("admin"))):
    target=db.get_user(user_id)
    if not target: raise HTTPException(404,"User not found")
    if user_id==user.get("id") and req.is_active is False: raise HTTPException(400,"You cannot disable your own account")
    removing_active_admin = bool(target.get("is_active") and target.get("role")=="admin" and (req.is_active is False or (req.role is not None and req.role != "admin")))
    if removing_active_admin and db.active_admin_count() <= 1:
        raise HTTPException(400,"At least one active administrator account must remain")
    try: updated=db.update_user(user_id,role=req.role,is_active=req.is_active,password=req.password,display_name=req.display_name)
    except ValueError as exc: raise HTTPException(400,str(exc))
    audit(request,user,"user.update",updated["username"],req.model_dump(exclude_none=True)); return updated

@app.get("/api/tokens")
async def list_tokens(user:dict=Depends(require_role("admin"))): return {"tokens":db.list_api_tokens()}
@app.post("/api/tokens")
async def create_token(req:ApiTokenReq,request:Request,user:dict=Depends(role_and_csrf("admin"))):
    try: meta,token=db.create_api_token(user["id"],req.name,req.role)
    except ValueError as exc: raise HTTPException(400,str(exc))
    audit(request,user,"token.create",req.name,{"role":req.role}); return {"token":token,"metadata":meta,"warning":"This token is shown once."}
@app.delete("/api/tokens/{token_id}")
async def delete_token(token_id:int,request:Request,user:dict=Depends(role_and_csrf("admin"))):
    db.delete_api_token(token_id); audit(request,user,"token.delete",str(token_id)); return {"ok":True}

# -----------------------------------------------------------------------------
# Dashboard / service health
# -----------------------------------------------------------------------------
@app.get("/api/status")
async def status(user:dict=Depends(current_user)): return await all_status()

@app.get("/api/nodeinfo")
async def nodeinfo(user:dict=Depends(current_user)):
    m=demo_metrics() if config.get("app.demo_mode") else await asyncio.to_thread(system_metrics,config.path_value("paths.hf_cache"))
    return {"hostname":m["hostname"],"ip":m["ip"],"port":config.get("app.port"),"arch":m["architecture"],"memory_gb":m["memory_total_gb"],"services":{k:service_base(k) for k in ("ollama","litellm",*ENGINE_KEYS)},"version":__version__}

@app.get("/api/dashboard")
async def dashboard(user:dict=Depends(current_user)):
    metrics=demo_metrics() if config.get("app.demo_mode") else await asyncio.to_thread(system_metrics,config.path_value("paths.hf_cache"))
    status=await all_status(); models=await inventory_all(); deployments=compose.list_deployments()
    if config.get("app.demo_mode"):
        deployments=[{"name":"Qwen3.6 35B","slug":"vllm-qwen36","engine":"vllm","model_name":"example/Qwen3.6-35B-A3B-NVFP4","fit_status":"good","estimated_runtime_gb":42.0,"port":8000},{"name":"Mistral Small 4","slug":"sglang-mistral","engine":"sglang","model_name":"example/Mistral-Small-4-119B-NVFP4","fit_status":"tight","estimated_runtime_gb":91.0,"port":30000}]
    return {"metrics":metrics,"status":status,"model_count":len(models),"model_size_gb":round(sum(float(m.get("size_gb") or 0) for m in models),1),"deployments":deployments,"nodes":db.list_nodes(),"compose_version":await docker_compose_version() if not config.get("app.demo_mode") else "5.4.0"}

# -----------------------------------------------------------------------------
# Ollama
# -----------------------------------------------------------------------------
class PullReq(BaseModel): name:str=Field(min_length=1,max_length=200)
@app.get("/api/ollama/models")
async def get_ollama(user:dict=Depends(current_user)):
    if config.get("app.demo_mode"): return {"models":[{"name":"glm-4.7-flash","size":19000000000,"details":{"parameter_size":"29.9B","quantization_level":"Q4_K_M"}}]}
    try:
        r=await app.state.http.get(service_base("ollama").rstrip("/")+"/api/tags",timeout=6); r.raise_for_status(); return r.json()
    except Exception as exc: raise HTTPException(502,f"Ollama unreachable: {type(exc).__name__}")

@app.post("/api/ollama/pull")
async def pull_ollama(req:PullReq,request:Request,user:dict=Depends(role_and_csrf("operator"))):
    if config.get("app.demo_mode"):
        async def demo():
            for pct in (12,38,67,91,100): yield f'data: {json.dumps({"status":"pulling manifest" if pct<100 else "success","completed":pct,"total":100})}\n\n'; await asyncio.sleep(.1)
            yield 'data: {"done":true}\n\n'
        return StreamingResponse(demo(),media_type="text/event-stream")
    audit(request,user,"ollama.pull",req.name)
    async def stream()->AsyncGenerator[str,None]:
        try:
            async with httpx.AsyncClient(timeout=None) as c:
                async with c.stream("POST",service_base("ollama").rstrip("/")+"/api/pull",json={"name":req.name,"stream":True}) as resp:
                    if resp.status_code>=400: yield f"data: {json.dumps({'error':f'HTTP {resp.status_code}'})}\n\n"; return
                    async for line in resp.aiter_lines():
                        if line: yield f"data: {line}\n\n"
        except Exception as exc: yield f"data: {json.dumps({'error':str(exc)})}\n\n"
        yield 'data: {"done":true}\n\n'
    return StreamingResponse(stream(),media_type="text/event-stream",headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

@app.delete("/api/ollama/models/{name:path}")
async def delete_ollama(name:str,request:Request,user:dict=Depends(role_and_csrf("operator"))):
    audit(request,user,"ollama.delete",name)
    if config.get("app.demo_mode"): return {"ok":True}
    r=await app.state.http.request("DELETE",service_base("ollama").rstrip("/")+"/api/delete",json={"name":name},timeout=60)
    if r.status_code not in {200,204}: raise HTTPException(r.status_code,"Ollama delete failed")
    return {"ok":True}

# -----------------------------------------------------------------------------
# LiteLLM
# -----------------------------------------------------------------------------
@app.get("/api/litellm/models")
async def litellm_models(user:dict=Depends(current_user)):
    if config.get("app.demo_mode"): return {"data":[{"id":"ollama/*"},{"id":"Qwen3.6-35B-A3B"}]}
    try:
        r=await app.state.http.get(
            service_base("litellm").rstrip("/")+"/v1/models",
            headers=litellm_auth_headers(),
            timeout=6,
        )
        r.raise_for_status()
        return r.json()
    except Exception as exc: raise HTTPException(502,f"LiteLLM unreachable: {type(exc).__name__}")

@app.get("/api/litellm/config")
async def litellm_config(user:dict=Depends(current_user)):
    cfg=load_litellm(); safe=redact(cfg); safe["_raw"]=yaml.safe_dump(safe,sort_keys=False); return safe

@app.post("/api/litellm/apply-wildcard")
async def litellm_wildcard(request:Request,user:dict=Depends(role_and_csrf("operator"))):
    cfg=load_litellm(); ml=cfg.get("model_list",[])
    if not any(x.get("model_name")=="ollama/*" for x in ml if isinstance(x,dict)):
        ml=[x for x in ml if not (isinstance(x,dict) and str(x.get("model_name","")).startswith("ollama/"))]
        ml.append({"model_name":"ollama/*","litellm_params":{"model":"ollama/*","api_base":service_base("ollama")}}); cfg["model_list"]=ml
        if not config.get("app.demo_mode"): save_litellm(cfg)
    result={"ok":True,"message":"Wildcard present"}
    if not config.get("app.demo_mode"): result["restart"]=await restart_litellm()
    audit(request,user,"litellm.wildcard"); return result

@app.post("/api/litellm/restart")
async def litellm_restart(request:Request,user:dict=Depends(role_and_csrf("operator"))):
    audit(request,user,"litellm.restart"); return {"ok":True,"output":"demo"} if config.get("app.demo_mode") else await restart_litellm()


@app.post("/api/compose/deployments/{engine}/{slug}/litellm")
async def route_deployment_litellm(engine:str,slug:str,request:Request,user:dict=Depends(role_and_csrf("operator"))):
    """Add a local OpenAI-compatible Compose deployment to LiteLLM."""
    if engine not in {"vllm","sglang","llamacpp"}:
        raise HTTPException(400,"This engine does not expose the model-serving API used by this routing helper")
    if config.get("app.demo_mode"):
        audit(request,user,"litellm.route_add",slug,{"engine":engine,"demo":True})
        return {"ok":True,"model_name":slug,"message":"Demo route added","restart":{"ok":True}}
    try: dep=compose.get(engine,slug)
    except FileNotFoundError: raise HTTPException(404,"Deployment not found")
    served=str(dep.get("served_model_name") or dep.get("model_name") or slug).strip()
    if not served: raise HTTPException(400,"Deployment does not have a routable model name")
    port=int(dep.get("port") or compose.engine(engine).get("port") or 8000)
    cfg=load_litellm(); model_list=cfg.get("model_list") or []
    if not isinstance(model_list,list): raise HTTPException(400,"LiteLLM model_list is not a list")
    entry={"model_name":served,"litellm_params":{"model":f"openai/{served}","api_base":f"http://127.0.0.1:{port}/v1"}}
    replaced=False
    for i,item in enumerate(model_list):
        if isinstance(item,dict) and item.get("model_name")==served:
            model_list[i]=entry; replaced=True; break
    if not replaced: model_list.append(entry)
    cfg["model_list"]=model_list; save_litellm(cfg)
    restart=await restart_litellm()
    audit(request,user,"litellm.route_add",served,{"engine":engine,"slug":slug,"port":port})
    return {"ok":True,"model_name":served,"message":"LiteLLM route saved","restart":restart}

@app.delete("/api/compose/deployments/{engine}/{slug}/litellm")
async def unroute_deployment_litellm(engine:str,slug:str,request:Request,user:dict=Depends(role_and_csrf("operator"))):
    if config.get("app.demo_mode"):
        audit(request,user,"litellm.route_remove",slug,{"engine":engine,"demo":True}); return {"ok":True,"restart":{"ok":True}}
    try: dep=compose.get(engine,slug)
    except FileNotFoundError: raise HTTPException(404,"Deployment not found")
    served=str(dep.get("served_model_name") or dep.get("model_name") or slug).strip()
    cfg=load_litellm(); ml=cfg.get("model_list") or []
    cfg["model_list"]=[x for x in ml if not (isinstance(x,dict) and x.get("model_name")==served)]
    save_litellm(cfg); restart=await restart_litellm()
    audit(request,user,"litellm.route_remove",served,{"engine":engine,"slug":slug})
    return {"ok":True,"model_name":served,"restart":restart}

# -----------------------------------------------------------------------------
# Inventory / HuggingFace
# -----------------------------------------------------------------------------
class AddDirReq(BaseModel): path:str
class DeleteModelReq(BaseModel): path:str
class HFDownloadReq(BaseModel): repo_id:str; local_dir:Optional[str]=None

@app.get("/api/inventory")
async def inventory(include_ollama:bool=True,user:dict=Depends(current_user)):
    rows=await inventory_all();
    if not include_ollama: rows=[x for x in rows if x.get("source")!="ollama"]
    return {"models":rows,"directories":[]}

@app.get("/api/hf/inventory")
async def hf_inventory(user:dict=Depends(current_user)): return {"models":[m for m in await inventory_all() if m.get("source")!="ollama"]}
@app.get("/api/hf/inventory/dirs")
async def inventory_dirs(user:dict=Depends(current_user)):
    return {"dirs":[{"path":str(config.path_value("paths.hf_cache")),"default":True}]+[{"path":d,"default":False} for d in load_custom_dirs(config)]}
@app.post("/api/hf/inventory/dirs")
async def add_dir(req:AddDirReq,request:Request,user:dict=Depends(role_and_csrf("operator"))):
    try: p=validate_custom_dir(req.path)
    except ValueError as exc: raise HTTPException(400,str(exc))
    dirs=load_custom_dirs(config); s=str(p)
    if s not in dirs: dirs.append(s); save_custom_dirs(config,dirs)
    audit(request,user,"inventory.dir.add",s); return {"ok":True,"dirs":dirs}
@app.delete("/api/hf/inventory/dirs")
async def remove_dir(path:str,request:Request,user:dict=Depends(role_and_csrf("operator"))):
    p=str(Path(os.path.expanduser(path)).resolve()); dirs=[d for d in load_custom_dirs(config) if str(Path(os.path.expanduser(d)).resolve())!=p]; save_custom_dirs(config,dirs); audit(request,user,"inventory.dir.remove",p); return {"ok":True,"dirs":dirs}
@app.post("/api/hf/inventory/delete")
async def delete_inv(req:DeleteModelReq,request:Request,user:dict=Depends(role_and_csrf("operator"))):
    try: deleted=safe_delete_model(config,req.path)
    except ValueError as exc: raise HTTPException(400,str(exc))
    audit(request,user,"inventory.model.delete",deleted); return {"ok":True,"deleted":deleted}

@app.get("/api/hf/search")
async def search_hf(q:str,sort:str="downloads",limit:int=20,pipeline_tag:Optional[str]=None,user:dict=Depends(current_user)):
    if config.get("app.demo_mode"):
        return {"models":[{"id":"google/gemma-4-31B-it","pipeline_tag":"image-text-to-text","task_label":"Vision LLM","downloads":5100000,"likes":2300,"tags":["safetensors","transformers","gemma4","image-text-to-text","apache-2.0"],"has_safetensors":True},{"id":"unsloth/gemma-4-26B-A4B-it-GGUF","pipeline_tag":"image-text-to-text","task_label":"Vision LLM","downloads":2800000,"likes":582,"tags":["gguf","gemma4","quantized"],"has_safetensors":False}]}
    try: return {"models":await hf_search(app.state.http,q,sort,limit,pipeline_tag)}
    except Exception as exc: raise HTTPException(502,f"HuggingFace API error: {type(exc).__name__}")
@app.get("/api/hf/search/variants")
async def variants(model_id:str,user:dict=Depends(current_user)):
    if config.get("app.demo_mode"): return {"variants":[{"id":model_id+"-GGUF","type":"GGUF","downloads":180000}]}
    return {"variants":await hf_variants(app.state.http,model_id)}
@app.get("/api/hf/model/{owner}/{name}/files")
async def model_files(owner:str,name:str,user:dict=Depends(current_user)):
    if config.get("app.demo_mode"): return {"files":[{"name":"config.json","size":12000},{"name":"model-00001-of-00002.safetensors","size":12000000000},{"name":"model-00002-of-00002.safetensors","size":11800000000}]}
    return {"files":await asyncio.to_thread(hf_files,owner,name)}

HF_RE=re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
@app.post("/api/hf/download")
async def hf_download(req:HFDownloadReq,request:Request,user:dict=Depends(role_and_csrf("operator"))):
    repo=req.repo_id.strip()
    if not HF_RE.fullmatch(repo): raise HTTPException(400,"Invalid repository ID")
    local_dir=(req.local_dir or "").strip()
    if local_dir and ("\0" in local_dir or "\n" in local_dir or ".." in Path(local_dir).parts): raise HTTPException(400,"Invalid local directory")
    if local_dir:
        parent=str(Path(os.path.expanduser(local_dir)).resolve().parent); dirs=load_custom_dirs(config)
        if parent not in dirs: dirs.append(parent); save_custom_dirs(config,dirs)
    audit(request,user,"hf.download",repo,{"custom_dir":bool(local_dir)})
    if config.get("app.demo_mode"):
        async def demo():
            for i,pct in enumerate((10,35,62,88,100),1):
                yield f"data: {json.dumps({'progress':{'pct':pct,'idx':i,'total_files':5,'file':f'file-{i}.safetensors','done_mb':pct*100,'total_mb':10000,'speed':'850 MiB/s'}})}\n\n"; await asyncio.sleep(.12)
            yield f"data: {json.dumps({'status':'complete','path':local_dir or '~/.cache/huggingface/hub','avg_speed':'820 MiB/s','elapsed':'14s','errors':0})}\n\n"
        return StreamingResponse(demo(),media_type="text/event-stream")
    async def stream():
        env={**os.environ,"HF_REPO_ID":repo};
        if local_dir: env["HF_LOCAL_DIR"]=local_dir
        code='''import json,os,time\nfrom huggingface_hub import list_repo_tree,hf_hub_download\nrepo=os.environ["HF_REPO_ID"]; local=os.environ.get("HF_LOCAL_DIR") or None\nJ=lambda **kw: print(json.dumps(kw),flush=True)\ntry:\n entries=[e for e in list_repo_tree(repo,recursive=True) if hasattr(e,"size") and not e.path.startswith(".")]\n total=sum(e.size or 0 for e in entries); done=0; start=time.time(); errors=0; result=None\n for i,e in enumerate(entries,1):\n  try:\n   kw={"repo_id":repo,"filename":e.path};\n   if local: kw["local_dir"]=local\n   path=hf_hub_download(**kw); result=result or os.path.dirname(path); done+=e.size or 0\n  except Exception as x: errors+=1; J(file_error={"idx":i,"name":e.path,"error":str(x)}); continue\n  el=max(time.time()-start,.01); J(progress={"pct":round(done/total*100,1) if total else 100,"idx":i,"total_files":len(entries),"file":e.path,"done_mb":round(done/1048576,1),"total_mb":round(total/1048576,1),"speed":f"{done/el/1048576:.0f} MiB/s"})\n J(status="complete",path=local or result or "HF cache",avg_speed=f"{done/max(time.time()-start,.01)/1048576:.0f} MiB/s",elapsed=f"{time.time()-start:.0f}s",errors=errors)\nexcept Exception as e: J(status="error",error=str(e))\n'''
        proc=await asyncio.create_subprocess_exec(sys.executable,"-c",code,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE,env=env)
        assert proc.stdout
        async for raw in proc.stdout:
            line=raw.decode().strip()
            if line: yield f"data: {line}\n\n"
        err=(await proc.stderr.read()).decode().strip()
        if err: yield f"data: {json.dumps({'log':err[-4000:]})}\n\n"
    return StreamingResponse(stream(),media_type="text/event-stream",headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

# -----------------------------------------------------------------------------
# Compose generator / deployments / legacy engines
# -----------------------------------------------------------------------------
class GenerateReq(BaseModel):
    model_id:str; engine:str; name:Optional[str]=None; context_length:Optional[int]=None; memory_reserve_gb:Optional[float]=None; profile:str="balanced"; bind_host:Optional[str]=None; expose_litellm:bool=True; node_id:Optional[int]=None
class SavePlanReq(BaseModel): plan:dict; node_id:Optional[int]=None
class EngineStartReq(BaseModel): profile:str

@app.get("/api/compose/catalog")
async def compose_catalog(user:dict=Depends(current_user)): return {"engines":compose.catalog}
@app.post("/api/compose/generate")
async def generate_compose(req:GenerateReq,user:dict=Depends(current_user)):
    if req.engine not in ENGINE_KEYS: raise HTTPException(400,"Unsupported engine")
    if req.node_id and not config.get("app.demo_mode"):
        payload=req.model_dump(exclude={"node_id"})
        try:
            plan=await node_client.generate(req.node_id,payload)
        except Exception as exc:
            raise HTTPException(502,f"Remote node could not generate the deployment: {exc}")
        node=db.get_node(req.node_id)
        plan["node_id"]=req.node_id; plan["node"]=node.get("name") if node else "remote"
        return plan
    model=await find_model(req.model_id)
    metrics=demo_metrics() if config.get("app.demo_mode") else await asyncio.to_thread(system_metrics,config.path_value("paths.hf_cache"))
    # Demo inventory paths are not real. Replace only for generator demonstration.
    if config.get("app.demo_mode"):
        model=dict(model); model["source"]="custom_dir"; model["runtime_path"]=str(APP_ROOT); model["dir_path"]=str(APP_ROOT)
    try: plan=compose.generate(model=model,engine_key=req.engine,node_metrics=metrics,name=req.name,context_length=req.context_length,memory_reserve_gb=req.memory_reserve_gb,profile=req.profile,bind_host=req.bind_host,expose_litellm=req.expose_litellm)
    except ValueError as exc: raise HTTPException(400,str(exc))
    return plan

@app.post("/api/compose/deployments")
async def save_plan(req:SavePlanReq,request:Request,user:dict=Depends(role_and_csrf("operator"))):
    plan=req.plan
    required={"slug","engine","yaml","model_name"}
    if not required.issubset(plan): raise HTTPException(400,"Invalid generated plan")
    if req.node_id:
        try: saved=await node_client.save_plan(req.node_id,plan)
        except Exception as exc: raise HTTPException(502,f"Remote node rejected deployment: {exc}")
    else:
        if config.get("app.demo_mode"): saved={**plan,"path":"/demo/compose/"+plan["slug"]}
        else:
            try:
                model=await find_model(str(plan.get("model_id") or ""))
                metrics=await asyncio.to_thread(system_metrics,config.path_value("paths.hf_cache"))
                safe=compose.validate_generated_plan(plan,model=model,node_metrics=metrics)
                saved=compose.save_generated(safe)
            except ValueError as exc:
                raise HTTPException(400,str(exc))
    audit(request,user,"compose.save",plan["slug"],{"engine":plan["engine"],"node_id":req.node_id}); return saved

@app.get("/api/compose/deployments")
async def list_deployments(user:dict=Depends(current_user)):
    local=compose.list_deployments()
    for x in local:
        x.setdefault("node","local"); x["node_id"]=None
    if config.get("app.demo_mode"):
        local=[{"name":"Qwen3.6 35B","slug":"vllm-qwen36","engine":"vllm","model_name":"example/Qwen3.6-35B-A3B-NVFP4","fit_status":"good","estimated_runtime_gb":42.0,"memory_budget_gb":104,"port":8000,"node":"spark-a","node_id":None},{"name":"Mistral Small 4","slug":"sglang-mistral","engine":"sglang","model_name":"example/Mistral-Small-4-119B-NVFP4","fit_status":"tight","estimated_runtime_gb":91.0,"memory_budget_gb":104,"port":30000,"node":"spark-a","node_id":None}]
        return {"deployments":local}
    remote=[]
    for n in db.list_nodes():
        if not n.get("enabled",1): continue
        try:
            data=await node_client.deployments(n["id"])
            for x in data.get("deployments",[]):
                x["node_id"]=n["id"]; x["node"]=n["name"]; remote.append(x)
        except Exception:
            pass
    return {"deployments":local+remote}

@app.get("/api/compose/deployments/{engine}/{slug}")
async def get_deployment(engine:str,slug:str,user:dict=Depends(current_user)):
    try: return compose.get(engine,slug)
    except FileNotFoundError: raise HTTPException(404,"Deployment not found")

@app.post("/api/compose/deployments/{engine}/{slug}/up")
async def deploy_up(engine:str,slug:str,request:Request,node_id:Optional[int]=None,user:dict=Depends(role_and_csrf("operator"))):
    if config.get("app.demo_mode"): result={"ok":True,"output":"Demo deployment started"}
    elif node_id: result=await node_client.up(node_id,engine,slug)
    else: result=await compose.up(engine,slug)
    audit(request,user,"compose.up",slug,{"engine":engine,"node_id":node_id}); return result
@app.post("/api/compose/deployments/{engine}/{slug}/down")
async def deploy_down(engine:str,slug:str,request:Request,node_id:Optional[int]=None,user:dict=Depends(role_and_csrf("operator"))):
    if config.get("app.demo_mode"): result={"ok":True,"output":"Demo deployment stopped"}
    elif node_id: result=await node_client.down(node_id,engine,slug)
    else: result=await compose.down(engine,slug)
    audit(request,user,"compose.down",slug,{"engine":engine,"node_id":node_id}); return result
@app.get("/api/compose/deployments/{engine}/{slug}/status")
async def deploy_status(engine:str,slug:str,node_id:Optional[int]=None,user:dict=Depends(current_user)):
    if config.get("app.demo_mode"): return {"running":slug=="vllm-qwen36","state":"running" if slug=="vllm-qwen36" else "stopped","containers":[]}
    return await node_client.status(node_id,engine,slug) if node_id else await compose.status(engine,slug)
@app.get("/api/compose/deployments/{engine}/{slug}/logs")
async def deploy_logs(engine:str,slug:str,lines:int=200,node_id:Optional[int]=None,user:dict=Depends(current_user)):
    if config.get("app.demo_mode"): return {"ok":True,"lines":["[demo] container initialized","[demo] model loading complete","[demo] API listening"]}
    return await node_client.logs(node_id,engine,slug,lines) if node_id else await compose.logs(engine,slug,lines)
@app.delete("/api/compose/deployments/{engine}/{slug}")
async def deploy_remove(engine:str,slug:str,request:Request,node_id:Optional[int]=None,user:dict=Depends(role_and_csrf("operator"))):
    if config.get("app.demo_mode"): result={"ok":True,"archived":"/demo/archive"}
    elif node_id: result=await node_client.remove(node_id,engine,slug)
    else: result=await compose.remove(engine,slug,True)
    audit(request,user,"compose.remove",slug,{"engine":engine,"node_id":node_id}); return result

@app.get("/api/{engine}/profiles")
async def engine_profiles(engine:str,user:dict=Depends(current_user)):
    if engine not in ENGINE_KEYS: raise HTTPException(404,"Unknown engine")
    profiles=compose.profiles_for_engine(engine)+scan_legacy(config,engine)
    if config.get("app.demo_mode") and engine=="vllm": profiles=[{"id":"compose:vllm:vllm-qwen36","name":"Qwen3.6 35B","description":"Compose deployment · example/Qwen3.6-35B-A3B-NVFP4","vram_gb":42,"kind":"compose","slug":"vllm-qwen36"}]
    return profiles
@app.get("/api/{engine}/status")
async def generic_engine_status(engine:str,user:dict=Depends(current_user)):
    if engine not in ENGINE_KEYS: raise HTTPException(404,"Unknown engine")
    if config.get("app.demo_mode"):
        running=engine=="vllm"; return {"running":running,"model":"example/Qwen3.6-35B-A3B-NVFP4" if running else None,"container_info":"dmm-vllm-qwen36 Up" if running else ""}
    return await engine_status(engine)
@app.post("/api/{engine}/start")
async def generic_engine_start(engine:str,req:EngineStartReq,request:Request,user:dict=Depends(role_and_csrf("operator"))):
    if engine not in ENGINE_KEYS: raise HTTPException(404,"Unknown engine")
    if req.profile.startswith("compose:"):
        parts=req.profile.split(":",2)
        if len(parts)!=3 or parts[1]!=engine: raise HTTPException(400,"Invalid profile")
        result={"ok":True,"message":"Demo deployment starting"} if config.get("app.demo_mode") else await compose.up(engine,parts[2])
    elif req.profile.startswith("legacy:"):
        if not config.get("app.legacy_scripts_enabled"): raise HTTPException(403,"Legacy script mode is disabled")
        p=next((x for x in scan_legacy(config,engine) if x["id"]==req.profile),None)
        if not p: raise HTTPException(404,"Legacy profile not found")
        log_path=launch_legacy(p,config.path_value("paths.compose_root").parent/"logs"); result={"ok":True,"message":f"Legacy script launched; logs: {log_path}"}
    else: raise HTTPException(404,"Profile not found")
    audit(request,user,"engine.start",req.profile,{"engine":engine}); return result
@app.post("/api/{engine}/stop")
async def generic_engine_stop(engine:str,request:Request,user:dict=Depends(role_and_csrf("operator"))):
    if engine not in ENGINE_KEYS: raise HTTPException(404,"Unknown engine")
    if config.get("app.demo_mode"): result={"ok":True,"output":"Demo engine stopped"}
    else:
        result=None
        for p in compose.profiles_for_engine(engine):
            st=await compose.status(engine,p["slug"])
            if st.get("running"): result=await compose.down(engine,p["slug"]); break
        if result is None:
            raise HTTPException(
                409,
                "Engine is running, but it is not managed by DGX Model Manager v2. "
                "Refusing to stop an externally managed service."
            )
    audit(request,user,"engine.stop",engine); return result

# -----------------------------------------------------------------------------
# Nodes
# -----------------------------------------------------------------------------
class NodeReq(BaseModel): id:Optional[int]=None; name:str; base_url:str; token:Optional[str]=None; verify_tls:bool=True; tls_fingerprint:str=""; notes:str=""
@app.get("/api/nodes")
async def nodes(user:dict=Depends(current_user)): return {"nodes":db.list_nodes(),"local":{"id":None,"name":"Local node","base_url":"local"}}
@app.get("/api/nodes/{node_id}/inventory")
async def node_inventory(node_id:int,user:dict=Depends(current_user)):
    try: return await node_client.inventory(node_id)
    except Exception as exc: raise HTTPException(502,f"Remote node inventory unavailable: {exc}")
@app.get("/api/nodes/{node_id}/dashboard")
async def node_dashboard(node_id:int,user:dict=Depends(current_user)):
    try:
        data=await node_client.dashboard(node_id)
        node=db.get_node(node_id)
        data["nodes"]=db.list_nodes()
        data["selected_node"]={"id":node_id,"name":node.get("name") if node else "remote"}
        return data
    except Exception as exc: raise HTTPException(502,f"Remote node dashboard unavailable: {exc}")
@app.post("/api/nodes")
async def save_node(req:NodeReq,request:Request,user:dict=Depends(role_and_csrf("admin"))):
    ok,msg=is_allowed_service_url(req.base_url,allow_public=False)
    if not ok: raise HTTPException(400,msg)
    if not req.base_url.startswith("https://") and not config.get("nodes.allow_insecure_http"): raise HTTPException(400,"Remote node agents must use HTTPS unless nodes.allow_insecure_http is explicitly enabled")
    fingerprint=req.tls_fingerprint.replace(":","").strip().lower()
    if req.base_url.startswith("https://") and not req.verify_tls and not re.fullmatch(r"[a-f0-9]{64}",fingerprint): raise HTTPException(400,"A 64-character SHA-256 certificate fingerprint is required when standard TLS verification is disabled")
    try: node=db.upsert_node(node_id=req.id,name=req.name.strip(),base_url=req.base_url.rstrip("/"),token=req.token,verify_tls=req.verify_tls,tls_fingerprint=fingerprint,notes=req.notes)
    except Exception as exc: raise HTTPException(400,str(exc))
    audit(request,user,"node.save",node["name"],{"url":node["base_url"],"verify_tls":node["verify_tls"]}); return node
@app.get("/api/nodes/{node_id}/test")
async def test_node(node_id:int,user:dict=Depends(require_role("admin"))):
    try: return await node_client.info(node_id)
    except Exception as exc: raise HTTPException(502,f"Node unavailable: {exc}")
@app.delete("/api/nodes/{node_id}")
async def remove_node(node_id:int,request:Request,user:dict=Depends(role_and_csrf("admin"))): db.delete_node(node_id); audit(request,user,"node.delete",str(node_id)); return {"ok":True}

# -----------------------------------------------------------------------------
# Settings / diagnostics / logs
# -----------------------------------------------------------------------------
class ConfigReq(BaseModel):
    display_name:Optional[str]=None; services:Optional[dict]=None; legacy_scripts_enabled:Optional[bool]=None; allow_registration:Optional[bool]=None; compose:Optional[dict]=None
class TestServiceReq(BaseModel): url:str; type:str

@app.get("/api/config")
async def get_config(user:dict=Depends(current_user)): return config.public_dict()
@app.put("/api/config")
async def update_config(req:ConfigReq,request:Request,user:dict=Depends(role_and_csrf("admin"))):
    if req.display_name is not None: config.set("app.display_name",req.display_name.strip()[:100])
    if req.legacy_scripts_enabled is not None: config.set("app.legacy_scripts_enabled",req.legacy_scripts_enabled)
    if req.allow_registration is not None: config.set("app.allow_registration",req.allow_registration)
    if req.services:
        for key,val in req.services.items():
            if key not in {"ollama_base","litellm_base",*[f"{k}_base" for k in ENGINE_KEYS]}: continue
            ok,msg=is_allowed_service_url(str(val),bool(config.get("security.allow_public_service_targets")))
            if not ok: raise HTTPException(400,f"{key}: {msg}")
            config.set(f"services.{key}",str(val).rstrip("/"))
    if req.compose:
        allowed={"bind_host","default_memory_reserve_gb","default_context_length","default_profile","images"}
        for key,val in req.compose.items():
            if key in allowed: config.set(f"compose.{key}",val)
    if not config.get("app.demo_mode"): config.save()
    audit(request,user,"config.update","settings",req.model_dump(exclude_none=True)); return config.public_dict()

@app.post("/api/test-service")
async def test_service(req:TestServiceReq,user:dict=Depends(role_and_csrf("admin"))):
    ok,msg=is_allowed_service_url(req.url,bool(config.get("security.allow_public_service_targets")))
    if not ok: raise HTTPException(400,msg)
    path={"ollama":"/api/tags","litellm":"/health",**{k:engine_health_path(k) for k in ENGINE_KEYS}}.get(req.type,"/health")
    headers=litellm_auth_headers() if req.type=="litellm" else None
    return await service_check(
        app.state.http,
        req.url.rstrip("/"),
        path,
        5,
        headers=headers,
    )

@app.get("/api/sudo/check")
async def sudo_check(user:dict=Depends(current_user)):
    if config.get("app.demo_mode"): return {"systemctl":True,"docker":True,"compose":True}
    d=await docker_available(); cv=await docker_compose_version(); r=await run_cmd("sudo","-n","systemctl","is-active","litellm",timeout=5)
    return {"systemctl":r.returncode in {0,3},"docker":d,"compose":bool(cv),"compose_version":cv}

@app.get("/api/debug/system")
async def debug_system(user:dict=Depends(current_user)):
    metrics=demo_metrics() if config.get("app.demo_mode") else await asyncio.to_thread(system_metrics,config.path_value("paths.hf_cache")); status=await all_status()
    return {**metrics,"python_version":sys.version.split()[0],"app_port":config.get("app.port"),"app_start_utc":START_UTC,"uptime_seconds":int(time.monotonic()-START_MONO),"services":status,"permissions":await sudo_check(user)}
@app.get("/api/debug/config")
async def debug_config(user:dict=Depends(require_role("admin"))): return {"config":config.public_dict(),"litellm":redact(load_litellm()),"deployments":compose.list_deployments(),"legacy_enabled":config.get("app.legacy_scripts_enabled")}
@app.get("/api/debug/docker")
async def debug_docker(user:dict=Depends(current_user)): return {"containers":[] if config.get("app.demo_mode") else await docker_containers(),"available":True}
@app.get("/api/logs/app")
async def app_logs(level:Optional[str]=None,search:Optional[str]=None,limit:int=250,user:dict=Depends(current_user)): return {"entries":ring.entries(level,search,limit),"total":len(ring.buffer),"buffer_size":ring.buffer.maxlen}
@app.delete("/api/logs/app")
async def clear_logs(request:Request,user:dict=Depends(role_and_csrf("admin"))): ring.buffer.clear(); audit(request,user,"logs.clear","app"); return {"ok":True}
@app.get("/api/logs/litellm")
async def litellm_logs(lines:int=100,search:Optional[str]=None,user:dict=Depends(current_user)):
    if config.get("app.demo_mode"): return {"lines":["2026-08-10T15:20:01 INFO LiteLLM proxy healthy","2026-08-10T15:20:05 INFO route Qwen3.6 ready"],"available":True}
    for cmd in (("journalctl","-u","litellm","--no-pager","-n",str(max(1,min(lines,1000))),"--output=short-iso"),("sudo","-n","journalctl","-u","litellm","--no-pager","-n",str(max(1,min(lines,1000))),"--output=short-iso")):
        r=await run_cmd(*cmd,timeout=10)
        if r.returncode==0:
            rows=r.stdout.splitlines();
            if search: rows=[x for x in rows if search.lower() in x.lower()]
            return {"lines":rows,"available":True}
    return {"lines":[],"available":False,"error":"journalctl access denied"}
@app.get("/api/logs/engine/{engine}")
async def engine_logs(engine:str,lines:int=200,search:Optional[str]=None,user:dict=Depends(current_user)):
    if engine not in ENGINE_KEYS: raise HTTPException(404,"Unknown engine")
    rows=[]
    for p in compose.profiles_for_engine(engine):
        try: rows.extend((await compose.logs(engine,p["slug"],lines)).get("lines",[]))
        except Exception: pass
    if config.get("app.legacy_scripts_enabled"):
        logdir=config.path_value("paths.compose_root").parent/"logs"
        for f in sorted(logdir.glob("legacy_*.log"),key=lambda p:p.stat().st_mtime,reverse=True)[:3] if logdir.exists() else []:
            try: rows.extend(f.read_text(errors="replace").splitlines()[-lines:])
            except Exception: pass
    if search: rows=[x for x in rows if search.lower() in x.lower()]
    return {"lines":rows[-lines:],"available":True}
@app.get("/api/audit")
async def audit_log(limit:int=200,user:dict=Depends(require_role("admin"))): return {"entries":db.list_audit(limit)}

# -----------------------------------------------------------------------------
# Backward-compatible aliases from the existing app where names differed.
# -----------------------------------------------------------------------------
@app.get("/api/scriptdirs")
async def scriptdirs(user:dict=Depends(current_user)):
    return {k:str(config.path_value(f"paths.legacy_{k}_scripts")) for k in ENGINE_KEYS}

if __name__=="__main__":
    host=str(config.get("app.host","0.0.0.0")); port=int(config.get("app.port",8091)); kwargs={}
    if config.get("tls.enabled") and not config.get("app.demo_mode"):
        cert=Path(os.path.expanduser(str(config.get("tls.cert_file")))); key=Path(os.path.expanduser(str(config.get("tls.key_file"))))
        if cert.exists() and key.exists(): kwargs={"ssl_certfile":str(cert),"ssl_keyfile":str(key)}
        elif config.get("security.require_https"):
            raise SystemExit(f"TLS is enabled but certificate files are missing. Run setup.sh or configure {cert} and {key}.")
    uvicorn.run(app,host=host,port=port,log_level="info",**kwargs)
