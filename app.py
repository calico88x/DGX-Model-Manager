#!/usr/bin/env python3
"""
DGX Spark Model Manager
Unified web UI for managing models across Ollama, SGLang, and LiteLLM.

Configure via config.json before running.
Default port: 8090
"""

import asyncio
import json
import os
import subprocess
from pathlib import Path
from typing import AsyncGenerator, Optional

import httpx
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn

# ─── Config ───────────────────────────────────────────────────────────────────

CONFIG_FILE  = Path(__file__).parent / "config.json"
PROFILES_FILE = Path(__file__).parent / "sglang_profiles.json"


def load_config() -> dict:
    defaults = {
        "app": {
            "host": "0.0.0.0",
            "port": 8090,
            "display_name": "DGX Spark",
        },
        "services": {
            "ollama_base":  "http://127.0.0.1:11434",
            "litellm_base": "http://127.0.0.1:4000",
            "sglang_base":  "http://127.0.0.1:30000",
        },
        "paths": {
            "litellm_config": "~/litellm/litellm_config.yaml",
            "hf_cache":       "~/.cache/huggingface/hub",
        },
    }
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            user = json.load(f)
        for section, values in user.items():
            if section in defaults and isinstance(values, dict):
                defaults[section].update(values)
    return defaults


CFG = load_config()

OLLAMA_BASE     = CFG["services"]["ollama_base"]
LITELLM_BASE    = CFG["services"]["litellm_base"]
SGLANG_BASE     = CFG["services"]["sglang_base"]
LITELLM_CONFIG  = Path(CFG["paths"]["litellm_config"]).expanduser()
HF_CACHE        = Path(CFG["paths"]["hf_cache"]).expanduser()
DISPLAY_NAME    = CFG["app"]["display_name"]
APP_HOST        = CFG["app"]["host"]
APP_PORT        = int(CFG["app"]["port"])

# ─── Pydantic models ──────────────────────────────────────────────────────────

class PullRequest(BaseModel):
    name: str

class SGLangStartRequest(BaseModel):
    profile: str

class HFDownloadRequest(BaseModel):
    repo_id: str
    local_dir: Optional[str] = None

# ─── Helpers ──────────────────────────────────────────────────────────────────

async def service_ok(base: str, path: str = "/health") -> bool:
    try:
        async with httpx.AsyncClient(timeout=3.0) as c:
            r = await c.get(base + path)
            return r.status_code < 400
    except Exception:
        return False


def load_litellm_config() -> dict:
    if LITELLM_CONFIG.exists():
        with open(LITELLM_CONFIG) as f:
            return yaml.safe_load(f) or {}
    return {}


def save_litellm_config(cfg: dict):
    LITELLM_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    with open(LITELLM_CONFIG, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def load_profiles() -> list:
    if PROFILES_FILE.exists():
        with open(PROFILES_FILE) as f:
            return json.load(f)
    return []

# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="DGX Model Manager")

# ── Config endpoint ───────────────────────────────────────────────────────────

@app.get("/api/config")
async def get_app_config():
    return {
        "display_name": DISPLAY_NAME,
        "sglang_base":  SGLANG_BASE,
        "litellm_base": LITELLM_BASE,
        "ollama_base":  OLLAMA_BASE,
        "port":         APP_PORT,
    }

# ── Status ────────────────────────────────────────────────────────────────────

@app.get("/api/status")
async def get_status():
    sglang_ok, ollama_ok, litellm_ok = await asyncio.gather(
        service_ok(SGLANG_BASE, "/health"),
        service_ok(OLLAMA_BASE, "/api/tags"),
        service_ok(LITELLM_BASE, "/health"),
    )
    sglang_model = None
    if sglang_ok:
        try:
            async with httpx.AsyncClient(timeout=3.0) as c:
                r = await c.get(SGLANG_BASE + "/v1/models")
                d = r.json().get("data", [])
                if d:
                    sglang_model = d[0]["id"]
        except Exception:
            pass
    return {
        "sglang":  {"ok": sglang_ok,  "model": sglang_model},
        "ollama":  {"ok": ollama_ok},
        "litellm": {"ok": litellm_ok},
    }

# ── Ollama ────────────────────────────────────────────────────────────────────

@app.get("/api/ollama/models")
async def list_ollama_models():
    try:
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.get(OLLAMA_BASE + "/api/tags")
            return r.json()
    except Exception as e:
        raise HTTPException(502, f"Ollama unreachable: {e}")


@app.post("/api/ollama/pull")
async def pull_ollama_model(req: PullRequest):
    async def stream() -> AsyncGenerator[str, None]:
        try:
            async with httpx.AsyncClient(timeout=None) as c:
                async with c.stream(
                    "POST", OLLAMA_BASE + "/api/pull",
                    json={"name": req.name, "stream": True},
                ) as resp:
                    async for line in resp.aiter_lines():
                        if line:
                            yield f"data: {line}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield 'data: {"done":true}\n\n'

    return StreamingResponse(
        stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.delete("/api/ollama/models/{name:path}")
async def delete_ollama_model(name: str):
    try:
        async with httpx.AsyncClient(timeout=60.0) as c:
            r = await c.delete(OLLAMA_BASE + "/api/delete", json={"name": name})
            if r.status_code == 404:
                raise HTTPException(404, f"Model '{name}' not found in Ollama")
            if r.status_code not in (200, 204):
                try:
                    detail = r.json()
                    msg = detail.get("error", r.text)
                except Exception:
                    msg = r.text
                raise HTTPException(r.status_code, msg)
        return {"ok": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, str(e))

# ── LiteLLM ───────────────────────────────────────────────────────────────────

@app.get("/api/litellm/models")
async def list_litellm_models():
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(
                LITELLM_BASE + "/v1/models",
                headers={"Authorization": "Bearer sk-placeholder"},
            )
            return r.json()
    except Exception as e:
        raise HTTPException(502, str(e))


@app.get("/api/litellm/config")
async def get_litellm_config():
    if not LITELLM_CONFIG.exists():
        return {"model_list": [], "_raw": "# config file not found at: " + str(LITELLM_CONFIG)}
    raw = LITELLM_CONFIG.read_text()
    cfg = yaml.safe_load(raw) or {}
    cfg["_raw"] = raw
    return cfg


@app.post("/api/litellm/apply-wildcard")
async def apply_litellm_wildcard():
    cfg = load_litellm_config()
    model_list = cfg.get("model_list", [])

    if any(m.get("model_name") == "ollama/*" for m in model_list):
        return {"ok": True, "message": "Wildcard already present"}

    model_list = [
        m for m in model_list
        if not str(m.get("litellm_params", {}).get("model", "")).startswith("ollama/")
    ]
    model_list.append({
        "model_name": "ollama/*",
        "litellm_params": {
            "model":    "ollama/*",
            "api_base": OLLAMA_BASE,
        },
    })
    cfg["model_list"] = model_list
    save_litellm_config(cfg)

    result = subprocess.run(
        ["sudo", "systemctl", "restart", "litellm"],
        capture_output=True, text=True, timeout=15,
    )
    if result.returncode != 0:
        return {"ok": True, "message": "Config saved. Restart LiteLLM manually: sudo systemctl restart litellm", "warning": result.stderr}
    return {"ok": True, "message": "Wildcard applied — LiteLLM restarted"}


@app.post("/api/litellm/restart")
async def restart_litellm():
    result = subprocess.run(
        ["sudo", "systemctl", "restart", "litellm"],
        capture_output=True, text=True, timeout=15,
    )
    if result.returncode != 0:
        raise HTTPException(500, result.stderr)
    return {"ok": True}

# ── SGLang ────────────────────────────────────────────────────────────────────

@app.get("/api/sglang/profiles")
async def get_sglang_profiles():
    return load_profiles()


@app.get("/api/sglang/status")
async def sglang_status():
    result = subprocess.run(
        ["docker", "ps", "--filter", "name=sglang", "--format", "{{.Names}}\t{{.Status}}"],
        capture_output=True, text=True, timeout=5,
    )
    running = bool(result.stdout.strip())
    model = None
    if running:
        try:
            async with httpx.AsyncClient(timeout=3.0) as c:
                r = await c.get(SGLANG_BASE + "/v1/models")
                d = r.json().get("data", [])
                if d:
                    model = d[0]["id"]
        except Exception:
            pass
    return {"running": running, "model": model, "container_info": result.stdout.strip()}


@app.post("/api/sglang/stop")
async def stop_sglang():
    result = subprocess.run(
        ["docker", "stop", "sglang"],
        capture_output=True, text=True, timeout=60,
    )
    ok = result.returncode == 0
    if not ok:
        result = subprocess.run(
            ["sudo", "docker", "stop", "sglang"],
            capture_output=True, text=True, timeout=60,
        )
        ok = result.returncode == 0
    return {"ok": ok, "output": (result.stdout + result.stderr).strip()}


@app.post("/api/sglang/start")
async def start_sglang(req: SGLangStartRequest):
    profiles = load_profiles()
    profile = next((p for p in profiles if p["id"] == req.profile), None)
    if not profile:
        raise HTTPException(404, f"Profile '{req.profile}' not found")
    script = Path(profile.get("script", "")).expanduser()
    if not script.exists():
        raise HTTPException(400, f"Script not found: {script}")

    log_path = f"/tmp/sglang_{req.profile}.log"
    with open(log_path, "w") as logf:
        subprocess.Popen(
            ["bash", str(script)],
            stdout=logf, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return {"ok": True, "message": f"Launched {profile['name']} — logs at {log_path}"}

# ── HuggingFace ───────────────────────────────────────────────────────────────

@app.post("/api/hf/download")
async def hf_download(req: HFDownloadRequest):
    safe_repo = req.repo_id.replace("'", "").replace('"', "")
    local_dir = req.local_dir or str(HF_CACHE / safe_repo.replace("/", "--"))

    script = f"""
import sys, json
sys.stdout.reconfigure(line_buffering=True)
from huggingface_hub import snapshot_download
print(json.dumps({{"status": "starting", "repo": "{safe_repo}"}}), flush=True)
try:
    path = snapshot_download(repo_id="{safe_repo}", local_dir="{local_dir}")
    print(json.dumps({{"status": "complete", "path": path}}), flush=True)
except Exception as e:
    print(json.dumps({{"status": "error", "error": str(e)}}), flush=True)
"""

    async def stream() -> AsyncGenerator[str, None]:
        try:
            proc = await asyncio.create_subprocess_exec(
                "python3", "-c", script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            assert proc.stdout
            async for raw in proc.stdout:
                line = raw.decode().strip()
                if line:
                    yield f"data: {line}\n\n"
            stderr_data = await proc.stderr.read()  # type: ignore[union-attr]
            for line in stderr_data.decode().split("\n"):
                s = line.strip()
                if s and "%" not in s and "it/s" not in s:
                    yield f"data: {json.dumps({'log': s})}\n\n"
            await proc.wait()
        except Exception as e:
            yield f"data: {json.dumps({'status': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

# ─── Frontend ─────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DGX Model Manager</title>
<link rel="icon" type="image/png" href="/favicon.png">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root {
  --bg:       #08080c;
  --s1:       #0f0f14;
  --s2:       #16161e;
  --s3:       #1e1e28;
  --border:   #252535;
  --border2:  #30304a;
  --text:     #d4d4e8;
  --muted:    #6a6a90;
  --amber:    #f0a034;
  --amber2:   #c07020;
  --green:    #3dba78;
  --red:      #e05050;
  --blue:     #5a9af5;
  --mono:     'IBM Plex Mono', monospace;
  --sans:     'Space Grotesk', sans-serif;
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%}
body{
  background:var(--bg);color:var(--text);
  font-family:var(--sans);font-size:14px;
  line-height:1.5;display:flex;flex-direction:column;
  height:100vh;overflow:hidden;
}
body::before{
  content:'';position:fixed;inset:0;
  background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,.08) 2px,rgba(0,0,0,.08) 4px);
  pointer-events:none;z-index:1000;opacity:.4;
}
.header{
  display:flex;align-items:center;gap:20px;
  padding:0 20px;height:52px;
  border-bottom:1px solid var(--border);
  background:var(--s1);flex-shrink:0;
  position:relative;z-index:10;
}
.hdr-logo{display:flex;align-items:center;gap:10px}
.hdr-sigil{
  width:28px;height:28px;background:var(--amber);
  clip-path:polygon(50% 0%,100% 25%,100% 75%,50% 100%,0% 75%,0% 25%);
  display:flex;align-items:center;justify-content:center;
  font-size:12px;font-weight:700;color:#000;
  font-family:var(--mono);flex-shrink:0;
}
.hdr-name{font-family:var(--mono);font-size:12px;font-weight:600;letter-spacing:.12em;color:var(--amber);text-transform:uppercase}
.hdr-node{font-family:var(--mono);font-size:10px;color:var(--muted);letter-spacing:.06em}
.hdr-sep{flex:1}
.status-cluster{display:flex;gap:6px;align-items:center}
.pill{
  display:flex;align-items:center;gap:5px;
  padding:3px 10px 3px 7px;border-radius:20px;
  border:1px solid var(--border);background:var(--s2);
  font-family:var(--mono);font-size:10px;color:var(--muted);
  transition:border-color .2s,color .2s;white-space:nowrap;
}
.pill.ok{border-color:#1e3a28;color:#8dd4a8}
.pill.err{border-color:#3a1818;color:#e08888}
.dot{width:5px;height:5px;border-radius:50%;background:var(--muted);transition:background .3s,box-shadow .3s}
.pill.ok .dot{background:var(--green);box-shadow:0 0 5px var(--green)}
.pill.err .dot{background:var(--red)}
.refresh-btn,.help-btn{
  height:26px;border-radius:6px;border:1px solid var(--border);
  background:var(--s2);color:var(--muted);cursor:pointer;
  font-size:11px;display:flex;align-items:center;
  justify-content:center;transition:all .15s;
  font-family:var(--mono);padding:0 10px;text-decoration:none;
}
.refresh-btn{width:26px;padding:0}
.refresh-btn:hover,.help-btn:hover{border-color:var(--amber);color:var(--amber)}
.body-wrap{display:flex;flex:1;overflow:hidden}
.sidebar{
  width:192px;flex-shrink:0;
  border-right:1px solid var(--border);background:var(--s1);
  display:flex;flex-direction:column;padding:12px 0;overflow-y:auto;
}
.nav-section-label{
  font-family:var(--mono);font-size:9px;letter-spacing:.14em;
  text-transform:uppercase;color:var(--muted);
  padding:12px 16px 6px;opacity:.6;
}
.nav-item{
  display:flex;align-items:center;gap:10px;
  padding:8px 16px;font-size:13px;font-weight:500;
  color:var(--muted);cursor:pointer;
  border-left:2px solid transparent;transition:all .12s;user-select:none;
}
.nav-item:hover{color:var(--text);background:var(--s2)}
.nav-item.active{color:var(--amber);border-left-color:var(--amber);background:linear-gradient(90deg,rgba(240,160,52,.07),transparent)}
.nav-icon{font-size:14px;width:16px;text-align:center;flex-shrink:0}
.nav-badge{
  margin-left:auto;background:var(--s3);border:1px solid var(--border);
  border-radius:10px;padding:1px 7px;
  font-family:var(--mono);font-size:10px;color:var(--muted);
}
.main{flex:1;overflow-y:auto;padding:24px}
.tab{display:none}
.tab.active{display:block;animation:fadein .15s ease}
@keyframes fadein{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}
.page-hdr{margin-bottom:20px}
.page-title{font-size:18px;font-weight:700;letter-spacing:-.01em}
.page-sub{font-size:12px;color:var(--muted);margin-top:3px;line-height:1.6}
.page-sub code{font-family:var(--mono);color:var(--amber);font-size:11px}
.sec-label{
  font-family:var(--mono);font-size:9px;letter-spacing:.14em;
  text-transform:uppercase;color:var(--muted);
  margin:20px 0 10px;display:flex;align-items:center;gap:10px;
}
.sec-label::after{content:'';flex:1;height:1px;background:var(--border)}
.card{background:var(--s1);border:1px solid var(--border);border-radius:8px;padding:16px 18px;margin-bottom:10px}
.card-row{display:flex;align-items:flex-start;gap:12px}
.card-icon{
  width:32px;height:32px;flex-shrink:0;
  background:var(--s2);border:1px solid var(--border);
  border-radius:7px;display:flex;align-items:center;justify-content:center;font-size:14px;
}
.card-info{flex:1;min-width:0}
.card-name{font-size:13px;font-weight:600}
.card-meta{font-size:11px;color:var(--muted);font-family:var(--mono);margin-top:2px}
.card-actions{margin-left:auto;display:flex;gap:6px;align-items:center;flex-shrink:0}
.card-desc{font-size:12px;color:var(--muted);line-height:1.6;margin-top:10px}
.card-desc code{font-family:var(--mono);color:var(--amber);font-size:11px}
.model-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:8px;margin-bottom:8px}
.model-card{
  background:var(--s1);border:1px solid var(--border);
  border-radius:8px;padding:12px 14px;
  display:flex;align-items:center;gap:10px;transition:border-color .15s;
}
.model-card:hover{border-color:var(--border2)}
.model-card-info{flex:1;min-width:0}
.model-card-name{font-family:var(--mono);font-size:12px;font-weight:500;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.model-card-meta{font-size:11px;color:var(--muted);margin-top:2px}
.model-card-right{display:flex;align-items:center;gap:6px;flex-shrink:0}
.tag{display:inline-block;padding:2px 7px;border-radius:4px;font-size:9px;font-family:var(--mono);font-weight:600;letter-spacing:.06em;text-transform:uppercase}
.tag-ollama{background:#0e2018;color:#5cc480;border:1px solid #1a3a28}
.tag-sglang{background:#101828;color:#6898e8;border:1px solid #1a2a40}
.btn{
  display:inline-flex;align-items:center;gap:6px;
  padding:7px 14px;border-radius:6px;font-size:12px;font-weight:600;
  font-family:var(--sans);cursor:pointer;border:1px solid var(--border);
  background:var(--s2);color:var(--text);transition:all .12s;white-space:nowrap;line-height:1;
}
.btn:hover{border-color:var(--border2);background:var(--s3)}
.btn:active{transform:scale(.97)}
.btn:disabled{opacity:.35;cursor:not-allowed;pointer-events:none}
.btn-primary{background:var(--amber);color:#000;border-color:var(--amber)}
.btn-primary:hover{background:var(--amber2);border-color:var(--amber2);color:#000}
.btn-danger{background:#180808;color:#e08888;border-color:#2a1010}
.btn-danger:hover{border-color:var(--red);color:var(--red)}
.btn-sm{padding:4px 10px;font-size:11px}
.btn-ghost{background:transparent;border-color:transparent;color:var(--muted)}
.btn-ghost:hover{color:var(--text);background:var(--s2);border-color:var(--border)}
.input-row{display:flex;gap:8px;align-items:stretch;margin-bottom:14px}
.input{
  flex:1;background:var(--s2);border:1px solid var(--border);
  border-radius:6px;padding:8px 12px;color:var(--text);
  font-size:13px;font-family:var(--mono);outline:none;transition:border-color .15s;
}
.input:focus{border-color:var(--amber)}
.input::placeholder{color:var(--muted)}
.progress-wrap{margin-top:10px;display:none}
.progress-wrap.show{display:block}
.prog-bar-outer{height:3px;background:var(--s3);border-radius:2px;overflow:hidden;margin-bottom:8px}
.prog-bar{height:100%;background:var(--amber);border-radius:2px;transition:width .3s;width:0}
.prog-bar.spin{width:35%!important;animation:pgslide 1.2s ease-in-out infinite}
@keyframes pgslide{0%{transform:translateX(-200%)}100%{transform:translateX(500%)}}
.prog-log{
  font-family:var(--mono);font-size:11px;color:var(--muted);
  background:#04040a;border:1px solid var(--border);
  border-radius:5px;padding:8px 10px;
  max-height:110px;overflow-y:auto;line-height:1.7;white-space:pre-wrap;
}
.engine-card{
  background:var(--s1);border:1px solid var(--border);
  border-radius:10px;padding:20px 22px;margin-bottom:14px;
  position:relative;overflow:hidden;
}
.engine-card::before{
  content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,transparent,var(--amber),transparent);
  opacity:0;transition:opacity .3s;
}
.engine-card.online::before{opacity:1}
.engine-status-row{display:flex;align-items:center;gap:14px;margin-bottom:12px}
.engine-led{width:10px;height:10px;border-radius:50%;background:var(--red);flex-shrink:0;transition:background .3s,box-shadow .3s}
.engine-led.on{background:var(--green);box-shadow:0 0 8px var(--green)}
.engine-title{font-size:15px;font-weight:700}
.engine-model{font-size:11px;color:var(--amber);font-family:var(--mono);margin-top:2px}
.engine-footer{font-size:11px;color:var(--muted);font-family:var(--mono)}
.engine-actions{margin-left:auto;display:flex;gap:6px}
.profile-list{display:flex;flex-direction:column;gap:6px}
.profile-item{
  display:flex;align-items:center;gap:12px;padding:12px 14px;
  background:var(--s1);border:1px solid var(--border);
  border-radius:8px;cursor:pointer;transition:border-color .15s;
}
.profile-item:hover{border-color:var(--border2)}
.profile-item.selected{border-color:var(--amber);background:linear-gradient(90deg,rgba(240,160,52,.05),transparent)}
.p-radio{width:14px;height:14px;border-radius:50%;border:2px solid var(--border);flex-shrink:0;transition:all .15s}
.profile-item.selected .p-radio{border-color:var(--amber);background:var(--amber);box-shadow:0 0 6px var(--amber)}
.p-info{flex:1;min-width:0}
.p-name{font-size:13px;font-weight:600}
.p-desc{font-size:11px;color:var(--muted);margin-top:2px}
.p-vram{font-family:var(--mono);font-size:11px;color:var(--amber);flex-shrink:0}
.config-block{
  font-family:var(--mono);font-size:11px;
  background:#04040a;border:1px solid var(--border);
  border-radius:6px;padding:14px;overflow:auto;
  max-height:280px;color:#a0a0c0;line-height:1.8;white-space:pre;
}
.wc-active{display:flex;align-items:center;gap:7px;font-size:12px;color:#5cc480;margin-top:8px}
.wc-inactive{font-size:12px;color:var(--muted);margin-top:8px}
.empty{text-align:center;padding:40px 20px;color:var(--muted)}
.empty-icon{font-size:28px;margin-bottom:10px;opacity:.5}
.empty-text{font-size:13px}
.spin-icon{
  width:13px;height:13px;border:2px solid rgba(255,255,255,.15);
  border-top-color:currentColor;border-radius:50%;
  animation:spin .6s linear infinite;flex-shrink:0;
}
@keyframes spin{to{transform:rotate(360deg)}}
#toast-root{position:fixed;bottom:20px;right:20px;display:flex;flex-direction:column;gap:6px;z-index:9999;pointer-events:none}
.toast{
  background:var(--s2);border:1px solid var(--border);
  border-radius:8px;padding:10px 14px;font-size:13px;
  max-width:320px;pointer-events:auto;animation:toast-in .2s ease;
}
@keyframes toast-in{from{transform:translateX(100%);opacity:0}to{opacity:1;transform:none}}
.toast.ok{border-color:#1e3a28}
.toast.err{border-color:#3a1818}
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
</style>
</head>
<body>

<header class="header">
  <div class="hdr-logo">
    <div class="hdr-sigil">M</div>
    <div>
      <div class="hdr-name" id="hdr-name">Model Manager</div>
      <div class="hdr-node" id="hdr-node">DGX Spark · :8090</div>
    </div>
  </div>
  <div class="hdr-sep"></div>
  <div class="status-cluster">
    <div class="pill" id="pill-sglang"><div class="dot"></div><span>SGLang</span></div>
    <div class="pill" id="pill-ollama"><div class="dot"></div><span>Ollama</span></div>
    <div class="pill" id="pill-litellm"><div class="dot"></div><span>LiteLLM</span></div>
    <button class="refresh-btn" onclick="pollStatus()" title="Refresh">↻</button>
    <a href="/help" target="_blank" class="help-btn">? Help</a>
  </div>
</header>

<div class="body-wrap">
  <nav class="sidebar">
    <div class="nav-section-label">Models</div>
    <div class="nav-item active" id="nav-ollama" onclick="switchTab('ollama')">
      <span class="nav-icon">🦙</span>Ollama
      <span class="nav-badge" id="badge-ollama">—</span>
    </div>
    <div class="nav-item" id="nav-hf" onclick="switchTab('hf')">
      <span class="nav-icon">🤗</span>HF Download
    </div>
    <div class="nav-section-label">Routing</div>
    <div class="nav-item" id="nav-litellm" onclick="switchTab('litellm')">
      <span class="nav-icon">⚡</span>LiteLLM
      <span class="nav-badge" id="badge-litellm">—</span>
    </div>
    <div class="nav-section-label">Engine</div>
    <div class="nav-item" id="nav-sglang" onclick="switchTab('sglang')">
      <span class="nav-icon">🚀</span>SGLang
    </div>
  </nav>

  <main class="main">

    <!-- OLLAMA -->
    <div class="tab active" id="tab-ollama">
      <div class="page-hdr">
        <div class="page-title">Ollama Models</div>
        <div class="page-sub">Pull models from the Ollama library. With wildcard routing enabled, every pulled model is instantly available at <code id="litellm-port">:4000</code>.</div>
      </div>
      <div class="input-row">
        <input class="input" id="pull-input"
          placeholder="Model name — e.g. llama3.2  qwen2.5:7b  phi4  gemma3:4b  qwen3-coder-next"
          onkeydown="if(event.key==='Enter')pullModel()">
        <button class="btn btn-primary" id="pull-btn" onclick="pullModel()">⬇ Pull</button>
      </div>
      <div class="progress-wrap" id="pull-progress">
        <div class="prog-bar-outer"><div class="prog-bar spin" id="pull-bar"></div></div>
        <div class="prog-log" id="pull-log"></div>
      </div>
      <div class="sec-label">Installed <span id="badge-ollama-inline"></span></div>
      <div id="ollama-list"><div class="empty"><div class="spin-icon" style="margin:0 auto 8px"></div></div></div>
    </div>

    <!-- HF DOWNLOAD -->
    <div class="tab" id="tab-hf">
      <div class="page-hdr">
        <div class="page-title">HuggingFace Download</div>
        <div class="page-sub">Download any model from HuggingFace Hub directly to the device. Large models land in <code>~/.cache/huggingface/hub/</code> — ready for SGLang or vLLM.</div>
      </div>
      <div class="card">
        <div class="sec-label" style="margin-top:0;margin-bottom:8px">Repository ID</div>
        <div class="input-row">
          <input class="input" id="hf-repo" placeholder="e.g. mistralai/Mistral-7B-Instruct-v0.3">
        </div>
        <div class="sec-label" style="margin-bottom:8px">Local Directory <span style="color:var(--muted);font-size:10px">(optional — leave blank for HF cache default)</span></div>
        <div class="input-row" style="margin-bottom:0">
          <input class="input" id="hf-dir" placeholder="/home/user/models/my-model">
          <button class="btn btn-primary" id="hf-btn" onclick="hfDownload()">⬇ Download</button>
        </div>
      </div>
      <div class="progress-wrap" id="hf-progress">
        <div class="prog-bar-outer"><div class="prog-bar spin"></div></div>
        <div class="prog-log" id="hf-log"></div>
      </div>
    </div>

    <!-- LITELLM -->
    <div class="tab" id="tab-litellm">
      <div class="page-hdr">
        <div class="page-title">LiteLLM Routing</div>
        <div class="page-sub">Unified gateway. All apps connect here. This config controls which models they can see.</div>
      </div>
      <div class="card">
        <div class="card-row">
          <div class="card-icon">🃏</div>
          <div class="card-info">
            <div class="card-name">Ollama Wildcard Routing</div>
            <div class="card-meta">ollama/* → auto-expose all Ollama models</div>
          </div>
          <div class="card-actions">
            <button class="btn btn-primary" id="wc-btn" onclick="applyWildcard()">Apply Wildcard</button>
          </div>
        </div>
        <div class="card-desc">
          Adds a single <code>ollama/*</code> entry to your config. After this, any model you pull into Ollama is automatically available — no config edits, no restarts.
        </div>
        <div id="wc-status"></div>
      </div>
      <div class="sec-label">Active Routes <span class="nav-badge" id="litellm-route-count">—</span></div>
      <div id="litellm-list"><div class="empty"><div class="spin-icon" style="margin:0 auto"></div></div></div>
      <div class="sec-label">Config File</div>
      <div class="card">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
          <span style="font-family:var(--mono);font-size:11px;color:var(--muted)" id="config-path-label">~/litellm/litellm_config.yaml</span>
          <div style="display:flex;gap:6px">
            <button class="btn btn-sm btn-ghost" onclick="loadLiteLLMConfig()">↻ Refresh</button>
            <button class="btn btn-sm" onclick="restartLiteLLM()">⟳ Restart</button>
          </div>
        </div>
        <div class="config-block" id="config-block">Loading…</div>
      </div>
    </div>

    <!-- SGLANG -->
    <div class="tab" id="tab-sglang">
      <div class="page-hdr">
        <div class="page-title">SGLang Engine</div>
        <div class="page-sub">SGLang loads one large model at a time. Select a profile and start — status updates automatically.</div>
      </div>
      <div class="engine-card" id="engine-card">
        <div class="engine-status-row">
          <div class="engine-led" id="engine-led"></div>
          <div>
            <div class="engine-title" id="engine-title">Checking…</div>
            <div class="engine-model" id="engine-model"></div>
          </div>
          <div class="engine-actions">
            <button class="btn btn-danger" id="stop-btn" onclick="stopSGLang()" disabled>■ Stop</button>
          </div>
        </div>
        <div class="engine-footer" id="engine-footer">Port :30000 · Docker</div>
      </div>
      <div class="sec-label">Profiles</div>
      <div class="profile-list" id="profile-list">
        <div class="empty"><div class="spin-icon" style="margin:0 auto"></div></div>
      </div>
      <div style="display:flex;align-items:center;gap:12px;margin-top:14px">
        <button class="btn btn-primary" id="start-btn" onclick="startSGLang()">▶ Start Selected</button>
        <span style="font-size:12px;color:var(--muted)">Runs start script in background · check status pill</span>
      </div>
      <div class="progress-wrap" id="sglang-progress" style="margin-top:14px">
        <div class="prog-bar-outer"><div class="prog-bar spin"></div></div>
        <div class="prog-log" id="sglang-log"></div>
      </div>
    </div>

  </main>
</div>

<div id="toast-root"></div>

<script>
let activeTab = 'ollama';
let selectedProfile = null;
let appConfig = {};

document.addEventListener('DOMContentLoaded', async () => {
  await loadAppConfig();
  pollStatus();
  loadOllamaModels();
  setInterval(pollStatus, 12000);
});

async function loadAppConfig() {
  try {
    appConfig = await apiFetch('/api/config');
    document.getElementById('hdr-name').textContent = appConfig.display_name || 'Model Manager';
    document.getElementById('hdr-node').textContent = (appConfig.display_name || 'DGX Spark') + ' · :' + (appConfig.port || 8090);
    const port = appConfig.litellm_base ? appConfig.litellm_base.split(':').pop() : '4000';
    document.getElementById('litellm-port').textContent = ':' + port;
  } catch(e) {}
}

function switchTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  document.getElementById('nav-' + name).classList.add('active');
  activeTab = name;
  if (name === 'litellm') { loadLiteLLMModels(); loadLiteLLMConfig(); checkWildcard(); }
  if (name === 'sglang')  { loadSGLangStatus(); loadProfiles(); }
  if (name === 'ollama')  { loadOllamaModels(); }
}

async function pollStatus() {
  try {
    const d = await apiFetch('/api/status');
    setPill('pill-sglang', d.sglang.ok, d.sglang.model ? 'SGLang · ' + d.sglang.model.split('/').pop().slice(0,18) : 'SGLang');
    setPill('pill-ollama', d.ollama.ok, 'Ollama');
    setPill('pill-litellm', d.litellm.ok, 'LiteLLM');
  } catch(e) {}
}

function setPill(id, ok, label) {
  const el = document.getElementById(id);
  el.className = 'pill ' + (ok ? 'ok' : 'err');
  el.querySelector('span').textContent = label;
}

async function loadOllamaModels() {
  const el = document.getElementById('ollama-list');
  try {
    const d = await apiFetch('/api/ollama/models');
    const models = d.models || [];
    document.getElementById('badge-ollama').textContent = models.length;
    document.getElementById('badge-ollama-inline').textContent = models.length + ' model' + (models.length !== 1 ? 's' : '');
    if (!models.length) {
      el.innerHTML = '<div class="empty"><div class="empty-icon">🦙</div><div class="empty-text">No models installed. Pull one above.</div></div>';
      return;
    }
    el.innerHTML = '<div class="model-grid">' + models.map(m => {
      const gb = m.size ? (m.size / 1e9).toFixed(1) + ' GB' : '?';
      const date = m.modified_at ? new Date(m.modified_at).toLocaleDateString() : '';
      const safe = m.name.replace(/'/g, "\\'");
      return `<div class="model-card">
        <div class="model-card-info">
          <div class="model-card-name">${m.name}</div>
          <div class="model-card-meta">${gb}${date ? ' · ' + date : ''}</div>
        </div>
        <div class="model-card-right">
          <span class="tag tag-ollama">ollama</span>
          <button class="btn btn-sm btn-danger" onclick="deleteModel('${safe}',this)">✕</button>
        </div>
      </div>`;
    }).join('') + '</div>';
  } catch(e) {
    el.innerHTML = '<div class="empty"><div class="empty-icon">⚠</div><div class="empty-text">Ollama unreachable</div></div>';
  }
}

async function pullModel() {
  const input = document.getElementById('pull-input');
  const name = input.value.trim();
  if (!name) { input.focus(); return; }
  const btn = document.getElementById('pull-btn');
  const prog = document.getElementById('pull-progress');
  const bar  = document.getElementById('pull-bar');
  const log  = document.getElementById('pull-log');
  btn.disabled = true;
  btn.innerHTML = '<div class="spin-icon"></div> Pulling…';
  prog.classList.add('show'); log.textContent = 'Connecting…'; bar.className = 'prog-bar spin';
  try {
    const resp = await fetch('/api/ollama/pull', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name})});
    const reader = resp.body.getReader(); const dec = new TextDecoder();
    while (true) {
      const {done, value} = await reader.read(); if (done) break;
      for (const line of dec.decode(value).split('\n')) {
        if (!line.startsWith('data: ')) continue;
        try {
          const ev = JSON.parse(line.slice(6));
          if (ev.done) break;
          if (ev.error) { toast('Error: '+ev.error,'err'); break; }
          if (ev.total && ev.completed) {
            const pct = Math.round(ev.completed/ev.total*100);
            bar.className = 'prog-bar'; bar.style.width = pct+'%';
            log.textContent = (ev.status||'')+' — '+pct+'% ('+( ev.completed/1e6).toFixed(0)+' / '+(ev.total/1e6).toFixed(0)+' MB)';
          } else if (ev.status) { log.textContent = ev.status; }
        } catch(e) {}
      }
    }
    bar.className = 'prog-bar'; bar.style.width = '100%';
    toast('✓ '+name+' ready','ok'); input.value = '';
    await loadOllamaModels();
    setTimeout(() => { prog.classList.remove('show'); bar.style.width='0'; }, 2000);
  } catch(e) { toast('Pull failed: '+e.message,'err'); }
  finally { btn.disabled=false; btn.innerHTML='⬇ Pull'; }
}

async function deleteModel(name, btn) {
  if (!confirm('Delete '+name+'?\nThis cannot be undone.')) return;
  btn.disabled=true; btn.innerHTML='…';
  try {
    const r = await fetch('/api/ollama/models/'+encodeURIComponent(name),{method:'DELETE'});
    if (!r.ok) {
      let msg = r.statusText;
      try { const d = await r.json(); msg = d.detail || JSON.stringify(d); } catch(e) { msg = await r.text(); }
      throw new Error(msg);
    }
    toast('✓ Deleted '+name,'ok'); loadOllamaModels();
  } catch(e) { toast('Delete failed: '+e.message,'err'); btn.disabled=false; btn.innerHTML='✕'; }
}

async function loadLiteLLMModels() {
  const el = document.getElementById('litellm-list');
  try {
    const d = await apiFetch('/api/litellm/models');
    const models = d.data || [];
    document.getElementById('badge-litellm').textContent = models.length;
    document.getElementById('litellm-route-count').textContent = models.length;
    if (!models.length) { el.innerHTML='<div class="empty"><div class="empty-text">No routes active</div></div>'; return; }
    el.innerHTML = '<div class="model-grid">'+models.map(m => {
      const isOllama = m.id.toLowerCase().includes('ollama') || m.id.includes(':');
      return `<div class="model-card"><div class="model-card-info"><div class="model-card-name">${m.id}</div></div><span class="tag ${isOllama?'tag-ollama':'tag-sglang'}">${isOllama?'ollama':'sglang'}</span></div>`;
    }).join('')+'</div>';
  } catch(e) { el.innerHTML='<div class="empty"><div class="empty-text">LiteLLM unreachable</div></div>'; }
}

async function loadLiteLLMConfig() {
  const el = document.getElementById('config-block');
  try {
    const d = await apiFetch('/api/litellm/config');
    el.textContent = d._raw || JSON.stringify(d, null, 2);
  } catch(e) { el.textContent='Could not load config'; }
}

async function checkWildcard() {
  try {
    const d = await apiFetch('/api/litellm/config');
    const has = (d.model_list||[]).some(m => m.model_name==='ollama/*');
    const status = document.getElementById('wc-status');
    const btn = document.getElementById('wc-btn');
    if (has) {
      status.innerHTML='<div class="wc-active">✓ Wildcard active — all Ollama models auto-exposed</div>';
      btn.textContent='✓ Applied'; btn.disabled=true;
    } else {
      status.innerHTML='<div class="wc-inactive">Not applied — each Ollama model requires a manual config entry</div>';
      btn.textContent='Apply Wildcard'; btn.disabled=false;
    }
  } catch(e) {}
}

async function applyWildcard() {
  const btn = document.getElementById('wc-btn');
  btn.disabled=true; btn.innerHTML='<div class="spin-icon"></div> Applying…';
  try {
    const d = await apiFetch('/api/litellm/apply-wildcard','POST');
    toast(d.message||'✓ Wildcard applied','ok');
    if (d.warning) toast('Note: '+d.warning, null);
    await checkWildcard(); await loadLiteLLMConfig(); setTimeout(loadLiteLLMModels,3500);
  } catch(e) { toast('Failed: '+e.message,'err'); btn.disabled=false; btn.textContent='Apply Wildcard'; }
}

async function restartLiteLLM() {
  toast('Restarting LiteLLM…',null);
  try { await apiFetch('/api/litellm/restart','POST'); toast('✓ LiteLLM restarted','ok'); setTimeout(loadLiteLLMModels,3500); }
  catch(e) { toast('Failed: '+e.message,'err'); }
}

async function loadSGLangStatus() {
  try {
    const d = await apiFetch('/api/sglang/status');
    const led=document.getElementById('engine-led'), title=document.getElementById('engine-title');
    const model=document.getElementById('engine-model'), card=document.getElementById('engine-card');
    const stop=document.getElementById('stop-btn');
    if (d.running) {
      led.className='engine-led on'; title.textContent='SGLang — Running';
      model.textContent=d.model||'Model loading…'; card.classList.add('online'); stop.disabled=false;
    } else {
      led.className='engine-led'; title.textContent='SGLang — Stopped';
      model.textContent=''; card.classList.remove('online'); stop.disabled=true;
    }
  } catch(e) {}
}

async function loadProfiles() {
  const el = document.getElementById('profile-list');
  try {
    const profiles = await apiFetch('/api/sglang/profiles');
    if (!profiles.length) { el.innerHTML='<div class="empty"><div class="empty-text">No profiles defined in sglang_profiles.json</div></div>'; return; }
    if (!selectedProfile) selectedProfile = profiles[0].id;
    el.innerHTML = profiles.map(p => `
      <div class="profile-item ${selectedProfile===p.id?'selected':''}" onclick="selectProfile('${p.id}',this)">
        <div class="p-radio"></div>
        <div class="p-info"><div class="p-name">${p.name}</div><div class="p-desc">${p.description}</div></div>
        <div class="p-vram">${p.vram_gb} GB</div>
      </div>`).join('');
  } catch(e) { el.innerHTML='<div class="empty"><div class="empty-text">Could not load profiles</div></div>'; }
}

function selectProfile(id, el) {
  selectedProfile=id;
  document.querySelectorAll('.profile-item').forEach(p=>p.classList.remove('selected'));
  el.classList.add('selected');
}

async function stopSGLang() {
  if (!confirm('Stop SGLang? This will interrupt any active inference requests.')) return;
  const btn=document.getElementById('stop-btn'); btn.disabled=true; btn.innerHTML='<div class="spin-icon"></div>';
  try {
    const d = await apiFetch('/api/sglang/stop','POST');
    toast(d.ok?'✓ SGLang stopped':'Stop may have failed: '+d.output, d.ok?'ok':'err');
    setTimeout(()=>{loadSGLangStatus();btn.innerHTML='■ Stop';},1500);
  } catch(e) { toast('Error: '+e.message,'err'); btn.innerHTML='■ Stop'; }
}

async function startSGLang() {
  if (!selectedProfile) { toast('Select a profile first','err'); return; }
  const btn=document.getElementById('start-btn'), prog=document.getElementById('sglang-progress'), log=document.getElementById('sglang-log');
  btn.disabled=true; btn.innerHTML='<div class="spin-icon"></div> Launching…';
  prog.classList.add('show'); log.textContent='Sending start command…';
  try {
    const d = await apiFetch('/api/sglang/start','POST',{profile:selectedProfile});
    toast('✓ SGLang launching — ~5 min warm-up','ok');
    log.textContent=d.message+'\n\nPolling every 20 seconds — ready toast fires when model is loaded…';
    const poll=setInterval(async()=>{
      const status = await apiFetch('/api/sglang/status');
      await loadSGLangStatus();
      if (status.running && status.model) {
        clearInterval(poll);
        toast('✓ SGLang is ready — '+status.model.split('/').pop(),'ok');
        prog.classList.remove('show');
      } else if (status.running && !status.model) {
        log.textContent=d.message+'\n\nContainer running — model still loading…';
      }
    },20000);
  } catch(e) { toast('Start failed: '+e.message,'err'); prog.classList.remove('show'); }
  finally { btn.disabled=false; btn.innerHTML='▶ Start Selected'; }
}

async function hfDownload() {
  const repo=document.getElementById('hf-repo').value.trim(), dir=document.getElementById('hf-dir').value.trim();
  if (!repo) { document.getElementById('hf-repo').focus(); return; }
  const btn=document.getElementById('hf-btn'), prog=document.getElementById('hf-progress'), log=document.getElementById('hf-log');
  btn.disabled=true; btn.innerHTML='<div class="spin-icon"></div> Downloading…';
  prog.classList.add('show'); log.textContent='Starting: '+repo;
  try {
    const resp=await fetch('/api/hf/download',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({repo_id:repo,local_dir:dir||undefined})});
    const reader=resp.body.getReader(); const dec=new TextDecoder();
    while(true){
      const {done,value}=await reader.read(); if(done) break;
      for(const line of dec.decode(value).split('\n')){
        if(!line.startsWith('data: ')) continue;
        try{
          const ev=JSON.parse(line.slice(6));
          if(ev.status==='complete'){toast('✓ Downloaded: '+ev.path,'ok');log.textContent+='\n✓ '+ev.path;}
          else if(ev.status==='error'){toast('Error: '+ev.error,'err');log.textContent+='\n✗ '+ev.error;}
          else if(ev.log){log.textContent+='\n'+ev.log;}
          else if(ev.status){log.textContent+='\n'+ev.status;}
          log.scrollTop=log.scrollHeight;
        }catch(e){}
      }
    }
  } catch(e) { toast('Failed: '+e.message,'err'); }
  finally { btn.disabled=false; btn.innerHTML='⬇ Download'; }
}

async function apiFetch(url, method='GET', body=null) {
  const opts={method,headers:{'Content-Type':'application/json'}};
  if(body) opts.body=JSON.stringify(body);
  const r=await fetch(url,opts);
  if(!r.ok){let msg=r.statusText;try{const d=await r.json();msg=d.detail||JSON.stringify(d);}catch(e){}throw new Error(msg);}
  return r.json();
}

function toast(msg, type) {
  const root=document.getElementById('toast-root');
  const el=document.createElement('div'); el.className='toast '+(type||''); el.textContent=msg;
  root.appendChild(el);
  setTimeout(()=>{el.style.transition='opacity .3s';el.style.opacity='0';setTimeout(()=>el.remove(),320);},3500);
}
</script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(HTML)


@app.get("/favicon.png")
async def favicon():
    favicon_path = Path(__file__).parent / "favicon.png"
    if not favicon_path.exists():
        raise HTTPException(404, "favicon.png not found")
    return FileResponse(favicon_path, media_type="image/png")


@app.get("/help", response_class=HTMLResponse)
async def help_page():
    help_path = Path(__file__).parent / "docs.html"
    if not help_path.exists():
        raise HTTPException(404, "docs.html not found")
    return HTMLResponse(help_path.read_text())


if __name__ == "__main__":
    uvicorn.run(app, host=APP_HOST, port=APP_PORT, log_level="info")
