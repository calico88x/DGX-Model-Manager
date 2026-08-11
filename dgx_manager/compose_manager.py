from __future__ import annotations

import json
import os
import re
import shutil
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .config import APP_ROOT, Config
from .system import run_cmd


def slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-._").lower()
    return value[:64] or "deployment"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _port_is_available(bind_host: str, port: int) -> bool:
    """Return True when the requested TCP host port can currently be bound."""
    probe_host = "0.0.0.0" if bind_host == "0.0.0.0" else "127.0.0.1"

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((probe_host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


class ComposeManager:
    def __init__(self, config: Config):
        self.config = config
        self.root = config.path_value("paths.compose_root")
        self.stacks = self.root / "stacks"
        self.archive = self.root / "archive"
        self.stacks.mkdir(parents=True, exist_ok=True)
        self.archive.mkdir(parents=True, exist_ok=True)
        catalog_file = APP_ROOT / "engine_catalog.yaml"
        self.catalog = yaml.safe_load(catalog_file.read_text()).get("engines", {})

    def engine(self, key: str) -> dict:
        if key not in self.catalog:
            raise ValueError(f"Unsupported engine: {key}")
        e = dict(self.catalog[key])
        override = self.config.get(f"compose.images.{key}")
        if override is not None:
            e["image"] = override
        return e

    def stack_dir(self, engine: str, slug: str) -> Path:
        return self.stacks / slugify(engine) / slugify(slug)

    def list_deployments(self) -> list[dict]:
        out = []
        if not self.stacks.exists(): return out
        for meta in self.stacks.glob("*/*/deployment.json"):
            try:
                d = json.loads(meta.read_text())
                d["path"] = str(meta.parent)
                d["compose_exists"] = (meta.parent/"compose.yaml").exists()
                out.append(d)
            except Exception:
                pass
        return sorted(out, key=lambda d: d.get("created_at", ""), reverse=True)

    def profiles_for_engine(self, engine: str) -> list[dict]:
        out=[]
        for d in self.list_deployments():
            if d.get("engine") == engine:
                out.append({
                    "id": f"compose:{engine}:{d['slug']}", "name": d.get("name") or d["slug"],
                    "description": d.get("description") or f"Compose deployment · {d.get('model_name','')}",
                    "vram_gb": d.get("estimated_runtime_gb"), "kind": "compose", "slug": d["slug"],
                    "model_name": d.get("model_name"), "node": d.get("node", "local"),
                })
        return out

    def _allocate_host_port(
        self,
        preferred: int,
        bind_host: str,
        *,
        max_attempts: int = 100,
    ) -> tuple[int, bool]:
        """Choose a free host port without colliding with managed deployments."""
        reserved: set[int] = set()

        for deployment in self.list_deployments():
            try:
                reserved.add(int(deployment.get("port")))
            except (TypeError, ValueError):
                pass

        for port in range(preferred, preferred + max_attempts):
            if port in reserved:
                continue
            if _port_is_available(bind_host, port):
                return port, port != preferred

        raise ValueError(
            f"No available host port found in range "
            f"{preferred}-{preferred + max_attempts - 1}"
        )

    def _container_model_path(self, model: dict) -> tuple[list[str], str, str | None]:
        runtime = Path(model.get("runtime_path") or model.get("dir_path") or "").resolve()
        fmt = model.get("format")
        if not runtime.exists():
            raise ValueError("The selected model path is not available on this node")
        if model.get("source") == "hf_cache":
            hub = self.config.path_value("paths.hf_cache")
            hf_root = hub.parent
            try: rel = runtime.relative_to(hf_root)
            except ValueError: rel = None
            if rel is not None:
                cpath = Path("/root/.cache/huggingface") / rel
                mounts = [f"{hf_root}:/root/.cache/huggingface:ro"]
                gguf = None
                if fmt == "gguf":
                    files = list(runtime.rglob("*.gguf")); gguf = str(cpath / files[0].relative_to(runtime)) if files else None
                return mounts, str(cpath), gguf
        mounts = [f"{runtime}:/models/current:ro"]
        gguf = None
        if fmt == "gguf":
            files=list(runtime.rglob("*.gguf")); gguf=str(Path("/models/current")/files[0].relative_to(runtime)) if files else None
        return mounts, "/models/current", gguf

    @staticmethod
    def _estimated_runtime(model: dict, context: int) -> tuple[float, list[str]]:
        size = float(model.get("size_gb") or 0)
        params = float(model.get("params_b") or 0)
        dtype = str(model.get("dtype") or "Unknown").upper()
        # Disk size is the safest baseline for quantized checkpoints. Add runtime and KV/cache headroom.
        weight = size if size > 0 else params * ({"FP4":0.55,"INT4":0.55,"FP8":1.1,"FP16":2.1,"BF16":2.1}.get(dtype,1.5))
        overhead = max(4.0, weight * 0.18)
        kv = max(1.0, (context / 32768.0) * max(params, 8) * 0.035)
        est = round(weight + overhead + kv, 1)
        notes=[f"Checkpoint baseline: {weight:.1f} GB", f"Runtime overhead allowance: {overhead:.1f} GB", f"Context/KV allowance: {kv:.1f} GB"]
        return est, notes

    def generate(self, *, model: dict, engine_key: str, node_metrics: dict, name: str | None = None,
                 context_length: int | None = None, memory_reserve_gb: float | None = None,
                 profile: str = "balanced", bind_host: str | None = None, expose_litellm: bool = True) -> dict:
        e = self.engine(engine_key)
        image = str(e.get("image") or "").strip()
        if not image:
            raise ValueError(f"No default {e['name']} container image is configured. Set compose.images.{engine_key} first.")
        if model.get("source") == "ollama":
            raise ValueError("Ollama models are served by Ollama and do not need a generated engine Compose stack")
        context = int(context_length or self.config.get("compose.default_context_length", 32768))
        context = max(1024, min(context, 1048576))
        reserve = float(memory_reserve_gb if memory_reserve_gb is not None else self.config.get("compose.default_memory_reserve_gb",24))
        total_mem = float(node_metrics.get("memory_total_gb") or 128)
        if profile == "conservative":
            reserve=max(reserve,32); util_cap=0.62
        elif profile == "performance":
            util_cap=0.88
        else:
            util_cap=0.76
        available_budget=max(1.0,total_mem-reserve)
        estimated, notes=self._estimated_runtime(model,context)
        fit_ratio=estimated/available_budget
        status="good" if fit_ratio <= .72 else "tight" if fit_ratio <= .95 else "risk"
        # vLLM/SGLang reserve a fraction of the GPU-visible memory pool for weights + KV cache.
        # On GB10 this is unified system/GPU memory, so derive the fraction from the workload
        # estimate instead of blindly consuming most of the 128 GB pool.
        floor=0.40 if estimated >= 24 else 0.30
        mem_util=min(util_cap,max(floor,(estimated/max(total_mem,1.0))+0.06))
        bind=bind_host or self.config.get("compose.bind_host","127.0.0.1")
        if bind not in {"127.0.0.1","0.0.0.0"}:
            raise ValueError("Bind address must be 127.0.0.1 or 0.0.0.0")

        preferred_port=int(e.get("port",8000))
        container_port=int(e.get("container_port",preferred_port))
        port, port_changed=self._allocate_host_port(preferred_port,bind)
        if port_changed:
            notes.append(
                f"Preferred host port {preferred_port} is unavailable; "
                f"selected {port} while keeping container port {container_port}."
            )
        mounts, model_path, gguf_path=self._container_model_path(model)
        slug=slugify(name or f"{engine_key}-{model.get('name','model')}")
        served_model_name=slugify(model.get("full_name") or model.get("name") or slug)
        env={"HF_HOME":"/root/.cache/huggingface","HF_HUB_DISABLE_TELEMETRY":"1"}
        gpu=True
        command:list[str]=[]
        volumes=list(mounts)
        top_volumes:dict[str,dict]={}
        quant_method=str(model.get("quant_method") or "").strip().lower() or None
        quant_bits=model.get("quant_bits")
        if quant_method:
            notes.append(f"Checkpoint quantization metadata: {quant_method}" + (f" ({quant_bits}-bit)" if quant_bits else ""))
        is_gb10 = bool(node_metrics.get("unified_memory")) or "GB10" in str(node_metrics.get("gpu",{}).get("name", ""))
        if is_gb10 and engine_key in {"vllm", "sglang"}:
            env["TRITON_PTXAS_PATH"]="/usr/local/cuda/bin/ptxas"
            notes.append("GB10/SM121A detected: system ptxas path is supplied to the Triton-based engine as a compatibility hint inherited from Spark launch guidance.")
        if engine_key == "vllm":
            command=["--model",model_path,"--served-model-name",served_model_name,"--host","0.0.0.0","--port",str(container_port),"--max-model-len",str(context),"--gpu-memory-utilization",f"{mem_util:.2f}"]
            if is_gb10:
                max_num_seqs = {"conservative": 1, "balanced": 2, "performance": 4}.get(profile, 2)
                command += ["--max-num-seqs", str(max_num_seqs)]
                notes.append(f"DGX Spark small-batch scheduling: --max-num-seqs {max_num_seqs} for the {profile} profile.")
            volumes.append("vllm-cache:/root/.cache/vllm"); top_volumes["vllm-cache"]={}
            notes.append("Persistent vLLM compile cache enabled.")
            if quant_method:
                notes.append("vLLM is left to auto-detect checkpoint quantization from the model configuration rather than forcing a potentially incompatible CLI override.")
        elif engine_key == "sglang":
            command=["sglang","serve","--model-path",model_path,"--served-model-name",served_model_name,"--host","0.0.0.0","--port",str(container_port),"--context-length",str(context),"--mem-fraction-static",f"{mem_util:.2f}"]
            # SGLang's ModelOpt loader requires an explicit method for pre-quantized
            # FP4/FP8 checkpoints. For AWQ/GPTQ/compressed-tensors, rely on the
            # checkpoint config instead of guessing flags that can vary by release.
            if quant_method in {"modelopt_fp4","modelopt_fp8"}:
                command += ["--quantization",quant_method]
                notes.append(f"SGLang explicit pre-quantized ModelOpt loader enabled: {quant_method}.")
            elif quant_method:
                notes.append("SGLang will use checkpoint-declared quantization metadata; no online quantization override was added.")
        elif engine_key == "llamacpp":
            if not gguf_path: raise ValueError("llama.cpp generation requires a GGUF file in the selected model")
            command=["-m",gguf_path,"--host","0.0.0.0","--port",str(container_port),"--ctx-size",str(context),"--n-gpu-layers","999"]
        elif engine_key == "localai":
            # LocalAI owns backend/model configuration. Mount only the selected model
            # repository under /models while preserving Hugging Face snapshot symlinks.
            env.update({"DEBUG":"false","NVIDIA_DRIVER_CAPABILITIES":"compute,utility"})
            runtime=Path(model.get("runtime_path") or model.get("dir_path") or "").resolve()
            model_root=Path(model.get("dir_path") or runtime).resolve()
            if model.get("source") == "hf_cache" and model_root.exists():
                try:
                    rel=runtime.relative_to(model_root)
                except ValueError:
                    rel=Path(".")
                volumes=[f"{model_root}:/models/selected:ro"]
                model_path=str(Path("/models/selected")/rel)
            else:
                volumes=[f"{runtime}:/models/selected:ro"]
                model_path="/models/selected"
            command=[]
            notes.append(f"Selected model is available to LocalAI at {model_path}.")
            notes.append("LocalAI may require a model configuration YAML depending on the selected backend/model format; the generator does not invent backend-specific model YAML.")
        elif engine_key == "comfyui":
            command=[]
            notes.append("ComfyUI image is user-configured; model-specific mount behavior may need customization for a particular workflow.")

        svc:dict[str,Any]={
            "image":image,"restart":"unless-stopped","init":True,"ipc":"host",
            "environment":env,"volumes":volumes,"ports":[f"{bind}:{port}:{container_port}"],
            "labels":{
                "io.dgx-model-manager.managed":"true","io.dgx-model-manager.version":"2",
                "io.dgx-model-manager.engine":engine_key,"io.dgx-model-manager.model":model.get("full_name") or model.get("name"),
                "io.dgx-model-manager.slug":slug,
            },
        }
        if command: svc["command"]=command
        if gpu:
            svc["deploy"]={"resources":{"reservations":{"devices":[{"driver":"nvidia","count":"all","capabilities":["gpu"]}]}}}
        compose={"name":f"dmm-{slug}","services":{"inference":svc}}
        if top_volumes: compose["volumes"]=top_volumes
        yaml_text=yaml.safe_dump(compose,sort_keys=False,default_flow_style=False)
        return {
            "name": name or model.get("name") or slug, "slug":slug,"engine":engine_key,"engine_name":e["name"],
            "model_id":model.get("id"),"model_name":model.get("full_name") or model.get("name"),"model_source":model.get("source"),
            "model_path":model.get("runtime_path"),"context_length":context,"memory_reserve_gb":reserve,
            "estimated_runtime_gb":estimated,"memory_budget_gb":round(available_budget,1),"fit_status":status,"fit_ratio":round(fit_ratio,2),
            "profile":profile,"bind_host":bind,"port":port,"expose_litellm":bool(expose_litellm),"served_model_name":served_model_name,
            "engine_memory_fraction":round(mem_util,2),"quant_method":quant_method,"quant_bits":quant_bits,"yaml":yaml_text,
            "notes":notes,"node":"local","generated_at":now_iso(),
        }


    def validate_generated_plan(self, plan: dict, *, model: dict, node_metrics: dict) -> dict:
        """Rebuild a submitted plan and require it to match the safe generator output.

        Browser clients and remote-agent callers never get to persist arbitrary Compose
        YAML.  The plan is treated as a set of generation inputs; the server regenerates
        the stack from trusted inventory/node data and compares the resulting Compose
        document before saving it.
        """
        if not isinstance(plan, dict):
            raise ValueError("Invalid deployment plan")
        required = {"engine", "model_id", "yaml"}
        if not required.issubset(plan):
            raise ValueError("Deployment plan is missing required generator metadata")
        if str(plan.get("model_id")) != str(model.get("id")):
            raise ValueError("Deployment model does not match target-node inventory")
        expected = self.generate(
            model=model,
            engine_key=str(plan.get("engine")),
            node_metrics=node_metrics,
            name=plan.get("name"),
            context_length=plan.get("context_length"),
            memory_reserve_gb=plan.get("memory_reserve_gb"),
            profile=str(plan.get("profile") or "balanced"),
            bind_host=plan.get("bind_host"),
            expose_litellm=bool(plan.get("expose_litellm", True)),
        )
        try:
            submitted_yaml = yaml.safe_load(str(plan.get("yaml") or ""))
            expected_yaml = yaml.safe_load(expected["yaml"])
        except Exception as exc:
            raise ValueError(f"Invalid Compose YAML: {exc}") from exc
        if submitted_yaml != expected_yaml:
            raise ValueError("Deployment plan was modified after generation; generate it again before saving")
        for key in ("slug", "engine", "model_id", "served_model_name", "port", "bind_host"):
            if plan.get(key) != expected.get(key):
                raise ValueError(f"Deployment plan metadata mismatch: {key}")
        return expected

    def save_generated(self, plan: dict, *, overwrite: bool=False) -> dict:
        d=self.stack_dir(plan["engine"],plan["slug"])
        if d.exists() and not overwrite: raise ValueError("A deployment with this name already exists")
        d.mkdir(parents=True,exist_ok=True)
        compose_path=d/"compose.yaml"; meta_path=d/"deployment.json"
        compose_path.write_text(plan["yaml"])
        meta={k:v for k,v in plan.items() if k!="yaml"}; meta["created_at"]=meta.get("created_at") or now_iso(); meta["updated_at"]=now_iso()
        meta_path.write_text(json.dumps(meta,indent=2)+"\n")
        return {**meta,"path":str(d)}

    def get(self, engine:str,slug:str)->dict:
        d=self.stack_dir(engine,slug); m=d/"deployment.json"; c=d/"compose.yaml"
        if not m.exists() or not c.exists(): raise FileNotFoundError(slug)
        meta=json.loads(m.read_text()); meta["yaml"]=c.read_text(); meta["path"]=str(d); return meta

    async def validate(self, engine:str,slug:str)->dict:
        d=self.stack_dir(engine,slug); f=d/"compose.yaml"
        if not f.exists(): raise FileNotFoundError(slug)
        r=await run_cmd("docker","compose","-f",str(f),"config","-q",timeout=15,cwd=d)
        return {"ok":r.returncode==0,"output":(r.stdout+r.stderr).strip()}

    async def up(self, engine:str,slug:str)->dict:
        d=self.stack_dir(engine,slug); f=d/"compose.yaml"
        if not f.exists(): raise FileNotFoundError(slug)
        r=await run_cmd("docker","compose","-f",str(f),"up","-d","--remove-orphans",timeout=600,cwd=d)
        return {"ok":r.returncode==0,"output":(r.stdout+r.stderr).strip()}

    async def down(self,engine:str,slug:str)->dict:
        d=self.stack_dir(engine,slug); f=d/"compose.yaml"
        if not f.exists(): raise FileNotFoundError(slug)
        r=await run_cmd("docker","compose","-f",str(f),"down",timeout=120,cwd=d)
        return {"ok":r.returncode==0,"output":(r.stdout+r.stderr).strip()}

    async def status(self,engine:str,slug:str)->dict:
        d=self.stack_dir(engine,slug); f=d/"compose.yaml"
        if not f.exists(): return {"running":False,"state":"missing","containers":[]}
        r=await run_cmd("docker","compose","-f",str(f),"ps","--format","json",timeout=10,cwd=d)
        containers=[]
        if r.returncode==0:
            text=r.stdout.strip()
            try:
                parsed=json.loads(text); containers=parsed if isinstance(parsed,list) else [parsed]
            except Exception:
                for line in text.splitlines():
                    try: containers.append(json.loads(line))
                    except Exception: pass
        running=any(str(x.get("State",x.get("state",""))).lower()=="running" for x in containers)
        return {"running":running,"state":"running" if running else "stopped","containers":containers}

    async def logs(self,engine:str,slug:str,lines:int=200)->dict:
        d=self.stack_dir(engine,slug); f=d/"compose.yaml"
        if not f.exists(): raise FileNotFoundError(slug)
        r=await run_cmd("docker","compose","-f",str(f),"logs","--no-color","--tail",str(max(1,min(lines,1000))),timeout=20,cwd=d)
        return {"ok":r.returncode==0,"lines":(r.stdout+r.stderr).splitlines()}

    async def remove(self,engine:str,slug:str,archive:bool=True)->dict:
        d=self.stack_dir(engine,slug)
        if not d.exists(): raise FileNotFoundError(slug)
        await self.down(engine,slug)
        if archive:
            dest=self.archive/f"{engine}-{slug}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            dest.parent.mkdir(parents=True,exist_ok=True); shutil.move(str(d),str(dest)); return {"ok":True,"archived":str(dest)}
        shutil.rmtree(d); return {"ok":True}
