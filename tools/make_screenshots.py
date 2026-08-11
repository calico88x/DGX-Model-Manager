#!/usr/bin/env python3
"""Render public-safe DGX Model Manager v2 screenshots from static demo data.

These screenshots intentionally contain no runtime data.  They use documentation-only
addresses and synthetic host/user names so they are safe to publish in the repository.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from html import escape

from weasyprint import HTML

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "misc" / "screenshots"
CSS = (ROOT / "static" / "app.css").read_text()

PAGE_CSS = r"""
@page { size: 1600px 1000px; margin: 0; }
html, body { width:1600px; height:1000px; overflow:hidden; }
body { zoom: 1; }
.app { width:1600px; height:1000px; }
.main { padding-top:24px; }
.page { max-width:none; }
.screenshot-note { font:9px var(--mono); color:var(--dim); }
.table td, .table th { white-space:nowrap; }
.btn { display:inline-block !important; appearance:none; }
.actions .btn + .btn { margin-left:6px; }
.code-pane { max-height:720px; min-height:650px; }
.builder { grid-template-columns:410px minmax(0,1fr); }
"""

NAV = [
    ("Overview", [("⌂", "Dashboard", "dashboard", None)]),
    ("Models", [("▦", "Inventory", "inventory", "8"), ("◉", "Ollama", "ollama", None), ("◎", "HF Browser", "hf", None), ("⇩", "Downloads", "downloads", None)]),
    ("Serving", [("▤", "Deployments", "deployments", "3"), ("◇", "Compose Builder", "builder", None), ("⇄", "LiteLLM Routes", "routing", None), ("⚡", "Engines", "engines", None)]),
    ("System", [("◫", "Cluster", "cluster", "2"), ("≡", "Logs & Diagnostics", "logs", None)]),
    ("Administration", [("◇", "Users & Access", "access", None), ("⚙", "Settings", "settings", None), ("?", "Documentation", "docs", None)]),
]


def nav_html(active: str) -> str:
    out = []
    for label, items in NAV:
        out.append(f'<div class="nav-group"><div class="nav-label">{label}</div>')
        for icon, title, key, badge in items:
            cls = "nav-item active" if key == active else "nav-item"
            b = f'<span class="nav-badge">{badge}</span>' if badge else ""
            out.append(f'<button class="{cls}"><span class="nav-icon">{icon}</span><span>{title}</span>{b}</button>')
        out.append("</div>")
    return "".join(out)


def shell(active: str, content: str, node: str = "spark-alpha") -> str:
    return f"""<!doctype html><html><head><meta charset='utf-8'><style>{CSS}\n{PAGE_CSS}</style></head>
<body><div class='app'>
<header class='topbar'>
  <div class='brand'><div class='sigil'>D</div><div class='brand-copy'><div class='brand-name'>Model Manager</div><div class='brand-sub'>{node} · DGX Spark control plane</div></div><span class='version-badge'>v2.0.0</span></div>
  <div class='top-spacer'></div>
  <div class='cluster-chip'><span class='live-dot'></span><span>2 nodes · healthy</span></div>
  <select class='node-select'><option>{node}</option></select>
  <button class='user-chip'><span class='avatar'>A</span><span class='user-name'>admin</span></button>
</header>
<aside class='sidebar'>{nav_html(active)}<div class='sidebar-footer'>DGX MODEL MANAGER<br>COMPOSE-FIRST CONTROL PLANE</div></aside>
<main class='main'><section class='page active'>{content}</section></main>
</div></body></html>"""


def metric(label: str, value: str, foot: str, pct: int | None = None, accent: str = "var(--amber)") -> str:
    bar = f"<div class='bar'><span style='width:{pct}%;--bar:{accent}'></span></div>" if pct is not None else ""
    return f"<div class='metric' style='--metric-accent:{accent}'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div>{bar}<div class='metric-foot'><span>{foot}</span></div></div>"


def dashboard() -> str:
    metrics = "".join([
        metric("Unified memory", "78.4 / 128 GB", "61% allocated · 49.6 GB available", 61, "var(--green)"),
        metric("Managed models", "8", "141.7 GB on disk · 3 sources", None, "var(--blue)"),
        metric("Compose stacks", "3", "2 running · 1 stopped", None, "var(--amber)"),
        metric("Nodes", "2 / 2", "spark-alpha · spark-beta", None, "var(--purple)"),
    ])
    services = "".join([
        service("Ollama", "http://127.0.0.1:11434", "Healthy", "5 ms", "ok"),
        service("LiteLLM", "http://127.0.0.1:4000", "Healthy", "11 ms", "ok"),
        service("vLLM", "Qwen3.5-35B-A3B", "Serving", "4 ms", "ok"),
        service("SGLang", "No active stack", "Offline", "—", "err"),
        service("ComfyUI", "No active stack", "Offline", "—", "err"),
    ])
    stacks = "".join([
        stack_row("qwen35-vllm", "vLLM · Qwen3.5-35B-A3B-NVFP4", "spark-alpha", "8000", "Running", "green"),
        stack_row("embed-mini", "vLLM · all-MiniLM-L6-v2", "spark-alpha", "8002", "Running", "green"),
        stack_row("vision-lab", "SGLang · example/Vision-LLM", "spark-beta", "30000", "Stopped", "red"),
    ])
    platform = "".join([
        mini_panel("GPU architecture", "NVIDIA GB10 / SM121A", "Unified CPU/GPU memory model detected"),
        mini_panel("Runtime", "aarch64 · CUDA 13", "Docker + NVIDIA runtime available"),
        mini_panel("Security", "HTTPS · RBAC · CSRF", "Authenticated reads and writes"),
    ])
    return f"""
<div class='page-head'><div class='page-head-copy'><div class='eyebrow'>Overview</div><h1>Infrastructure dashboard</h1><div class='page-desc'>Operational state for the selected DGX node, model inventory, inference services, and managed Compose deployments.</div></div><div class='actions'><button class='btn'>↻ Refresh</button></div></div>
<div class='grid cols-4'>{metrics}</div>
<div class='grid cols-2 section-gap'>
<div class='panel'><div class='panel-head'><div class='panel-title'>Service health</div><div class='panel-sub'>authenticated control-plane view</div></div><div class='panel-body'><div class='service-list'>{services}</div></div></div>
<div class='panel'><div class='panel-head'><div class='panel-title'>Managed deployments</div><div class='panel-sub'>Compose stacks</div><div class='actions'><button class='btn small'>Open all</button></div></div><div class='panel-body'><div class='stack-list'>{stacks}</div></div></div>
</div>
<div class='panel section-gap'><div class='panel-head'><div class='panel-title'>Platform summary</div><div class='panel-sub'>spark-alpha · 192.0.2.21</div></div><div class='panel-body'><div class='grid cols-3'>{platform}</div></div></div>
"""


def service(name, meta, status, latency, cls):
    return f"<div class='service-row'><span class='status-dot {cls}'></span><div class='row-main'><div class='service-name'>{name}</div><div class='service-meta'>{meta}</div></div><div class='row-right'><span>{latency}</span><span class='badge {'green' if cls=='ok' else 'red'}'>{status}</span></div></div>"


def stack_row(name, meta, node, port, status, badge):
    return f"<div class='stack-row'><div class='iconbox'>◆</div><div class='row-main'><div class='stack-name'>{name}</div><div class='stack-meta'>{meta}</div></div><div class='row-right'><span>{node}</span><span>:{port}</span><span class='badge {badge}'>{status}</span></div></div>"


def mini_panel(k, v, sub):
    return f"<div class='mini'><div class='mini-l'>{k}</div><div class='mini-v'>{v}</div><div class='subtle mt12'>{sub}</div></div>"


def inventory() -> str:
    rows = [
        ("Qwen3.5-35B-A3B-NVFP4","example-org","Vision LLM","safetensors","FP4","35B","25.1 GB","HF Cache","Build"),
        ("Mistral-Small-4-119B-NVFP4","example-org","Text Gen","safetensors","FP4","119B","70.8 GB","HF Cache","Build"),
        ("all-MiniLM-L6-v2","sentence-transformers","Embedding","safetensors","FP32","—","0.9 GB","HF Cache","Build"),
        ("hubert-base-ls960","facebook","Audio","safetensors","FP32","—","0.8 GB","HF Cache","Build"),
        ("qwen3:8b","","Text Gen","ollama","Q4_K_M","8B","5.2 GB","Ollama","Open"),
        ("gemma3:4b","","Vision LLM","ollama","Q4_K_M","4B","3.3 GB","Ollama","Open"),
        ("lab-gguf-q5","demo","Text Gen","gguf","Q5_K_M","14B","10.6 GB","Custom","Build"),
        ("reranker-small","demo","Embedding","safetensors","FP16","0.6B","1.2 GB","Custom","Build"),
    ]
    tr=[]
    for n,o,t,f,d,p,s,src,act in rows:
        owner=f"<div class='model-owner'>{o}</div>" if o else ""
        srccls="green" if src=="Ollama" else "amber" if src=="HF Cache" else "purple"
        tr.append(f"<tr><td><div class='model-title'>{n}</div>{owner}</td><td><span class='badge blue'>{t}</span></td><td><span class='badge'>{f}</span></td><td><span class='badge amber'>{d}</span></td><td class='mono'>{p}</td><td class='mono'>{s}</td><td><span class='badge {srccls}'>{src}</span></td><td><button class='btn small'>{act}</button></td></tr>")
    return f"""
<div class='page-head'><div class='page-head-copy'><div class='eyebrow'>Models</div><h1>Model inventory</h1><div class='page-desc'>Existing HuggingFace cache and Ollama paths are retained. Custom directories can be added without moving model data.</div></div><div class='actions'><button class='btn'>↻ Refresh</button></div></div>
<div class='toolbar'><input class='input' value='' placeholder='Search models'><select class='select'><option>All sources</option></select><select class='select'><option>All formats</option></select><select class='select'><option>All tasks</option></select></div>
<div class='panel'><div class='panel-head'><div class='panel-title'>8 models · 117.1 GB</div><div class='panel-sub'>HF Cache + Ollama + Custom</div><div class='actions'><button class='btn small'>Add directory</button></div></div><div class='table-wrap'><table class='table'><thead><tr><th>Model</th><th>Task</th><th>Format</th><th>Dtype</th><th>Params</th><th>Size</th><th>Source</th><th>Action</th></tr></thead><tbody>{''.join(tr)}</tbody></table></div></div>
<div class='panel section-gap'><div class='panel-head'><div class='panel-title'>Scan directories</div></div><div class='panel-body'><div class='service-row'><div class='iconbox'>HF</div><div class='row-main'><div class='service-name'>Default HuggingFace cache</div><div class='service-meta'>~/.cache/huggingface/hub</div></div><span class='badge green'>6 models</span></div><div class='service-row'><div class='iconbox'>＋</div><div class='row-main'><div class='service-name'>Custom model directory</div><div class='service-meta'>~/models/lab</div></div><span class='badge purple'>2 models</span></div></div></div>
"""


def deployments() -> str:
    rows = [
        ("qwen35-vllm","vLLM","Qwen3.5-35B-A3B-NVFP4","spark-alpha","Good",":8000","Running"),
        ("embed-mini","vLLM","all-MiniLM-L6-v2","spark-alpha","Good",":8002","Running"),
        ("vision-lab","SGLang","example/Vision-LLM","spark-beta","Good",":30000","Stopped"),
    ]
    tr=[]
    for name,eng,model,node,fit,port,status in rows:
        st='green' if status=='Running' else 'red'
        tr.append(f"<tr><td><div class='model-title'>{name}</div><div class='model-owner'>managed compose</div></td><td><span class='badge blue'>{eng}</span></td><td>{model}</td><td class='mono'>{node}</td><td><span class='badge green'>{fit}</span></td><td class='mono'>{port}</td><td><span class='badge {st}'>{status}</span></td><td><div class='actions'><button class='btn small'>{'Stop' if status=='Running' else 'Start'}</button><button class='btn small'>Logs</button><button class='btn small'>LiteLLM</button></div></td></tr>")
    return f"""
<div class='page-head'><div class='page-head-copy'><div class='eyebrow'>Serving</div><h1>Compose deployments</h1><div class='page-desc'>Declarative model-serving stacks managed with Docker Compose. Existing shell scripts remain available only when Legacy Script Mode is enabled.</div></div><div class='actions'><button class='btn primary'>＋ New deployment</button><button class='btn'>↻ Refresh</button></div></div>
<div class='grid cols-4'>{metric('Running stacks','2','across 2 nodes',None,'var(--green)')}{metric('Reserved memory','61 GB','estimated model budget',48,'var(--blue)')}{metric('LiteLLM routes','4','2 compose · 2 Ollama',None,'var(--amber)')}{metric('Legacy mode','Off','Compose-first posture',None,'var(--purple)')}</div>
<div class='panel section-gap'><div class='table-wrap'><table class='table'><thead><tr><th>Deployment</th><th>Engine</th><th>Model</th><th>Node</th><th>Fit</th><th>Port</th><th>Status</th><th>Actions</th></tr></thead><tbody>{''.join(tr)}</tbody></table></div></div>
<div class='callout green section-gap'><strong>Compose-first:</strong> model checkpoints are referenced from their existing cache paths. Saving a deployment creates only its Compose YAML and metadata under the v2 data directory.</div>
"""


def builder() -> str:
    yaml = """name: dmm-qwen35-vllm
services:
  inference:
    image: vllm/vllm-openai:cu130-nightly
    restart: unless-stopped
    init: true
    ipc: host
    environment:
      HF_HOME: /root/.cache/huggingface
      HF_HUB_DISABLE_TELEMETRY: "1"
      TRITON_PTXAS_PATH: /usr/local/cuda/bin/ptxas
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface:ro
      - vllm-cache:/root/.cache/vllm
    ports:
      - "127.0.0.1:8000:8000"
    labels:
      io.dgx-model-manager.managed: "true"
      io.dgx-model-manager.version: "2"
      io.dgx-model-manager.engine: vllm
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    command:
      - --model
      - /root/.cache/huggingface/hub/models--example-org--Qwen3.5-35B-A3B-NVFP4/snapshots/main
      - --served-model-name
      - example-org-qwen3.5-35b-a3b-nvfp4
      - --host
      - 0.0.0.0
      - --port
      - "8000"
      - --max-model-len
      - "32768"
      - --gpu-memory-utilization
      - "0.30"
      - --max-num-seqs
      - "2"
volumes:
  vllm-cache: {}
"""
    signals = "".join([
        "<div class='signal'><div class='signal-k'>Architecture</div><div class='signal-v'>GB10 · SM121A</div></div>",
        "<div class='signal'><div class='signal-k'>Unified memory</div><div class='signal-v'>128 GB</div></div>",
        "<div class='signal'><div class='signal-k'>Model footprint</div><div class='signal-v'>25.1 GB · FP4</div></div>",
        "<div class='signal'><div class='signal-k'>Estimated fit</div><div class='signal-v fit-good'>GOOD · 61 GB budget</div></div>",
    ])
    return f"""
<div class='page-head'><div class='page-head-copy'><div class='eyebrow'>Serving</div><h1>Compose Builder</h1><div class='page-desc'>Generate a DGX-aware Compose stack from model metadata and target-node capacity. The planner uses the existing model path rather than copying the checkpoint.</div></div></div>
<div class='builder'><div class='panel'><div class='panel-head'><div class='panel-title'>Deployment inputs</div></div><div class='builder-form'><div class='stack'>
<div><label class='label'>Model</label><select class='select w100'><option>example-org/Qwen3.5-35B-A3B-NVFP4</option></select></div>
<div class='form-grid'><div><label class='label'>Engine</label><select class='select w100'><option>vLLM</option></select></div><div><label class='label'>Target node</label><select class='select w100'><option>spark-alpha</option></select></div></div>
<div><label class='label'>Deployment name</label><input class='input w100' value='qwen35-vllm'></div>
<div class='form-grid'><div><label class='label'>Context length</label><input class='input w100' value='32768'></div><div><label class='label'>System reserve (GB)</label><input class='input w100' value='24'></div></div>
<div class='form-grid'><div><label class='label'>Optimization profile</label><select class='select w100'><option>Balanced</option></select></div><div><label class='label'>Bind address</label><select class='select w100'><option>Loopback — recommended</option></select></div></div>
<label class='toggle-row'><div class='toggle-copy'><div class='toggle-title'>Prepare for LiteLLM routing</div><div class='toggle-sub'>Records routing intent in deployment metadata.</div></div><div class='switch on'></div></label>
<button class='btn primary w100'>Generate Compose</button></div>{signals}<div class='decision'><strong>Plan:</strong> model fits comfortably with a 24 GB host reserve. GB10 compatibility settings were added automatically; service remains loopback-only until explicitly exposed.</div></div></div>
<div class='panel'><div class='panel-head'><div class='panel-title'>Generated compose.yaml</div><div class='actions'><button class='btn small'>Copy</button><button class='btn primary small'>Save deployment</button></div></div><pre class='code-pane'>{escape(yaml)}</pre></div></div>
"""


def cluster() -> str:
    nodes = "".join([
        node_row("spark-alpha","Local node · 192.0.2.21","GB10 · aarch64 · 128 GB","61% memory","Healthy","green"),
        node_row("spark-beta","Remote agent · 192.0.2.22","GB10 · aarch64 · 128 GB","34% memory","Healthy","green"),
    ])
    return f"""
<div class='page-head'><div class='page-head-copy'><div class='eyebrow'>System</div><h1>Cluster</h1><div class='page-desc'>The local Spark works without an agent. Additional DGX Spark nodes can enroll through the optional authenticated node agent.</div></div><div class='actions'><button class='btn primary'>＋ Add node</button><button class='btn'>↻ Refresh</button></div></div>
<div class='grid cols-3'>{metric('Nodes online','2 / 2','mutually authenticated control plane',None,'var(--green)')}{metric('Unified memory','256 GB','aggregate across enrolled nodes',48,'var(--blue)')}{metric('Running stacks','2','both currently on spark-alpha',None,'var(--amber)')}</div>
<div class='panel section-gap'><div class='panel-head'><div class='panel-title'>Nodes</div><div class='panel-sub'>certificate-pinned agents supported</div></div><div class='panel-body'><div class='node-list'>{nodes}</div></div></div>
<div class='grid cols-2 section-gap'><div class='panel'><div class='panel-head'><div class='panel-title'>Placement posture</div></div><div class='panel-body'><div class='secure-row'><span class='check'>✓</span><div><strong>Local operations stay local</strong><div class='subtle'>No agent is required for the manager's own DGX Spark.</div></div></div><div class='secure-row'><span class='check'>✓</span><div><strong>Remote nodes expose constrained APIs</strong><div class='subtle'>No arbitrary remote shell and no network-exposed Docker socket.</div></div></div><div class='secure-row'><span class='check'>✓</span><div><strong>Enrollment token encrypted at rest</strong><div class='subtle'>Self-signed nodes can be pinned by SHA-256 certificate fingerprint.</div></div></div></div></div><div class='panel'><div class='panel-head'><div class='panel-title'>Node capacity</div></div><div class='panel-body'><div class='service-row'><div class='row-main'><div class='service-name'>spark-alpha</div><div class='service-meta'>78.4 / 128 GB used</div></div><span class='badge amber'>61%</span></div><div class='bar'><span style='width:61%;--bar:var(--amber)'></span></div><div class='service-row'><div class='row-main'><div class='service-name'>spark-beta</div><div class='service-meta'>43.8 / 128 GB used</div></div><span class='badge green'>34%</span></div><div class='bar'><span style='width:34%;--bar:var(--green)'></span></div></div></div></div>
"""


def node_row(name, meta, spec, mem, status, badge):
    return f"<div class='node-row'><div class='iconbox'>DGX</div><div class='row-main'><div class='node-name'>{name}</div><div class='node-meta'>{meta}</div></div><div class='row-right'><span>{spec}</span><span class='badge amber'>{mem}</span><span class='badge {badge}'>{status}</span><button class='btn small'>Details</button></div></div>"


def access() -> str:
    users = """
<table class='user-table'><thead><tr><th>User</th><th>Role</th><th>Status</th><th>Last sign-in</th><th></th></tr></thead><tbody>
<tr><td><div class='model-title'>admin</div><div class='model-owner'>Administrator</div></td><td><span class='badge amber'>Admin</span></td><td><span class='badge green'>Enabled</span></td><td class='mono'>Today 14:32</td><td><button class='btn small'>Manage</button></td></tr>
<tr><td><div class='model-title'>operator</div><div class='model-owner'>Model Operator</div></td><td><span class='badge blue'>Operator</span></td><td><span class='badge green'>Enabled</span></td><td class='mono'>Today 13:08</td><td><button class='btn small'>Manage</button></td></tr>
<tr><td><div class='model-title'>viewer</div><div class='model-owner'>Read Only</div></td><td><span class='badge purple'>Viewer</span></td><td><span class='badge green'>Enabled</span></td><td class='mono'>Yesterday</td><td><button class='btn small'>Manage</button></td></tr>
</tbody></table>"""
    tokens = "".join([
        "<div class='service-row'><div class='iconbox'>API</div><div class='row-main'><div class='service-name'>automation-readonly</div><div class='service-meta'>Viewer · created 2026-08-01</div></div><span class='badge green'>Active</span><button class='btn small danger'>Revoke</button></div>",
        "<div class='service-row'><div class='iconbox'>API</div><div class='row-main'><div class='service-name'>lab-operator</div><div class='service-meta'>Operator · created 2026-08-06</div></div><span class='badge green'>Active</span><button class='btn small danger'>Revoke</button></div>",
    ])
    audits = "".join([
        audit("14:32:18","admin","compose.start","qwen35-vllm on spark-alpha","green"),
        audit("14:29:42","admin","litellm.route.update","Qwen3.5-35B-A3B","blue"),
        audit("13:08:04","operator","inventory.refresh","8 models discovered","purple"),
        audit("12:51:17","admin","user.create","viewer · Viewer role","amber"),
    ])
    return f"""
<div class='page-head'><div class='page-head-copy'><div class='eyebrow'>Administration</div><h1>Users & access</h1><div class='page-desc'>Local accounts, role-based access, session authentication, and scoped API tokens. Public self-registration is disabled by default.</div></div><div class='actions'><button class='btn primary'>＋ Create user</button><button class='btn'>↻ Refresh</button></div></div>
<div class='grid cols-4'>{metric('Accounts','3','1 Admin · 1 Operator · 1 Viewer',None,'var(--amber)')}{metric('Active tokens','2','opaque values shown only once',None,'var(--blue)')}{metric('Registration','Off','recommended on untrusted LAN',None,'var(--green)')}{metric('Session policy','Strict','HttpOnly · SameSite · CSRF',None,'var(--purple)')}</div>
<div class='grid cols-2 section-gap'><div class='panel'><div class='panel-head'><div class='panel-title'>Accounts</div></div><div class='panel-body'>{users}</div></div><div class='panel'><div class='panel-head'><div class='panel-title'>API tokens</div><div class='actions'><button class='btn small'>＋ Token</button></div></div><div class='panel-body'>{tokens}<div class='callout amber mt12'>Token plaintext is displayed only at creation. Only a one-way hash is retained afterward.</div></div></div></div>
<div class='panel section-gap'><div class='panel-head'><div class='panel-title'>Recent audit events</div></div><div class='panel-body'><div class='activity-list'>{audits}</div></div></div>
"""


def audit(ts, actor, action, detail, badge):
    return f"<div class='activity-row'><div class='iconbox'>•</div><div class='row-main'><div class='service-name'>{action}</div><div class='service-meta'>{detail}</div></div><div class='row-right'><span class='mono'>{ts}</span><span class='badge {badge}'>{actor}</span></div></div>"


PAGES = {
    "dashboard": dashboard,
    "inventory": inventory,
    "deployments": deployments,
    "compose-builder": builder,
    "cluster": cluster,
    "users-access": access,
}


def render(name: str, active: str, content: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="dmm2-shot-") as td:
        pdf = Path(td) / f"{name}.pdf"
        HTML(string=shell(active, content), base_url=str(ROOT)).write_pdf(pdf)
        outbase = OUT / name
        subprocess.run([
            "pdftoppm", "-png", "-singlefile", "-r", "96", str(pdf), str(outbase)
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> int:
    targets = [
        ("dashboard", "dashboard"),
        ("inventory", "inventory"),
        ("deployments", "deployments"),
        ("compose-builder", "builder"),
        ("cluster", "cluster"),
        ("users-access", "access"),
    ]
    for name, active in targets:
        print(f"rendering {name}...")
        render(name, active, PAGES[name]())
    print(f"wrote {len(targets)} screenshots to {OUT}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
