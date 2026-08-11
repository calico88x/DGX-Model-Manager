# DGX Model Manager v2.0.0

Release date: **2026-08-10**

DGX Model Manager v2 is a substantial redesign of the control plane around authenticated access, Docker Compose, explicit workload ownership, and safer operation on DGX Spark.

## Highlights

### Secure by default

v2 assumes the LAN is not a trust boundary.

Operational reads require authentication, browser mutations require CSRF protection, HTTPS is the default non-loopback transport, passwords use Argon2id, and the first administrator requires a one-time token generated on the host.

### Compose-first serving

Model-serving plans are generated as Docker Compose projects.

Saving a generated plan does not start it.

Generated YAML is protected by a server-side integrity boundary rather than being accepted as arbitrary browser-edited Compose.

### Existing services are not automatically claimed

v2 can observe an externally managed vLLM, ComfyUI, or other configured service without assuming ownership.

The manager will not stop an external container merely because it publishes the engine's conventional port.

### Better model metadata

The 2.0.0 inventory understands local Safetensors metadata, Hugging Face base-model relationships, mixed quantization, multimodal configurations, SentenceTransformers structure, and physical Hugging Face cache storage more accurately than the original implementation.

### Authenticated LiteLLM integration

Installations whose LiteLLM proxy protects `/health` and `/v1/models` can provide only the `LITELLM_MASTER_KEY` to v2 using a dedicated systemd credential.

The full LiteLLM environment file does not need to be readable by Model Manager.

### Multi-node architecture

One manager can enroll restricted DGX node agents.

The local node requires no agent.

Remote nodes expose only the narrow inventory, metrics, generation, and managed Compose lifecycle interface rather than a general remote shell.

## Installation

For a side-by-side installation:

```bash
bash setup.sh
```

The default v2 service is:

```text
model-manager-v2.service
```

and the default test/coexistence HTTPS port is:

```text
8091
```

Runtime state is isolated from an existing v1 installation.

## Optional LiteLLM permissions

Setup can optionally configure two separate capabilities:

1. **LiteLLM API authentication** using a dedicated systemd credential containing only `LITELLM_MASTER_KEY`.
2. **LiteLLM restart** using a narrowly scoped passwordless sudo rule.

Neither capability requires publishing credentials into the repository or browser configuration.

## Compatibility

Primary target:

- NVIDIA DGX Spark / GB10
- Linux / aarch64
- Docker + Docker Compose
- NVIDIA Container Toolkit
- systemd

Model integrations include:

- Hugging Face cache
- Ollama
- vLLM
- SGLang
- llama.cpp
- LocalAI
- ComfyUI
- LiteLLM

Engine/container compatibility still depends on the selected image, model architecture, CUDA stack, quantization format, and upstream engine support.

## Hardware acceptance status

The 2.0.0 release received a live side-by-side acceptance pass on a DGX Spark / GB10 while existing workloads remained available.

Validated live:

- setup and bootstrap
- authenticated HTTPS UI
- dashboard telemetry
- model inventory classification and size accounting
- Hugging Face metadata/search/files/download/delete
- Ollama service discovery
- Compose generation
- real port-collision avoidance
- Compose save without start
- `docker compose config` validation
- stopped deployment status
- deployment logs
- deployment archive
- authenticated LiteLLM `/v1/models`
- redacted LiteLLM configuration display
- serving-engine discovery
- protection of externally managed engines
- local cluster/node view
- diagnostics
- user/admin lockout safeguards
- Settings service tests and persistence
- Field Manual rendering

Not hardware-exercised before publication:

- first launch/stop of a new v2-managed GPU-serving stack;
- LiteLLM route mutation/restart during active client traffic;
- remote lifecycle against a second physical DGX Spark.

The application is expected to evolve based on field feedback, particularly around multi-node operation and engine/image compatibility.

## Upgrade strategy

v2 is intentionally suitable for side-by-side deployment.

Do not replace an existing production installation until you have validated:

- authentication
- current model inventory
- service endpoints
- Docker/Compose access
- generated serving plans
- TLS access
- any LiteLLM integration you depend on

Use `scripts/migrate_from_v1.py` to import compatible settings without overwriting the source installation.

Use `scripts/promote_v2.sh` only after acceptance.

## Security note

Do not commit:

- runtime `config.json`
- `.env` files
- LiteLLM keys
- database credentials
- TLS private keys
- API tokens
- local databases
- generated logs
- host-specific runtime state

The public repository contains `config.example.json` instead.

## Release validation

Before packaging:

```bash
pytest -q
python3 scripts/validate_release.py
bash scripts/build_release.sh
```

Release archives contain their own `MANIFEST.sha256`, and top-level archive checksums are written beside the generated `.tar.gz` and `.zip` files in `dist/`.
