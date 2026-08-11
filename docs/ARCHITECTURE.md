# v2 Architecture

## Control plane

DGX Model Manager v2 is a control plane, not an inference proxy.

```text
Browser
   │ HTTPS + authenticated session
   ▼
DGX Model Manager
   ├── local model inventory
   ├── Hugging Face API
   ├── Ollama API
   ├── LiteLLM API/config
   ├── Docker Compose
   ├── host diagnostics
   └── optional remote DGX agents

Inference clients
   └──────────────► LiteLLM / vLLM / SGLang / llama.cpp / other engine
```

The manager is not placed in the model token-generation data path.

## Local deployment state

Generated Compose projects are stored under the configured Compose root, normally:

```text
~/.local/share/dgx-model-manager-v2/compose/
├── stacks/
│   └── <engine>/<slug>/
│       ├── compose.yaml
│       └── deployment.json
└── archive/
```

Saving a deployment creates only local Compose YAML and metadata.

It does not implicitly execute `docker compose up`.

## Workload ownership

Service reachability and workload ownership are separate concepts.

An engine can be reachable at its configured base URL while being managed by:

- another systemd service;
- another Compose project;
- a manually launched container;
- another operator.

v2 will report such a service as Online but will not treat it as a managed deployment.

Stop operations are restricted to v2-managed Compose profiles.

There is deliberately no port-based fallback that stops whichever container happens to publish the conventional engine port.

## Compose trust boundary

Compose Builder operates on trusted server-side inventory and configuration.

The browser receives generated YAML for review/copying, but normal save operations do not accept arbitrary edited Compose YAML from the client.

This prevents a browser user from turning the generator API into an arbitrary:

- host mount
- privileged container
- Docker socket mount
- host-networking
- arbitrary-command

interface.

Users with host filesystem authority may still edit saved Compose files directly, outside the generator trust boundary.

## Host-port allocation

Each engine has a preferred host port.

The generator checks:

1. ports recorded by saved v2 deployments; and
2. whether the host can actually bind the candidate port.

When occupied, it scans subsequent ports and records the selected replacement in deployment metadata and generator notes.

Container-internal engine ports are unchanged.

## Authentication boundary

The browser authenticates to DGX Model Manager.

Credentials for downstream local services remain server-side.

For LiteLLM, v2 can read `LITELLM_MASTER_KEY` from a dedicated systemd service credential.

The application does not need access to the broader LiteLLM environment file.

Configuration returned to the browser passes through recursive secret redaction.

## User authority

Roles are:

- Viewer
- Operator
- Admin

The backend enforces:

- CSRF on browser mutations;
- last-active-admin preservation;
- self-disable protection;
- API-token authority no greater than the owner's current role.

UI restrictions are convenience and clarity; backend authorization remains authoritative.

## Model inventory

Inventory is local-first.

Hugging Face enrichment is supplementary and cached.

Safetensors metadata can be inspected without loading model weights.

Quantized model metadata separates:

- base dtype;
- checkpoint dtype;
- quantizer implementation;
- quantization format;
- bit widths;
- mixed-quantization state.

Hugging Face cache disk usage de-duplicates inode-backed shared storage.

## LiteLLM boundary

The manager may:

- read active models from `/v1/models`;
- display a redacted local LiteLLM configuration;
- add/remove supported generated routes;
- maintain an Ollama wildcard;
- optionally restart LiteLLM.

A dedicated systemd credential can carry only `LITELLM_MASTER_KEY`.

A separate optional sudo capability controls whether v2 can restart the LiteLLM systemd unit.

## Remote nodes

A local DGX node requires no agent.

Remote nodes use the restricted v2 node agent.

The agent exposes specific management RPCs for:

- node information
- metrics
- inventory
- Compose generation
- managed Compose lifecycle

It does not expose a generic shell.

Remote TLS uses normal certificate validation where possible. Self-signed operation requires explicit certificate fingerprint pinning when standard verification is disabled.

## Current multi-node scope

Each deployment belongs to one node.

v2 does not currently distribute one model or one inference job across multiple DGX Spark systems.

The architecture is centralized fleet orchestration rather than distributed tensor/model parallelism.
