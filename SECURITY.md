# Security Policy

DGX Model Manager is a privileged infrastructure control plane. A vulnerability can affect model files, inference availability, Docker workloads, LiteLLM routing, or host information. Treat security reports accordingly.

## Supported release line

The current v2 release line is supported. Security fixes should target the latest v2 revision unless a release note states otherwise.

## Threat model

v2 assumes:

- the LAN may contain untrusted or compromised clients;
- browser access must not be trusted merely because the source address is RFC1918/private;
- Docker control is high privilege and must not be remotely exposed as a raw socket;
- runtime configuration may contain credentials even when the source repository does not;
- operational metadata such as hostname, IP address, model inventory, service topology, logs, and paths is private unless an authenticated operator chooses to disclose it;
- remote DGX nodes require authenticated encrypted transport.

## Default controls

- HTTPS required for non-loopback manager access
- one-time local bootstrap token required to create the first administrator
- Argon2id password hashing
- opaque HttpOnly session cookies, stored as hashes server-side
- SameSite=Strict and Secure cookies by default
- per-session CSRF protection for browser mutations
- login throttling
- Viewer / Operator / Admin RBAC
- invariant requiring at least one active Admin
- password-change session revocation
- API-token effective role bounded by the owner's current role
- hashed scoped API tokens
- private/loopback-only service-target policy by default
- LiteLLM secret redaction
- audit records for security-sensitive operations
- encrypted remote-node enrollment tokens
- TLS certificate verification or explicit SHA-256 pinning for remote self-signed agents
- server-side regeneration/validation of Compose plans before persistence
- request-size guards on manager and node-agent HTTP endpoints
- no general remote shell and no network-exposed Docker socket

## Deployment recommendations

1. Replace the setup-generated self-signed certificate with one trusted by your environment when practical.
2. Keep self-registration disabled unless required.
3. Keep Legacy Script Mode disabled after migration.
4. Keep inference backends bound to loopback unless remote clients genuinely require direct access.
5. Restrict OS-level access to the Model Manager service account and v2 runtime directories.
6. Keep Docker, NVIDIA Container Toolkit, Python dependencies, inference engines, and the host OS patched.
7. Treat generated Compose review as inspection, not as a way to bypass the generator; save endpoints reject modified plans.
8. Do not commit runtime `config.json`, databases, certificates/private keys, enrollment tokens, API tokens, LiteLLM secrets, or logs.
9. Review generated Compose YAML before first launch of an unfamiliar model/image.

## Reporting a vulnerability

Do **not** open a public issue containing exploit details, credentials, tokens, private addresses, or sensitive logs.

Use GitHub's private security-advisory / private vulnerability-reporting mechanism for this repository when available. Include:

- affected version/commit;
- reproduction steps;
- security impact;
- whether the issue requires authentication or a particular role;
- relevant sanitized logs;
- suggested mitigation if known.

## Scope notes

DGX Model Manager launches and controls third-party software. A vulnerability solely in an upstream engine, model, Docker image, Docker Engine, NVIDIA runtime, LiteLLM, Ollama, or HuggingFace library may need to be reported to that project. If Model Manager makes the upstream issue materially worse through unsafe defaults or privilege boundaries, it is in scope here as well.
