# Changelog

All notable changes to DGX Model Manager are documented here.

## [2.0.0] - 2026-08-10

### Added

- Authenticated HTTPS control plane with Viewer, Operator, and Admin roles.
- One-time host-local administrator bootstrap token.
- Argon2id password hashing, opaque browser sessions, and CSRF protection.
- Scoped API tokens and administrative audit history.
- DGX Spark / GB10 platform and unified-memory awareness.
- Hugging Face cache inventory with custom scan directories.
- Hugging Face repository search, file-size inspection, download progress, and local deletion.
- Ollama inventory, pull, and delete support.
- Compose Builder with DGX-aware runtime estimates and optimization profiles.
- Collision-aware host-port allocation for generated Compose deployments.
- Saved Compose deployment lifecycle: validate, start, stop, logs, archive.
- Optional LiteLLM route management for supported generated deployments.
- Optional Ollama wildcard management.
- Optional multi-node DGX Spark agent architecture.
- Diagnostics for host, Docker, Compose, application logs, LiteLLM journal, and container discovery.
- Public Field Manual (`docs.html`).
- Offline release validator and reproducible release archive builder.

### Model metadata

- Structural SentenceTransformers detection.
- Whisper/STT, TTS, audio, embedding, text-generation, and multimodal classification improvements.
- Nested vision/audio configuration recognition.
- Exact Safetensors parameter counting for non-quantized checkpoints without loading tensor data.
- Safetensors dtype inference.
- Separation of base dtype from checkpoint storage precision.
- Quantizer implementation and quantization format tracked independently.
- Mixed quantization represented explicitly with all discovered bit widths.
- Logical parameter resolution for quantized Hugging Face derivatives from declared base models.
- Persistent private Hugging Face metadata cache with offline-tolerant enrichment.
- Metadata-only Hugging Face cache directories excluded from installed model inventory.
- Hugging Face cache physical-size accounting de-duplicates shared inode-backed blobs/snapshot links.
- Sub-gigabyte model sizes retained and displayed using adaptive units.

### Compose Builder

- Generated-plan integrity guard prevents browser-supplied arbitrary YAML from becoming a privileged Compose injection path.
- Saved deployments remain stopped until explicitly launched.
- Preferred engine ports are checked against both saved deployment metadata and real host bind availability.
- Port reassignment is recorded in generator decision notes.
- vLLM relies on checkpoint quantization auto-detection.
- Stable served-model names are generated for routable OpenAI-compatible deployments.
- Deployment log viewer enlarged for operational use.

### LiteLLM

- LiteLLM health detection recognizes authenticated endpoints.
- Optional server-side authentication using `LITELLM_MASTER_KEY`.
- LiteLLM master key can be supplied to Model Manager through a dedicated systemd `LoadCredential=` credential.
- The Model Manager does not need access to the full LiteLLM environment file.
- Authenticated `/v1/models` discovery for Routing.
- Credential-aware Settings health tests.
- Refined secret redaction avoids false positives such as `max_tokens`.
- LiteLLM config is redacted before reaching the browser.
- Optional narrowly scoped sudo rule for LiteLLM restart.

### Serving-engine safety

- Running externally managed serving services can be detected without being claimed as v2-managed.
- Removed unsafe fallback that stopped the first container publishing an engine's conventional port.
- Stop operations now apply only to v2-managed Compose deployments.
- Serving Engines UI disables Stop for externally managed services.

### Users and access

- Backend prevents disabling the currently signed-in account.
- Backend prevents disabling or demoting the last active administrator.
- UI mirrors those invariants.
- Signed-in user is identified with a `You` badge.

### Settings

- Canonical service display names used consistently.
- Endpoint tests display persistent inline success/failure state and latency in addition to toast notifications.
- LiteLLM endpoint tests authenticate when a systemd credential is available.

### Security

- HTTPS-first browser access.
- Authenticated operational reads.
- CSRF protection for mutations.
- Private/loopback-first service target policy.
- Public/unsafe target rejection by default.
- Secret redaction before configuration reaches the browser.
- Runtime configuration, databases, TLS keys, tokens, environment files, logs, and generated artifacts excluded from publication.
- Compose generator refuses arbitrary browser-edited YAML.
- External-service ownership boundary prevents accidental shutdown of workloads not created by v2.

### Release / packaging

- Public `config.example.json` separated from local runtime configuration.
- Maintainer-only hardware acceptance plan and release checklist excluded from Git/public archives.
- Generated release packages are written under ignored `dist/`.
- Per-file SHA-256 manifest generated inside release archives.
- Archive-level SHA-256 files generated for `.tar.gz` and `.zip`.

### Hardware acceptance

Live DGX Spark / GB10 acceptance validated the primary local control-plane workflows including model inventory, Hugging Face operations, collision-aware Compose generation/save/archive, authenticated LiteLLM reads, engine ownership detection, local-node display, diagnostics, access controls, and settings.

The following were intentionally not exercised against live production-serving resources before publication:

- launching/stopping a new v2-managed GPU-serving stack while another active inference workload was in use;
- mutating/restarting LiteLLM routing during active client use;
- second-physical-DGX remote-node lifecycle testing.

These remain explicit validation gaps for post-release field testing.
