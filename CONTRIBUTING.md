# Contributing to DGX Model Manager

Contributions are welcome. DGX Model Manager controls privileged local AI infrastructure, so changes should preserve the project's security and upgrade guarantees.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
DMM_DEMO_MODE=1 python app.py
```

Demo mode uses synthetic data and is intended only for development/screenshots. Do not use it as a deployment mode.

## Before opening a pull request

```bash
python3 -m compileall -q app.py agent.py dgx_manager scripts tools
node --check static/app.js
bash -n setup.sh setup-agent.sh scripts/promote_v2.sh
pytest -q
python3 scripts/validate_release.py
```

## Security-sensitive changes

Please treat these areas as security boundaries:

- authentication/session/CSRF logic;
- service-target URL validation/SSRF controls;
- filesystem deletion and custom model directories;
- Docker Compose lifecycle operations;
- legacy script execution;
- LiteLLM configuration and secret redaction;
- remote-node tokens and TLS verification;
- systemd/sudo integration.

Do not weaken authentication because a deployment is assumed to be on a private LAN. The LAN is explicitly considered untrusted.

## Public test fixtures

Repository fixtures, docs, screenshots, test data, and examples must use synthetic identities and documentation-safe addresses. Never commit runtime config, secrets, real home paths, real hostnames, or private infrastructure details.

## Engine catalog changes

Inference-engine CLI flags and container tags change frequently. Changes to `engine_catalog.yaml` or Compose generation should cite/check current upstream project documentation and include generator tests.
