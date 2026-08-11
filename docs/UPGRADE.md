# v2 Upgrade / Side-by-Side Test Procedure

v2 is designed to coexist with an existing installation during acceptance.

## 1. Install separately

Use a separate checkout/extraction directory.

Run:

```bash
bash setup.sh
```

Default coexistence resources are:

```text
service: model-manager-v2.service
HTTPS:   8091
config:  ~/.config/dgx-model-manager-v2/config.json
data:    ~/.local/share/dgx-model-manager-v2
```

The installer does not replace an existing `model-manager.service`.

## 2. Bootstrap administrator

Use the one-time host-local bootstrap token printed by setup.

Confirm:

- HTTPS access
- administrator creation
- login
- logout/login
- dashboard access

## 3. Confirm existing model inventory

Validate existing Hugging Face model paths and optional custom directories.

No checkpoint migration should be required for the standard Hugging Face cache.

## 4. Configure optional LiteLLM integration

If LiteLLM requires API authentication, setup can install an isolated systemd credential containing only:

```text
LITELLM_MASTER_KEY
```

The broader LiteLLM environment does not need to be readable by the Model Manager service user.

If v2 should modify routes and restart LiteLLM, separately opt into the restricted sudo restart capability.

These are independent permissions.

## 5. Generate before launching

Use Compose Builder to generate a representative model deployment.

Review:

- engine image
- host/container ports
- model mount
- served-model name
- context
- memory estimate
- quantization decision
- GPU reservation

Saving the deployment is safe to perform separately from starting it.

## 6. Validate Compose syntax

Before first launch:

```bash
docker compose -f <saved-stack>/compose.yaml config -q
```

## 7. Preserve active workloads

If an existing production vLLM, ComfyUI, or other inference service is in use, do not launch a competing GPU-heavy deployment merely to complete an acceptance checklist.

v2 can detect external services without claiming ownership.

Externally managed workloads cannot be stopped from the Serving Engines Stop control.

## 8. Test managed lifecycle during a safe window

When capacity and client impact permit:

1. Start a v2-managed deployment.
2. Verify engine logs.
3. Verify health/model endpoint.
4. Verify LiteLLM routing if desired.
5. Stop the deployment.
6. Restart it.
7. Archive it.

## 9. Multi-node

Additional DGX Spark systems require the restricted node agent.

A second physical node should be validated independently before relying on remote lifecycle operations in production.

## Promotion boundary

Installing and validating v2 does not require promotion.

Promotion is a separate administrative operation.

Review:

```bash
python3 scripts/promote_v2.sh --help
```

Use the dry-run behavior first.

Promotion should occur only after:

- authentication is confirmed;
- required service integrations work;
- important models are visible;
- generated deployments are acceptable;
- production clients have a rollback plan.

## Rollback

Preserve the old application/service until promotion has been accepted.

A side-by-side v2 installation uses separate runtime state, so stopping/removing v2 should not require deleting or rewriting existing model checkpoints.
