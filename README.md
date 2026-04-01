# DGX Spark Model Manager

A lightweight web UI for managing AI models on the **NVIDIA DGX Spark / HP ZGX Nano G1n** (GB10, 128 GB unified memory). Pull Ollama models, download from HuggingFace, manage LiteLLM routing, and control SGLang — all from one browser tab.

![Python](https://img.shields.io/badge/python-3.10+-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Platform](https://img.shields.io/badge/platform-aarch64%20Ubuntu-orange)

---

## Features

- **Ollama Models** — pull, list, and delete models with live download progress
- **LiteLLM Routing** — one-click wildcard routing so every Ollama model is auto-exposed to all your apps
- **SGLang Engine** — start/stop the SGLang Docker container via configurable launch profiles
- **HuggingFace Download** — download any model from HF Hub directly to the device
- **Live Status Bar** — real-time health indicators for SGLang, Ollama, and LiteLLM
- **Built-in Help Page** — documentation at `/help`

---

## Requirements

| Component | Required | Notes |
|-----------|----------|-------|
| Python 3.10+ | ✅ | Pre-installed on DGX Spark |
| [Ollama](https://ollama.com) | ✅ | Core model management |
| [LiteLLM](https://github.com/BerriAI/litellm) | Optional | Unified API routing |
| [SGLang](https://github.com/sgl-project/sglang) | Optional | Large model inference |
| Docker | Optional | Required for SGLang start/stop |

The app works with just Ollama installed. LiteLLM and SGLang tabs gracefully show offline status if those services aren't running.

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/yourusername/dgx-model-manager
cd dgx-model-manager

# 2. Configure (edit before running)
nano config.json

# 3. Run setup
bash setup.sh

# 4. Open in browser
http://<your-dgx-ip>:8090
```

---

## Configuration

Edit `config.json` before running setup. All fields have sensible defaults.

```json
{
  "app": {
    "host": "0.0.0.0",
    "port": 8090,
    "display_name": "DGX Spark"
  },
  "services": {
    "ollama_base":  "http://127.0.0.1:11434",
    "litellm_base": "http://127.0.0.1:4000",
    "sglang_base":  "http://127.0.0.1:30000"
  },
  "paths": {
    "litellm_config": "~/litellm/litellm_config.yaml",
    "hf_cache":       "~/.cache/huggingface/hub"
  }
}
```

| Field | Default | Description |
|-------|---------|-------------|
| `app.port` | `8090` | Port the app listens on |
| `app.display_name` | `DGX Spark` | Name shown in the UI header |
| `services.ollama_base` | `http://127.0.0.1:11434` | Ollama API URL |
| `services.litellm_base` | `http://127.0.0.1:4000` | LiteLLM proxy URL |
| `services.sglang_base` | `http://127.0.0.1:30000` | SGLang API URL |
| `paths.litellm_config` | `~/litellm/litellm_config.yaml` | Path to your LiteLLM config file |
| `paths.hf_cache` | `~/.cache/huggingface/hub` | HuggingFace model cache directory |

---

## SGLang Profiles

SGLang profiles define startup configurations for large models. Edit `sglang_profiles.json` to match your setup:

```json
[
  {
    "id": "my-model",
    "name": "My Model Name",
    "script": "~/sglang/start_my_model.sh",
    "description": "70B model · ~45 GB VRAM · ~5 min warm-up",
    "vram_gb": 45
  }
]
```

Each profile points to a shell script that launches the SGLang Docker container. Add as many profiles as you have models — they'll all appear in the SGLang tab as selectable options.

### Example SGLang start script

```bash
#!/usr/bin/env bash
sudo docker run --rm --gpus all --ipc=host \
  --name sglang \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 30000:30000 \
  lmsysorg/sglang:nightly-dev-cu13 \
  python3 -m sglang.launch_server \
    --model-path /root/.cache/huggingface/hub/models--your-org--your-model/snapshots/main \
    --host 0.0.0.0 \
    --port 30000 \
    --attention-backend triton \
    --tool-call-parser mistral
```

> **GB10 / SM121A note:** Do not use `--quantization modelopt_fp4` or `--fp4-gemm-backend` — these flags cause PTXAS failures on the GB10 architecture. Always use `--attention-backend triton`.

---

## LiteLLM Wildcard Routing

The **Apply Wildcard** button in the LiteLLM tab adds this single entry to your `litellm_config.yaml`:

```yaml
- model_name: ollama/*
  litellm_params:
    model: ollama/*
    api_base: http://127.0.0.1:11434
```

After this one-time change, every model you pull into Ollama is automatically available to all apps connected to LiteLLM at `:4000` — no further config edits required.

The button also restarts the LiteLLM service automatically. This requires passwordless sudo for `systemctl restart litellm` — the setup script will offer to configure this for you.

---

## Running Without systemd

```bash
# Activate the venv and run directly
source venv/bin/activate
python3 app.py
```

Or with uvicorn for more control:

```bash
venv/bin/uvicorn app:app --host 0.0.0.0 --port 8090 --reload
```

---

## Service Management

```bash
# Status
sudo systemctl status model-manager

# Restart
sudo systemctl restart model-manager

# Logs
sudo journalctl -u model-manager -f

# Disable autostart
sudo systemctl disable model-manager
```

---

## Stack Architecture

```
Your Apps (OpenClaw, Open WebUI, scripts, etc.)
         │
         ▼
   LiteLLM :4000  ──────────────────────────────┐
         │                                       │
         ▼                                       ▼
  SGLang :30000                          Ollama :11434
  (large models,                         (small/medium models,
   NVFP4, MoE)                            hot-swap on demand)
```

---

## Model Recommendations for DGX Spark (128 GB)

### With SGLang running (~31 GB available for Ollama)

| Model | Pull Name | Size | Use Case |
|-------|-----------|------|----------|
| Qwen2.5 Coder 32B | `qwen2.5-coder:32b` | ~20 GB | Coding |
| QwQ 32B | `qwq:32b` | ~20 GB | Reasoning |
| Phi-4 | `phi4` | ~8 GB | General, fast |
| DeepSeek-R1 14B | `deepseek-r1:14b` | ~9 GB | Reasoning |
| Nomic Embed | `nomic-embed-text` | <1 GB | Embeddings/RAG |

### With SGLang stopped (full 128 GB)

| Model | Pull Name | Size | Use Case |
|-------|-----------|------|----------|
| Qwen3 Coder Next | `qwen3-coder-next` | ~54 GB | Agentic coding, 256K context |
| DeepSeek-R1 70B | `deepseek-r1:70b` | ~43 GB | Heavy reasoning |
| Qwen2.5 72B | `qwen2.5:72b` | ~45 GB | General, large |

---

## License

MIT
