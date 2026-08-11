# Compose Builder Notes

## Target-node authority

The selected node owns the facts used to generate its deployment.

For a local deployment, local inventory and host metrics are used.

For a remote deployment, the target agent supplies its own:

- model inventory
- filesystem paths
- platform metrics
- generation result

The central manager must not substitute local model paths into a remote-node deployment.

## Inputs

Compose Builder currently considers:

- model
- engine
- target node
- context length
- host memory reserve
- optimization profile
- bind address
- optional LiteLLM exposure preparation

## Fit categories

The runtime estimate is compared against the selected node's usable memory budget after host reserve.

The UI reports qualitative fit status such as:

- good
- tight
- risk

This is an operational estimate rather than an engine allocation guarantee.

## Runtime estimate limitations

Checkpoint size is a useful baseline, but runtime memory also includes:

- engine/runtime overhead
- CUDA kernels and libraries
- KV cache
- context length
- sequence concurrency
- architecture-specific attention geometry
- graph/workspace allocations
- tokenizer/processor overhead
- quantization implementation details

The current estimate is intentionally conservative but approximate.

MoE and multimodal architectures can diverge substantially from simple parameter-based KV-cache heuristics.

## DGX Spark unified memory

GB10 uses unified memory rather than a conventional isolated VRAM pool.

Do not treat the entire physical-memory figure as safely allocatable model memory.

The builder preserves a configurable host reserve and derives an engine memory fraction from the remaining budget.

## Quantization

Metadata concepts are kept distinct:

- `base_dtype`
- `checkpoint_dtype`
- `quant_method`
- `quant_format`
- `quant_bits_all`
- `quantization_mixed`

A BF16 base model can have an FP4 checkpoint.

An NVFP4 model can use a quantizer implementation such as ModelOpt or compressed-tensors.

A checkpoint can also contain more than one quantized precision.

vLLM normally relies on checkpoint auto-detection.

SGLang receives explicit quantization selection only when the model metadata maps unambiguously to a supported loader mode.

llama.cpp requires GGUF.

## Logical parameters

Packed quantized tensor shapes are not assumed to equal logical model parameter counts.

When Hugging Face metadata declares an upstream base model, the inventory can resolve logical parameter count from that base model.

## Model mounts

Hugging Face cache deployments mount the cache root rather than copying a snapshot.

This preserves the cache's shared blob/snapshot layout.

The default mount is read-only.

Engine-specific writable caches are provided separately when needed.

## Host-port selection

Each engine defines a preferred host port.

Before generation, v2 checks:

1. saved v2 deployment metadata;
2. actual host bind availability.

If the preferred port is unavailable, subsequent ports are considered.

The selected host port is recorded in the plan.

The engine continues to listen on its normal container port.

## Generated-plan integrity

Generated YAML is reviewable and copyable in the browser but intentionally not editable through the normal UI.

The save endpoint relies on trusted generation state rather than accepting arbitrary client Compose documents.

This is a security boundary.

## Saved versus running

Saving a plan writes:

- `compose.yaml`
- `deployment.json`

It does not run the deployment.

Start is a separate operation that executes:

```text
docker compose ... up -d --remove-orphans
```

Stop operates only on v2-managed Compose projects.

## External engine protection

An externally running engine may be visible on the Serving Engines page.

v2 does not stop externally managed services by conventional port number.

The Stop control is enabled only when a running v2-managed profile exists.

## Review before launch

Before starting a newly generated GPU-serving stack, review:

1. selected model
2. image architecture/CUDA compatibility
3. checkpoint quantization support
4. host port
5. bind address
6. model path/mount
7. context length
8. memory fraction
9. sequence concurrency
10. existing GPU/unified-memory workloads

For production systems, validate a new engine/image combination during a maintenance window rather than while another latency-sensitive model workload is active.
