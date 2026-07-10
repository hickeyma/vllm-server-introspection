# vllm-server-introspection

`GET /plugins/vllm-server-introspection/*` introspection endpoints for [vLLM](https://github.com/vllm-project/vllm), integrated as [endpoint plugins](https://docs.vllm.ai/en/latest/design/endpoint_plugins/) rather than core code.

## `GET /plugins/vllm-server-introspection/config`

Operator supplied, config time values of how the server was launched. Nothing profiled or derived from model internals. Requires no engine, so it also works on the CPU only render server.

```jsonc
{
  "model": {
    "name": "llama3",
    "served_names": ["llama3"],
    "dtype": "bfloat16",
    "quantization": null,
    "max_model_len": 8192
  },
  "kv_cache": {
    "gpu_memory_utilization": 0.9,
    "dtype": "bfloat16",
    "enable_prefix_caching": true
  },
  "scheduler": {
    "max_num_seqs": 256,
    "max_num_batched_tokens": 8192,
    "enable_chunked_prefill": true,
    "policy": "fcfs"
  },
  "parallelism": {
    "tensor_parallel_size": 2,
    "pipeline_parallel_size": 1,
    "data_parallel_size": 1,
    "data_parallel_rank": 0
  },
  "features": {
    "speculative_decoding": false,
    "lora": true,
    "hma": true
  }
}
```

## `GET /plugins/vllm-server-introspection/devices`

Per rank hardware properties, gathered once at startup via `collective_rpc` and cached for the server's lifetime. Requires an engine (`503` on the CPU only render server) and the worker side `get_device_properties` method installed by `device_worker_ext.DeviceInfoWorkerExtension` via `--worker-extension-cls`.

```jsonc
{
  "devices": [
    {
      "rank": 0,
      "name": "A100-PCIE-40GB",
      "total_memory_bytes": 42949672960,
      "compute_capability": { "major": 8, "minor": 0 },
      "num_compute_units": 108
    }
  ]
}
```

## `GET /plugins/vllm-server-introspection/kv-cache`

Post profiling KV cache capacity and attention group structure, gathered once at startup and cached for the server's lifetime. Requires an engine (`503` on the CPU only render server). `groups` is a discriminated union keyed on `spec_type`, correctly representing hybrid models with multiple attention/mamba groups.

Against a vLLM build without that method, this plugin falls back to capacity only fields read directly off `vllm_config.cache_config` and omits `groups`.

```jsonc
{
  "kv_cache_size_tokens": 393216,
  "max_concurrency": 48.0,
  "num_gpu_blocks": 24576,
  "num_cpu_blocks": 0,
  "groups": [
    {
      "group_id": 0,
      "spec_type": "FullAttentionSpec",
      "layer_names": ["model.layers.0.self_attn", "..."],
      "block_size": 16,
      "page_size_bytes": 131072,
      "num_kv_heads": 8,
      "head_size": 128,
      "head_size_v": 128,
      "dtype": "bfloat16",
      "sliding_window": null,
      "attention_chunk_size": null
    }
  ]
}
```

## Install

```bash
pip install -e .
```

## Enable

Endpoint plugins load only when explicitly named in `VLLM_PLUGINS` (off by default):

```bash
# config only
VLLM_PLUGINS=vllm_server_introspection_config vllm serve <model>

# devices only
VLLM_PLUGINS=vllm_server_introspection_devices vllm serve <model> \
  --worker-extension-cls vllm_server_introspection.device_worker_ext.DeviceInfoWorkerExtension

# kv cache only
VLLM_PLUGINS=vllm_server_introspection_kv_cache vllm serve <model>

# all three
VLLM_PLUGINS=vllm_server_introspection_config,vllm_server_introspection_devices,vllm_server_introspection_kv_cache \
  vllm serve <model> \
  --worker-extension-cls vllm_server_introspection.device_worker_ext.DeviceInfoWorkerExtension
```

```bash
curl http://localhost:8000/plugins/vllm-server-introspection/config
curl http://localhost:8000/plugins/vllm-server-introspection/devices
curl http://localhost:8000/plugins/vllm-server-introspection/kv-cache
```

## Test

```bash
pip install -e .[test]

# fast unit tests (schema + FastAPI route, no engine, no real vLLM/model)
pytest tests/ -v

# real server e2e (downloads model weights, launches a subprocess server)
RUN_VLLM_E2E=1 pytest tests/test_e2e.py -v
```
