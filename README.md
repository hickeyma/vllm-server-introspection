# vllm-server-introspection

`GET /plugins/vllm-server-introspection/*` introspection endpoints for [vLLM](https://github.com/vllm-project/vllm), integrated as [endpoint plugins](https://github.com/vllm-project/vllm/pull/47454) rather than core code.

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

## Install

```bash
pip install -e .
```

## Enable

Endpoint plugins load only when explicitly named in `VLLM_PLUGINS` (off by default):

```bash
VLLM_PLUGINS=vllm_server_introspection_config vllm serve <model>
```

```bash
curl http://localhost:8000/plugins/vllm-server-introspection/config
```

## Test

```bash
pip install -e .[test]

# fast unit tests (schema + FastAPI route, no engine, no real vLLM/model)
pytest tests/ -v

# real server e2e (downloads model weights, launches a subprocess server)
RUN_VLLM_E2E=1 pytest tests/test_e2e.py -v
```
