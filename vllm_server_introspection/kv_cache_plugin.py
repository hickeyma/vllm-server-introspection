# SPDX-License-Identifier: Apache-2.0
"""`vllm.endpoint_plugins` entry point: `GET /plugins/vllm-server-introspection/kv-cache`.

Post profiling KV cache capacity plus attention group structure, sourced from
`engine_client.get_kv_cache_config()` once at startup and cached for the
server's lifetime (immutable once profiling has run).

`get_kv_cache_config` is not yet part of a released `EngineClient` — it is
proposed upstream in vllm-project/vllm#43793, which this plugin's group
schema and response shape are ported from (`spec_type` values are the
`KVCacheSpec` subclass names that PR serializes groups under). Feature
detected via `hasattr` so this package still installs against a vLLM build
that predates that method. Against such a build, `init_state` fallback to
capacity only fields read directly off `vllm_config.cache_config` and omits
`groups` entirely.

`required_tasks` excludes the `render` frontend task. This plugin needs an
engine (there is nothing to introspect on the CPU only render server).
"""

from argparse import Namespace
from typing import TYPE_CHECKING

from fastapi import APIRouter, FastAPI, HTTPException, Request
from starlette.datastructures import State
from vllm.tasks import GENERATION_TASKS, POOLING_TASKS

from .schemas import (
    ChunkedLocalAttentionGroupSpec,
    CrossAttentionGroupSpec,
    FullAttentionGroupSpec,
    KVCacheResponse,
    MambaGroupSpec,
    MLAAttentionGroupSpec,
    SinkFullAttentionGroupSpec,
    SlidingWindowGroupSpec,
    UniformTypeGroupSpec,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.engine.protocol import EngineClient

_UNSET = object()

_SPEC_TYPE_TO_MODEL: dict[str, type] = {
    "FullAttentionSpec": FullAttentionGroupSpec,
    "MLAAttentionSpec": MLAAttentionGroupSpec,
    "SlidingWindowSpec": SlidingWindowGroupSpec,
    "ChunkedLocalAttentionSpec": ChunkedLocalAttentionGroupSpec,
    "MambaSpec": MambaGroupSpec,
    "CrossAttentionSpec": CrossAttentionGroupSpec,
    "SinkFullAttentionSpec": SinkFullAttentionGroupSpec,
    "UniformTypeKVCacheSpecs": UniformTypeGroupSpec,
}


def _build_group_spec(group: dict):
    spec_type = group["spec_type"]
    model_cls = _SPEC_TYPE_TO_MODEL.get(spec_type)
    if model_cls is None:
        raise ValueError(f"Unhandled KVCacheSpec type: {spec_type!r}")
    return model_cls(**group)


def _build_response(kv_cache_data: dict | None) -> KVCacheResponse:
    if kv_cache_data is None:
        return KVCacheResponse()
    return KVCacheResponse(
        kv_cache_size_tokens=kv_cache_data.get("kv_cache_size_tokens"),
        max_concurrency=kv_cache_data.get("max_concurrency"),
        num_gpu_blocks=kv_cache_data.get("num_gpu_blocks"),
        num_cpu_blocks=kv_cache_data.get("num_cpu_blocks"),
        groups=[_build_group_spec(g) for g in kv_cache_data.get("groups", [])],
    )


def _capacity_only_from_vllm_config(vllm_config: "VllmConfig") -> dict:
    # Fallback path for a vLLM build without `get_kv_cache_config` (pre PR
    # #43793). `cache_config.num_gpu_blocks` / `.block_size` are already
    # post profiling values by the time `init_state` runs. EngineCore
    # mutates the shared config in place during `_initialize_kv_caches`
    # just without per group structure or the `max_concurrency` convenience
    # field the newer method also computes from the scheduler's config.
    cache_cfg = vllm_config.cache_config
    num_gpu_blocks = cache_cfg.num_gpu_blocks
    block_size = cache_cfg.block_size
    kv_cache_size_tokens = (
        num_gpu_blocks * block_size
        if num_gpu_blocks is not None and block_size is not None
        else None
    )
    max_model_len = vllm_config.model_config.max_model_len
    max_concurrency = (
        kv_cache_size_tokens / max_model_len
        if kv_cache_size_tokens is not None and max_model_len
        else None
    )
    return {
        "kv_cache_size_tokens": kv_cache_size_tokens,
        "max_concurrency": max_concurrency,
        "num_gpu_blocks": num_gpu_blocks,
        "num_cpu_blocks": cache_cfg.num_cpu_blocks,
        "groups": [],
    }


class ServerKVCachePlugin:
    name = "vllm_server_introspection_kv_cache"
    required_tasks: tuple[str, ...] | None = GENERATION_TASKS + POOLING_TASKS

    def attach_router(self, app: FastAPI) -> None:
        router = APIRouter()

        @router.get(
            "/plugins/vllm-server-introspection/kv-cache",
            response_model=KVCacheResponse,
        )
        async def get_kv_cache(raw_request: Request) -> KVCacheResponse:
            response = getattr(
                raw_request.app.state, "server_kv_cache_response", _UNSET
            )
            if response is _UNSET:
                raise HTTPException(
                    status_code=500,
                    detail="server_kv_cache plugin state was never initialized",
                )
            if response is None:
                raise HTTPException(
                    status_code=503,
                    detail="kv-cache requires an engine, which this server "
                    "does not have",
                )
            return response

        app.include_router(router)

    async def init_state(
        self, engine_client: "EngineClient | None", state: State, args: Namespace
    ) -> None:
        if engine_client is None:
            state.server_kv_cache_response = None
            return
        if hasattr(engine_client, "get_kv_cache_config"):
            kv_cache_data = await engine_client.get_kv_cache_config()
        else:
            kv_cache_data = _capacity_only_from_vllm_config(state.vllm_config)
        state.server_kv_cache_response = _build_response(kv_cache_data)
