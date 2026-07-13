# SPDX-License-Identifier: Apache-2.0
"""Pydantic response models for the `/plugins/vllm-server-introspection/*` introspection endpoints.

Shared across all endpoint plugins in this package.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    name: str
    served_names: list[str]
    dtype: str
    quantization: str | None
    max_model_len: int


class KVCacheInfo(BaseModel):
    gpu_memory_utilization: float
    dtype: str
    enable_prefix_caching: bool


class SchedulerInfo(BaseModel):
    max_num_seqs: int
    max_num_batched_tokens: int
    enable_chunked_prefill: bool
    policy: str


class ParallelismInfo(BaseModel):
    tensor_parallel_size: int
    pipeline_parallel_size: int
    data_parallel_size: int
    data_parallel_rank: int


class FeaturesInfo(BaseModel):
    speculative_decoding: bool
    lora: bool
    hma: bool


class KVTransferInfo(BaseModel):
    kv_connector: str | None
    kv_role: str | None
    kv_connector_module_path: str | None = None
    kv_buffer_device: str | None = None
    kv_buffer_size: float | None = None
    kv_ip: str | None = None
    kv_port: int | None = None
    kv_parallel_size: int | None = None
    kv_rank: int | None = None
    engine_id: str | None = None
    extra_config: dict = Field(default_factory=dict)
    # NIXL side channel base endpoint (env derived only when connector is
    # NixlConnector). This is the base host/port. NixlConnector derives the
    # actual per rank port as base_port + rank_offset inside the worker.
    nixl_side_channel_host: str | None = None
    nixl_side_channel_port: int | None = None


class ServerConfigResponse(BaseModel):
    model: ModelInfo
    kv_cache: KVCacheInfo
    scheduler: SchedulerInfo
    parallelism: ParallelismInfo
    features: FeaturesInfo
    kv_transfer: KVTransferInfo | None = None


class ComputeCapability(BaseModel):
    major: int
    minor: int


class DeviceInfo(BaseModel):
    rank: int
    name: str
    total_memory_bytes: int
    compute_capability: ComputeCapability | None = None
    num_compute_units: int | None = None


class DevicesResponse(BaseModel):
    devices: list[DeviceInfo]


# ---------------------------------------------------------------------------
# `/plugins/vllm-server-introspection/kv-cache`: group specs are a
# discriminated union keyed on `spec_type` which are ported from the `KVCacheSpec`
# subclass names `EngineClient.get_kv_cache_config()` that serializes them as
# ---------------------------------------------------------------------------


class _KVCacheGroupBase(BaseModel):
    group_id: int
    layer_names: list[str]
    block_size: int
    page_size_bytes: int


class _FullAttentionBaseSpec(_KVCacheGroupBase):
    num_kv_heads: int
    head_size: int
    head_size_v: int
    dtype: str
    sliding_window: int | None = None
    attention_chunk_size: int | None = None


class FullAttentionGroupSpec(_FullAttentionBaseSpec):
    spec_type: Literal["FullAttentionSpec"] = "FullAttentionSpec"


class MLAAttentionGroupSpec(_FullAttentionBaseSpec):
    spec_type: Literal["MLAAttentionSpec"] = "MLAAttentionSpec"
    cache_dtype_str: str | None = None


class SlidingWindowGroupSpec(_KVCacheGroupBase):
    spec_type: Literal["SlidingWindowSpec"] = "SlidingWindowSpec"
    num_kv_heads: int
    head_size: int
    dtype: str
    sliding_window: int


class ChunkedLocalAttentionGroupSpec(_KVCacheGroupBase):
    spec_type: Literal["ChunkedLocalAttentionSpec"] = "ChunkedLocalAttentionSpec"
    num_kv_heads: int
    head_size: int
    dtype: str
    attention_chunk_size: int


class MambaGroupSpec(_KVCacheGroupBase):
    spec_type: Literal["MambaSpec"] = "MambaSpec"
    shapes: list[list[int]]
    dtypes: list[str]
    mamba_type: str
    mamba_cache_mode: str


class CrossAttentionGroupSpec(_KVCacheGroupBase):
    spec_type: Literal["CrossAttentionSpec"] = "CrossAttentionSpec"
    num_kv_heads: int
    head_size: int
    dtype: str


class SinkFullAttentionGroupSpec(_FullAttentionBaseSpec):
    spec_type: Literal["SinkFullAttentionSpec"] = "SinkFullAttentionSpec"
    sink_len: int | None = None


class UniformTypeGroupSpec(_KVCacheGroupBase):
    spec_type: Literal["UniformTypeKVCacheSpecs"] = "UniformTypeKVCacheSpecs"
    layer_specs: list[dict]


KVCacheGroupSpec = Annotated[
    FullAttentionGroupSpec
    | MLAAttentionGroupSpec
    | SlidingWindowGroupSpec
    | ChunkedLocalAttentionGroupSpec
    | MambaGroupSpec
    | CrossAttentionGroupSpec
    | SinkFullAttentionGroupSpec
    | UniformTypeGroupSpec,
    Field(discriminator="spec_type"),
]


class KVCacheResponse(BaseModel):
    kv_cache_size_tokens: int | None = None
    max_concurrency: float | None = None
    num_gpu_blocks: int | None = None
    num_cpu_blocks: int | None = None
    groups: list[KVCacheGroupSpec] = Field(default_factory=list)
