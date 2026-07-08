# SPDX-License-Identifier: Apache-2.0
"""Pydantic model tests for the `/plugins/vllm-server-introspection/config` response schema."""

from vllm_server_introspection.schemas import (
    FeaturesInfo,
    KVCacheInfo,
    ModelInfo,
    ParallelismInfo,
    SchedulerInfo,
    ServerConfigResponse,
)


def _server_config_response() -> ServerConfigResponse:
    return ServerConfigResponse(
        model=ModelInfo(
            name="llama",
            served_names=["llama"],
            dtype="bfloat16",
            quantization=None,
            max_model_len=4096,
        ),
        kv_cache=KVCacheInfo(
            gpu_memory_utilization=0.9,
            dtype="bfloat16",
            enable_prefix_caching=True,
        ),
        scheduler=SchedulerInfo(
            max_num_seqs=64,
            max_num_batched_tokens=1024,
            enable_chunked_prefill=True,
            policy="fcfs",
        ),
        parallelism=ParallelismInfo(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            data_parallel_rank=0,
        ),
        features=FeaturesInfo(
            speculative_decoding=False,
            lora=False,
            hma=True,
        ),
    )


def test_server_config_response_roundtrip():
    original = _server_config_response()
    restored = ServerConfigResponse.model_validate(original.model_dump())
    assert restored == original


def test_quantization_null_serialization():
    dumped = _server_config_response().model_dump()
    assert dumped["model"]["quantization"] is None


def test_quantization_string_serialization():
    resp = _server_config_response()
    resp.model.quantization = "gptq"
    assert resp.model_dump()["model"]["quantization"] == "gptq"


def test_features_info_all_bool():
    feat = FeaturesInfo(speculative_decoding=True, lora=False, hma=True)
    assert feat.model_dump() == {
        "speculative_decoding": True,
        "lora": False,
        "hma": True,
    }


def test_server_config_response_json_schema_has_required_sections():
    schema = ServerConfigResponse.model_json_schema()
    assert set(schema["properties"]) == {
        "model",
        "kv_cache",
        "scheduler",
        "parallelism",
        "features",
    }
