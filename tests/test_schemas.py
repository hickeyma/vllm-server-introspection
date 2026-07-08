# SPDX-License-Identifier: Apache-2.0
"""Pydantic model tests for the `/plugins/vllm-server-introspection/config` response schema."""

from vllm_server_introspection.schemas import (
    ComputeCapability,
    DeviceInfo,
    DevicesResponse,
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


def _device_info(**overrides) -> DeviceInfo:
    fields = dict(
        rank=0,
        name="A100-PCIE-40GB",
        total_memory_bytes=42_949_672_960,
        compute_capability=ComputeCapability(major=8, minor=0),
        num_compute_units=108,
    )
    fields.update(overrides)
    return DeviceInfo(**fields)


def test_devices_response_roundtrip():
    original = DevicesResponse(devices=[_device_info(), _device_info(rank=1)])
    restored = DevicesResponse.model_validate(original.model_dump())
    assert restored == original


def test_devices_response_empty():
    assert DevicesResponse(devices=[]).devices == []


def test_device_info_compute_capability_defaults_to_none():
    device = DeviceInfo(rank=0, name="cpu", total_memory_bytes=1024)
    assert device.compute_capability is None
    assert device.num_compute_units is None


def test_device_info_null_capability_serialization():
    dumped = _device_info(compute_capability=None).model_dump()
    assert dumped["compute_capability"] is None


def test_devices_response_json_schema_shape():
    schema = DevicesResponse.model_json_schema()
    assert set(schema["properties"]) == {"devices"}
