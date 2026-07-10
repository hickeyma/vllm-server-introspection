# SPDX-License-Identifier: Apache-2.0
"""Unit tests for `GET /plugins/vllm-server-introspection/kv-cache`.

Fast, no engine and no real vLLM/model. `_FakeEngineClient` optionally
exposes `get_kv_cache_config` so both the "new method present" and "older vLLM of
fallback to capacity only" `init_state` paths get exercised.
"""

import asyncio
from argparse import Namespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_server_introspection.kv_cache_plugin import (
    ServerKVCachePlugin,
    _build_group_spec,
    _build_response,
    _capacity_only_from_vllm_config,
)
from vllm_server_introspection.schemas import (
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

# ---------------------------------------------------------------------------
# Helpers — serialized group dicts (mirrors what `get_kv_cache_config` produces)
# ---------------------------------------------------------------------------

_BASE_GROUP = {
    "group_id": 0,
    "layer_names": ["model.layers.0.self_attn", "model.layers.1.self_attn"],
    "block_size": 16,
    "page_size_bytes": 131072,
}


def _full_attention_dict(**overrides) -> dict:
    return {
        **_BASE_GROUP,
        "spec_type": "FullAttentionSpec",
        "num_kv_heads": 8,
        "head_size": 128,
        "head_size_v": 128,
        "dtype": "bfloat16",
        "sliding_window": None,
        "attention_chunk_size": None,
        **overrides,
    }


def _mla_attention_dict(**overrides) -> dict:
    return {
        **_BASE_GROUP,
        "spec_type": "MLAAttentionSpec",
        "num_kv_heads": 8,
        "head_size": 128,
        "head_size_v": 64,
        "dtype": "bfloat16",
        "sliding_window": None,
        "attention_chunk_size": None,
        "cache_dtype_str": "float8_e4m3fn",
        **overrides,
    }


def _sliding_window_dict(**overrides) -> dict:
    return {
        **_BASE_GROUP,
        "spec_type": "SlidingWindowSpec",
        "num_kv_heads": 8,
        "head_size": 128,
        "dtype": "float16",
        "sliding_window": 4096,
        **overrides,
    }


def _chunked_local_attention_dict(**overrides) -> dict:
    return {
        **_BASE_GROUP,
        "spec_type": "ChunkedLocalAttentionSpec",
        "num_kv_heads": 8,
        "head_size": 128,
        "dtype": "float16",
        "attention_chunk_size": 2048,
        **overrides,
    }


def _mamba_dict(**overrides) -> dict:
    return {
        **_BASE_GROUP,
        "spec_type": "MambaSpec",
        "block_size": 1,
        "page_size_bytes": 4096,
        "shapes": [[16, 128], [16, 64]],
        "dtypes": ["float32", "float32"],
        "mamba_type": "mamba2",
        "mamba_cache_mode": "none",
        **overrides,
    }


def _cross_attention_dict(**overrides) -> dict:
    return {
        **_BASE_GROUP,
        "spec_type": "CrossAttentionSpec",
        "num_kv_heads": 8,
        "head_size": 128,
        "dtype": "bfloat16",
        **overrides,
    }


def _sink_full_attention_dict(**overrides) -> dict:
    return {
        **_BASE_GROUP,
        "spec_type": "SinkFullAttentionSpec",
        "num_kv_heads": 8,
        "head_size": 128,
        "head_size_v": 128,
        "dtype": "bfloat16",
        "sliding_window": 2048,
        "attention_chunk_size": None,
        "sink_len": 4,
        **overrides,
    }


def _uniform_type_dict(**overrides) -> dict:
    return {
        **_BASE_GROUP,
        "spec_type": "UniformTypeKVCacheSpecs",
        "layer_specs": [
            {"head_size": 128, "dtype": "bfloat16"},
            {"head_size": 64, "dtype": "bfloat16"},
        ],
        **overrides,
    }


def _kv_cache_data(**overrides) -> dict:
    return {
        "kv_cache_size_tokens": 16384,
        "max_concurrency": 0.5,
        "num_gpu_blocks": 1024,
        "num_cpu_blocks": 256,
        "groups": [_full_attention_dict()],
        **overrides,
    }


class _FakeEngineClient:
    """Minimal stand in exercising `get_kv_cache_config`. Not a real engine.

    `get_kv_cache_config` is only defined as an attribute when
    `has_new_method=True`, so `hasattr` feature detection in the plugin can
    be exercised both ways.
    """

    def __init__(self, kv_cache_data: Any = None, has_new_method: bool = True):
        self._kv_cache_data = kv_cache_data
        self.calls = 0
        if has_new_method:
            self.get_kv_cache_config = self._get_kv_cache_config

    async def _get_kv_cache_config(self) -> Any:
        self.calls += 1
        return self._kv_cache_data


def _make_vllm_config(
    *,
    num_gpu_blocks: int | None = 1024,
    block_size: int | None = 16,
    num_cpu_blocks: int | None = 0,
    max_model_len: int = 4096,
) -> MagicMock:
    cache_cfg = MagicMock()
    cache_cfg.num_gpu_blocks = num_gpu_blocks
    cache_cfg.block_size = block_size
    cache_cfg.num_cpu_blocks = num_cpu_blocks

    model_cfg = MagicMock()
    model_cfg.max_model_len = max_model_len

    vllm_config = MagicMock()
    vllm_config.cache_config = cache_cfg
    vllm_config.model_config = model_cfg
    return vllm_config


def _make_test_app(
    engine_client: "_FakeEngineClient | None", vllm_config: MagicMock | None = None
) -> FastAPI:
    """FastAPI app with the plugin's route + state wired through both phases."""
    app = FastAPI()
    plugin = ServerKVCachePlugin()
    plugin.attach_router(app)
    if vllm_config is not None:
        app.state.vllm_config = vllm_config
    asyncio.run(plugin.init_state(engine_client, app.state, Namespace()))
    return app


# ---------------------------------------------------------------------------
# _build_group_spec unit tests — discriminator dispatch
# ---------------------------------------------------------------------------


class TestBuildGroupSpecDispatch:
    def test_full_attention_spec(self):
        result = _build_group_spec(_full_attention_dict())
        assert isinstance(result, FullAttentionGroupSpec)
        assert result.spec_type == "FullAttentionSpec"

    def test_mla_attention_spec(self):
        result = _build_group_spec(_mla_attention_dict())
        assert isinstance(result, MLAAttentionGroupSpec)

    def test_sliding_window_spec(self):
        result = _build_group_spec(_sliding_window_dict())
        assert isinstance(result, SlidingWindowGroupSpec)

    def test_chunked_local_attention_spec(self):
        result = _build_group_spec(_chunked_local_attention_dict())
        assert isinstance(result, ChunkedLocalAttentionGroupSpec)

    def test_mamba_spec(self):
        result = _build_group_spec(_mamba_dict())
        assert isinstance(result, MambaGroupSpec)

    def test_cross_attention_spec(self):
        result = _build_group_spec(_cross_attention_dict())
        assert isinstance(result, CrossAttentionGroupSpec)

    def test_sink_full_attention_spec(self):
        result = _build_group_spec(_sink_full_attention_dict())
        assert isinstance(result, SinkFullAttentionGroupSpec)

    def test_uniform_type_spec(self):
        result = _build_group_spec(_uniform_type_dict())
        assert isinstance(result, UniformTypeGroupSpec)

    def test_unknown_spec_type_raises_value_error(self):
        group = {**_BASE_GROUP, "spec_type": "UnknownSpec"}
        with pytest.raises(ValueError, match="Unhandled KVCacheSpec type"):
            _build_group_spec(group)

    def test_preserves_base_fields(self):
        result = _build_group_spec(_full_attention_dict(group_id=3))
        assert result.group_id == 3
        assert result.layer_names == _BASE_GROUP["layer_names"]
        assert result.block_size == 16
        assert result.page_size_bytes == 131072


# ---------------------------------------------------------------------------
# _build_response unit tests — pure function
# ---------------------------------------------------------------------------


class TestBuildResponseNullState:
    def test_none_state(self):
        resp = _build_response(None)
        assert isinstance(resp, KVCacheResponse)
        assert resp.kv_cache_size_tokens is None
        assert resp.max_concurrency is None
        assert resp.num_gpu_blocks is None
        assert resp.num_cpu_blocks is None
        assert resp.groups == []


class TestBuildResponseCapacityFields:
    def test_kv_cache_size_tokens(self):
        resp = _build_response(_kv_cache_data(kv_cache_size_tokens=32768))
        assert resp.kv_cache_size_tokens == 32768

    def test_max_concurrency(self):
        resp = _build_response(_kv_cache_data(max_concurrency=1.25))
        assert resp.max_concurrency == pytest.approx(1.25)

    def test_num_gpu_blocks(self):
        resp = _build_response(_kv_cache_data(num_gpu_blocks=2048))
        assert resp.num_gpu_blocks == 2048

    def test_num_cpu_blocks_zero(self):
        resp = _build_response(_kv_cache_data(num_cpu_blocks=0))
        assert resp.num_cpu_blocks == 0

    def test_missing_capacity_fields_resolve_to_none(self):
        resp = _build_response({"groups": []})
        assert resp.kv_cache_size_tokens is None
        assert resp.num_gpu_blocks is None


class TestBuildResponseGroups:
    def test_hybrid_model_two_groups(self):
        groups = [_full_attention_dict(group_id=0), _mamba_dict(group_id=1)]
        resp = _build_response(_kv_cache_data(groups=groups))
        assert len(resp.groups) == 2
        assert isinstance(resp.groups[0], FullAttentionGroupSpec)
        assert isinstance(resp.groups[1], MambaGroupSpec)

    def test_empty_groups_list(self):
        resp = _build_response(_kv_cache_data(groups=[]))
        assert resp.groups == []


# ---------------------------------------------------------------------------
# _capacity_only_from_vllm_config unit tests — older vLLM fallback path
# ---------------------------------------------------------------------------


class TestCapacityOnlyFromVllmConfig:
    def test_computes_kv_cache_size_tokens(self):
        cfg = _make_vllm_config(num_gpu_blocks=1024, block_size=16)
        data = _capacity_only_from_vllm_config(cfg)
        assert data["kv_cache_size_tokens"] == 16384
        assert data["num_gpu_blocks"] == 1024
        assert data["groups"] == []

    def test_computes_max_concurrency(self):
        cfg = _make_vllm_config(num_gpu_blocks=1024, block_size=16, max_model_len=4096)
        data = _capacity_only_from_vllm_config(cfg)
        assert data["max_concurrency"] == pytest.approx(16384 / 4096)

    def test_none_num_gpu_blocks_yields_none_fields(self):
        cfg = _make_vllm_config(num_gpu_blocks=None)
        data = _capacity_only_from_vllm_config(cfg)
        assert data["kv_cache_size_tokens"] is None
        assert data["max_concurrency"] is None
        assert data["num_gpu_blocks"] is None

    def test_num_cpu_blocks_passthrough(self):
        cfg = _make_vllm_config(num_cpu_blocks=512)
        data = _capacity_only_from_vllm_config(cfg)
        assert data["num_cpu_blocks"] == 512


# ---------------------------------------------------------------------------
# HTTP endpoint tests via TestClient
# ---------------------------------------------------------------------------


class TestGetKVCacheEndpoint:
    def test_returns_200_with_data_from_new_method(self):
        fake_client = _FakeEngineClient(_kv_cache_data(), has_new_method=True)
        app = _make_test_app(fake_client)
        with TestClient(app) as client:
            resp = client.get("/plugins/vllm-server-introspection/kv-cache")
        assert resp.status_code == 200
        data = resp.json()
        assert data["num_gpu_blocks"] == 1024
        assert len(data["groups"]) == 1
        assert fake_client.calls == 1

    def test_calls_get_kv_cache_config_once_and_caches(self):
        fake_client = _FakeEngineClient(_kv_cache_data(), has_new_method=True)
        app = _make_test_app(fake_client)
        with TestClient(app) as client:
            client.get("/plugins/vllm-server-introspection/kv-cache")
            client.get("/plugins/vllm-server-introspection/kv-cache")
        assert fake_client.calls == 1

    def test_degrades_to_capacity_only_without_new_method(self):
        fake_client = _FakeEngineClient(has_new_method=False)
        vllm_config = _make_vllm_config(num_gpu_blocks=2048, block_size=16)
        app = _make_test_app(fake_client, vllm_config=vllm_config)
        with TestClient(app) as client:
            resp = client.get("/plugins/vllm-server-introspection/kv-cache")
        assert resp.status_code == 200
        data = resp.json()
        assert data["num_gpu_blocks"] == 2048
        assert data["kv_cache_size_tokens"] == 32768
        assert data["groups"] == []

    def test_engine_client_none_returns_503(self):
        app = _make_test_app(None)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/plugins/vllm-server-introspection/kv-cache")
        assert resp.status_code == 503

    def test_missing_state_returns_500(self):
        app = FastAPI()
        ServerKVCachePlugin().attach_router(app)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/plugins/vllm-server-introspection/kv-cache")
        assert resp.status_code == 500

    def test_content_type_is_json(self):
        fake_client = _FakeEngineClient(_kv_cache_data())
        app = _make_test_app(fake_client)
        with TestClient(app) as client:
            resp = client.get("/plugins/vllm-server-introspection/kv-cache")
        assert "application/json" in resp.headers["content-type"]

    def test_wrong_method_returns_405(self):
        app = _make_test_app(_FakeEngineClient(_kv_cache_data()))
        with TestClient(app) as client:
            resp = client.post("/plugins/vllm-server-introspection/kv-cache")
        assert resp.status_code == 405

    def test_groups_serialized_correctly(self):
        groups = [_full_attention_dict(group_id=0), _mamba_dict(group_id=1)]
        fake_client = _FakeEngineClient(_kv_cache_data(groups=groups))
        app = _make_test_app(fake_client)
        with TestClient(app) as client:
            resp = client.get("/plugins/vllm-server-introspection/kv-cache")
        data = resp.json()["groups"]
        assert data[0]["spec_type"] == "FullAttentionSpec"
        assert data[1]["spec_type"] == "MambaSpec"

    def test_top_level_keys_present(self):
        fake_client = _FakeEngineClient(_kv_cache_data())
        app = _make_test_app(fake_client)
        with TestClient(app) as client:
            resp = client.get("/plugins/vllm-server-introspection/kv-cache")
        assert set(resp.json().keys()) == {
            "kv_cache_size_tokens",
            "max_concurrency",
            "num_gpu_blocks",
            "num_cpu_blocks",
            "groups",
        }


# ---------------------------------------------------------------------------
# Plugin metadata
# ---------------------------------------------------------------------------


class TestPluginMetadata:
    def test_name(self):
        assert ServerKVCachePlugin().name == "vllm_server_introspection_kv_cache"

    def test_required_tasks_excludes_render(self):
        assert "render" not in ServerKVCachePlugin().required_tasks

    def test_required_tasks_includes_generate(self):
        assert "generate" in ServerKVCachePlugin().required_tasks
