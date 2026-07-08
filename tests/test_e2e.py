# SPDX-License-Identifier: Apache-2.0
"""End-to-end test for `GET /plugins/vllm-server-introspection/config` against a real `vllm serve`.

Mirrors `tests/plugins_tests/test_endpoint_plugins.py` in vllm-project/vllm:
`pip install -e` this package, launch a tiny model with `VLLM_PLUGINS` set,
assert a real HTTP 200 + schema.

Unlike the unit tests in `test_config_plugin.py`, this needs `vllm` importable
and downloads model weights over the network, so it is opt-in:

    pip install -e .[test]
    RUN_VLLM_E2E=1 pytest tests/test_e2e.py -v

It is skipped by default (no `vllm` install, no `RUN_VLLM_E2E`) so `pytest`
without extra setup only runs the fast unit tests.
"""

import contextlib
import os
import socket
import subprocess
import sys
import time

import httpx
import pytest

pytest.importorskip("vllm")

if not os.environ.get("RUN_VLLM_E2E"):
    pytest.skip(
        "set RUN_VLLM_E2E=1 to run the real server e2e test (downloads model "
        "weights, launches a subprocess server)",
        allow_module_level=True,
    )

MODEL = "facebook/opt-125m"
STARTUP_TIMEOUT_S = 300


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def _running_server(*, allowlist_plugin: bool):
    port = _free_port()
    env = dict(os.environ)
    if allowlist_plugin:
        env["VLLM_PLUGINS"] = "server_config"
    else:
        env.pop("VLLM_PLUGINS", None)
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            MODEL,
            "--port",
            str(port),
        ],
        env=env,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"server process exited early with {proc.returncode}")
            try:
                resp = httpx.get(f"{base_url}/health", timeout=5)
                if resp.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(1)
        else:
            raise TimeoutError("server did not become healthy in time")
        yield base_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


def test_server_config_endpoint_returns_200_with_valid_schema():
    with _running_server(allowlist_plugin=True) as base_url:
        resp = httpx.get(f"{base_url}/plugins/vllm-server-introspection/config", timeout=10)

    assert resp.status_code == 200
    data = resp.json()
    assert set(data.keys()) == {
        "model",
        "kv_cache",
        "scheduler",
        "parallelism",
        "features",
    }
    assert data["model"]["name"]
    assert data["parallelism"]["tensor_parallel_size"] == 1


def test_server_config_not_attached_without_allowlist():
    """No VLLM_PLUGINS set -> route must not exist (strict allowlist)."""
    with _running_server(allowlist_plugin=False) as base_url:
        resp = httpx.get(f"{base_url}/plugins/vllm-server-introspection/config", timeout=10)

    assert resp.status_code == 404
