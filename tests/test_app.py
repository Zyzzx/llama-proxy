"""End-to-end tests: proxy app against fake llama-server backends.

The fakes are real uvicorn servers on localhost ports so the proxy's
httpx client exercises genuine networking (as in production).
"""
import asyncio
import json
import socket
import threading
import time

import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from starlette.testclient import TestClient


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def make_fake_backend(model_id: str, busy: int = 1, delay: float = 0.0):
    app = FastAPI()
    state = {"busy": busy}

    @app.get("/metrics")
    async def metrics():
        # real llama-server serves plain text; FastAPI would otherwise
        # JSON-encode a bare str return value. New llama.cpp metric names;
        # 'busy' is exposed as requests_processing (deferred stays 0).
        return PlainTextResponse(
            "# HELP llamacpp:requests_processing Number of requests processing\n"
            "# TYPE llamacpp:requests_processing gauge\n"
            f"llamacpp:requests_processing {state['busy']}\n"
            "# HELP llamacpp:requests_deferred Number of requests deferred\n"
            "# TYPE llamacpp:requests_deferred gauge\n"
            "llamacpp:requests_deferred 0\n"
        )

    @app.get("/v1/models")
    async def models():
        return {"object": "list",
                "data": [{"id": model_id, "object": "model"}]}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        payload = await request.json()
        if delay:
            await asyncio.sleep(delay)
        if payload.get("stream"):
            async def gen():
                for i, tok in enumerate(["Hel", "lo"]):
                    chunk = {
                        "id": "cmpl-1", "object": "chat.completion.chunk",
                        "choices": [{"index": 0,
                                     "delta": {"content": tok},
                                     "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield 'data: [DONE]\n\n'
            return StreamingResponse(gen(), media_type="text/event-stream")
        return {"id": "cmpl-1", "object": "chat.completion",
                "choices": [{"message": {"role": "assistant", "content": "Hello"}}],
                "model": model_id}

    return app, state


def run_fake(app):
    port = free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    # wait for startup
    deadline = time.time() + 5
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return port, server
        except OSError:
            time.sleep(0.05)
    raise RuntimeError("fake backend did not start")


@pytest.fixture()
def backend_a():
    app, state = make_fake_backend("model-a")
    port, server = run_fake(app)
    yield f"http://127.0.0.1:{port}", state
    server.should_exit = True


@pytest.fixture()
def backend_b():
    app, state = make_fake_backend("model-b")
    port, server = run_fake(app)
    yield f"http://127.0.0.1:{port}", state
    server.should_exit = True


def make_proxy(backend_a, backend_b, **server_kw):
    from llm_proxy.config import AppConfig
    from llm_proxy.app import create_app

    cfg = AppConfig(
        server={"poll_interval": 0.2, "stale_after": 2.0, **server_kw},
        backends=[
            {"name": "a", "url": backend_a[0], "slots": 4},
            {"name": "b", "url": backend_b[0], "slots": 4},
        ],
    )
    return create_app(cfg)


def test_health_and_ready(backend_a, backend_b):
    app = make_proxy(backend_a, backend_b)
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200 and r.json()["status"] == "ok"
        # wait for the first poll cycle
        time.sleep(0.5)
        r = client.get("/ready")
        assert r.status_code == 200
        assert r.json()["fresh_backends"] == 2


def test_non_streaming_passthrough(backend_a, backend_b):
    app = make_proxy(backend_a, backend_b)
    with TestClient(app) as client:
        time.sleep(0.5)
        r = client.post("/v1/chat/completions",
                        json={"model": "model-a", "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 200
        body = r.json()
        assert body["choices"][0]["message"]["content"] == "Hello"
        assert r.headers["x-proxy-backend"] in ("a", "b")
        assert "x-request-id" in r.headers


def test_streaming_passthrough(backend_a, backend_b):
    app = make_proxy(backend_a, backend_b)
    with TestClient(app) as client:
        time.sleep(0.5)
        with client.stream(
            "POST", "/v1/chat/completions",
            json={"model": "model-a", "stream": True,
                  "messages": [{"role": "user", "content": "hi"}]},
        ) as r:
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("text/event-stream")
            text = "".join(chunk for chunk in r.iter_text())
        assert "data: " in text
        assert "data: [DONE]" in text
        assert "Hel" in text and "lo" in text


def test_no_v1_prefix_fallback(backend_a, backend_b):
    """Clients that omit /v1 get rewritten for OpenAI paths."""
    app = make_proxy(backend_a, backend_b)
    with TestClient(app) as client:
        time.sleep(0.5)
        r = client.get("/models")
        assert r.status_code == 200
        ids = {m["id"] for m in r.json()["data"]}
        assert ids == {"model-a", "model-b"}
        r = client.post("/chat/completions",
                        json={"model": "model-a", "messages": []})
        assert r.status_code == 200
        assert r.json()["choices"][0]["message"]["content"] == "Hello"


def test_model_pin_routes_to_matching_backend(backend_a, backend_b):
    """The OpenAI ``model`` field pins the backend by exact id match."""
    app = make_proxy(backend_a, backend_b)
    with TestClient(app) as client:
        time.sleep(0.6)  # let the poller discover each backend's model id
        r = client.post("/v1/chat/completions",
                        json={"model": "model-a", "messages": []})
        assert r.status_code == 200
        assert r.headers["x-proxy-backend"] == "a"
        r = client.post("/v1/chat/completions",
                        json={"model": "model-b", "messages": []})
        assert r.status_code == 200
        assert r.headers["x-proxy-backend"] == "b"


def test_model_unknown_falls_back_to_normal_routing(backend_a, backend_b):
    """A model id no backend serves does not pin; normal routing applies."""
    app = make_proxy(backend_a, backend_b)
    with TestClient(app) as client:
        time.sleep(0.6)
        r = client.post("/v1/chat/completions",
                        json={"model": "does-not-exist", "messages": []})
        assert r.status_code == 200
        assert r.headers["x-proxy-backend"] in ("a", "b")


def test_models_merged(backend_a, backend_b):
    app = make_proxy(backend_a, backend_b)
    with TestClient(app) as client:
        time.sleep(0.6)
        r = client.get("/v1/models")
        assert r.status_code == 200
        ids = {m["id"] for m in r.json()["data"]}
        assert ids == {"model-a", "model-b"}


def test_routing_prefers_less_busy(backend_a, backend_b):
    # backend b is busier -> all traffic should go to a
    backend_b[1]["busy"] = 4
    app = make_proxy(backend_a, backend_b)
    with TestClient(app) as client:
        time.sleep(0.5)
        backends = set()
        for _ in range(3):
            r = client.post("/v1/chat/completions",
                            json={"model": "x", "messages": []})
            assert r.status_code == 200
            backends.add(r.headers["x-proxy-backend"])
        assert backends == {"a"}


def test_burst_spreads_across_backends():
    """A burst of concurrent requests must spread across backends even
    while the metrics sample is stale — in-flight tracking handles it."""
    import concurrent.futures
    from collections import Counter

    from llm_proxy.config import AppConfig
    from llm_proxy.app import create_app

    # slow fakes so the 4 requests genuinely overlap in flight
    app_a, _ = make_fake_backend("model-a", busy=0, delay=0.6)
    pa, sa = run_fake(app_a)
    app_b, _ = make_fake_backend("model-b", busy=0, delay=0.6)
    pb, sb = run_fake(app_b)
    try:
        cfg = AppConfig(
            server={"poll_interval": 0.2, "stale_after": 2.0},
            backends=[
                {"name": "a", "url": f"http://127.0.0.1:{pa}", "slots": 4},
                {"name": "b", "url": f"http://127.0.0.1:{pb}", "slots": 4},
            ],
        )
        with TestClient(create_app(cfg)) as client:
            time.sleep(0.5)

            def fire(_):
                r = client.post("/v1/chat/completions",
                                json={"model": "x", "messages": []})
                assert r.status_code == 200
                return r.headers["x-proxy-backend"]

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
                backends = list(ex.map(fire, range(4)))

            counts = Counter(backends)
            assert counts == Counter({"a": 2, "b": 2}), counts
    finally:
        sa.should_exit = True
        sb.should_exit = True


def test_auth_required(backend_a, backend_b):
    app = make_proxy(backend_a, backend_b, api_keys=["sekret"])
    with TestClient(app) as client:
        time.sleep(0.5)
        r = client.post("/v1/chat/completions", json={"model": "x", "messages": []})
        assert r.status_code == 401
        r = client.post(
            "/v1/chat/completions",
            json={"model": "x", "messages": []},
            headers={"authorization": "Bearer sekret"},
        )
        assert r.status_code == 200
        # health endpoints stay open even with auth enabled (LB probes)
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code in (200, 503)
        # ...but /proxy/metrics leaks topology and stays protected
        assert client.get("/proxy/metrics").status_code == 401


def test_all_backends_down_returns_503():
    # point at dead ports: poller marks them stale after the first cycle
    app = make_proxy(("http://127.0.0.1:1", None), ("http://127.0.0.1:2", None))
    with TestClient(app) as client:
        time.sleep(0.5)
        r = client.get("/ready")
        assert r.status_code == 503
        r = client.post("/v1/chat/completions", json={"model": "x", "messages": []})
        assert r.status_code == 503
        assert r.json()["error"]["type"] == "proxy_error"
