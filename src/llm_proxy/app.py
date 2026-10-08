"""FastAPI application: transparent OpenAI-compatible passthrough.

Clients see an ordinary llama-server endpoint. The only differences are:
the host/port, and that /v1/models is a merged union across backends when
``server.merge_models`` is true.
"""
from __future__ import annotations

import asyncio
import ijson
import apyio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .config import AppConfig
from .router import BackendFull, NoBackendAvailable, Router
from .state import BackendRegistry, MetricsPoller

logger = logging.getLogger("llm_proxy.app")

HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
    "content-length", "host",
}

# Paths that are unambiguously OpenAI API (and therefore belong under
# /v1/ on the backend) even when the client omits the prefix. Root-level
# llama.cpp paths like /metrics, /slots, /health are intentionally NOT
# rewritten — they live outside /v1 on the backend.
import re

V1_REWRITE_PREFIXES = re.compile(
    r"^(models|chat/completions|embeddings|moderations"
    r"|audio/(transcriptions|translations|speech)"
    r"|images/(generations|edits)"
    r"|files|fine_tuning|assistants|threads|runs)(/.*)?$"
)


class NoBackendAvailableError(Exception):
    """Raised internally; mapped to a 503 by the exception handler."""


# ---------------------------------------------------------------------------
# application factory
# ---------------------------------------------------------------------------

def create_app(cfg: AppConfig) -> FastAPI:
    proxy = ProxyApp(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await proxy.start()
        try:
            yield
        finally:
            await proxy.stop()

    app = FastAPI(
        title="llm-proxy",
        version="0.1.0",
        lifespan=lifespan,
        docs_url=None,   # no docs in production
        redoc_url=None,
        openapi_url=None,
    )
    app.state.proxy = proxy

    server_cfg = cfg.server

    @app.middleware("http")
    async def auth_and_log(request: Request, call_next):
        if server_cfg.api_keys:
            # health endpoints stay open: load balancers and container
            # HEALTHCHECKs probe them without (and must not need) a key.
            # /proxy/metrics stays protected — it leaks backend topology.
            if request.url.path not in ("/health", "/ready"):
                auth = request.headers.get("authorization", "")
                token = auth[7:] if auth.lower().startswith("bearer ") else ""
                if token not in server_cfg.api_keys:
                    return JSONResponse(
                        status_code=401,
                        content={"error": {"message": "invalid or missing API key",
                                           "type": "auth_error", "code": 401}},
                    )
        rid = uuid.uuid4().hex[:8]
        request.state.rid = rid
        start = time.monotonic()
        response = await call_next(request)
        if server_cfg.access_log:
            backend = getattr(request.state, "backend", "-")
            logger.info("rid=%s %s %s -> %d (%.2fs, backend=%s)",
                        rid, request.method, request.url.path.lstrip("/"),
                        response.status_code, time.monotonic() - start, backend)
        return response

    @app.get("/health")
    async def health():
        """Liveness: the proxy process is up."""
        return {"status": "ok"}

    @app.get("/ready")
    async def ready():
        """Readiness: at least one backend has fresh metrics."""
        fresh = proxy.registry.fresh_states()
        return JSONResponse(
            status_code=200 if fresh else 503,
            content={
                "status": "ready" if fresh else "degraded",
                "fresh_backends": len(fresh),
                "total_backends": len(proxy.registry.states),
            },
        )

    @app.get("/proxy/metrics")
    async def proxy_metrics():
        """Routing observability: per-backend state and counters."""
        return {
            "server": {
                "strategy": cfg.routing.strategy,
                "full_policy": cfg.routing.full_policy,
                "poll_interval": cfg.server.poll_interval,
                "stale_after": cfg.server.stale_after,
            },
            "backends": [s.snapshot(cfg.server.stale_after)
                         for s in proxy.registry.all_states()],
        }

    @app.get("/v1/models")
    async def models(request: Request):
        if not server_cfg.merge_models:
            return await proxy.passthrough(request, "models")
        return await proxy.merged_models()

    @app.api_route("/v1/{path:path}", methods=["GET", "POST", "DELETE"])
    async def proxy_route(request: Request, path: str):
        return await proxy.passthrough(request, path)

    @app.api_route("/{path:path}", methods=["GET", "POST", "DELETE"])
    async def proxy_route_no_v1(request: Request, path: str):
        """Convenience fallback: clients that omit the /v1 prefix.

        OpenAI-standard paths are rewritten to /v1/<path> on the backend.
        Everything else (e.g. llama.cpp's root-level /metrics, /health
        completions, /slots) is passed through unchanged.
        """
        if path == "models" and server_cfg.merge_models:
            # same merged union as /v1/models
            return await proxy.merged_models()
        if V1_REWRITE_PREFIXES.match(path):
            # OpenAI-standard path: passthrough adds /v1/ as usual
            return await proxy.passthrough(request, path)
        # root-level llama.cpp path (e.g. /metrics, /slots): forward as-is
        return await proxy.passthrough(request, path, v1=False)

    @app.exception_handler(NoBackendAvailableError)
    async def _503_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=503,
            content={"error": {"message": str(exc), "type": "proxy_error", "code": 503}},
        )

    return app


# ---------------------------------------------------------------------------
# proxy core
# ---------------------------------------------------------------------------

class ProxyApp:
    """Owns the httpx client, registry, poller and routing."""

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.registry = BackendRegistry(cfg)
        self.router = Router(self.registry, cfg)
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=10.0,
                read=cfg.server.request_timeout,
                write=30.0,
                pool=10.0,
            ),
            limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
            follow_redirects=False,
        )
        self.poller = MetricsPoller(self.registry, self.client)
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        await self.poller.start()
        self._started = True

    async def stop(self) -> None:
        if not self._started:
            return
        await self.poller.stop()
        await self.client.aclose()
        self._started = False

    # ------------------------------------------------------------------
    # routing
    # ------------------------------------------------------------------

    async def _select(self, request: Request, model: Optional[str] = None):
        """Pick a backend; converts router failures into 503 errors.

        ``model`` (when given) may pin the choice to backends that serve
        that exact model id — see Router.pick().
        """
        try:
            state = await self.router.pick(model=model)
        except (NoBackendAvailable, BackendFull) as e:
            raise NoBackendAvailableError(str(e)) from e
        request.state.backend = state.name
        return state

    # ------------------------------------------------------------------
    # passthrough
    # ------------------------------------------------------------------

    async def passthrough(self, request: Request, path: str, v1: bool = True):
        body = await request.body()
        # the OpenAI ``model`` field may pin the backend (exact id match);
        # only POST inference bodies carry it
        model = await self._extract_model(body) if request.method == "POST" else None
        state = await self._select(request, model=model)
        prefix = "v1/" if v1 else ""
        target = f"{state.url}/{prefix}{path}"
        headers = self._forward_headers(request, state)
        rid = request.state.rid

        if request.method == "POST" and await self._wants_stream(body):
            return await self._stream(rid, state, target, body, headers)

        # in_flight was reserved atomically inside router.pick()
        try:
            upstream = await self.client.request(
                request.method, target, content=body, headers=headers,
            )
        except httpx.HTTPError as e:
            self.router.record_result(state, ok=False)
            logger.warning("rid=%s upstream request failed (%s): %s", rid, state.name, e)
            return JSONResponse(
                status_code=502,
                content={"error": {"message": f"backend {state.name} unreachable: {e}",
                                   "type": "proxy_error", "code": 502}},
            )
        finally:
            self.router.release(state)
        self.router.record_result(state, ok=upstream.status_code < 400)
        # backend handled this request; reset its idle clock (only on the
        # success path — a 502 above means the GPU never did the work)
        state.mark_completed()
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "application/json"),
            headers={"x-proxy-backend": state.name, "x-request-id": rid},
        )

    async def _stream(self, rid: str, state, target: str, body: bytes, headers: dict):
        logger.info("rid=%s streaming to %s (busy=%d/%d)",
                    rid, state.name, state.busy, state.slots)
        # in_flight was reserved atomically inside router.pick()
        try:
            upstream = await self.client.send(
                self.client.build_request("POST", target, content=body, headers=headers),
                stream=True,
            )
        except httpx.HTTPError as e:
            self.router.release(state)
            self.router.record_result(state, ok=False)
            logger.warning("rid=%s upstream connect failed (%s): %s", rid, state.name, e)
            return JSONResponse(
                status_code=502,
                content={"error": {"message": f"backend {state.name} unreachable: {e}",
                                   "type": "proxy_error", "code": 502}},
            )
        self.router.record_result(state, ok=upstream.status_code < 400)
        if upstream.status_code >= 400:
            logger.warning("rid=%s backend %s returned HTTP %d",
                           rid, state.name, upstream.status_code)

        content_type = upstream.headers.get("content-type", "text/event-stream")

        async def gen():
            try:
                async for chunk in upstream.aiter_bytes():
                    yield chunk
            finally:
                # the stream keeps running after the handler returns, so
                # the in-flight slot is released here, at true completion,
                # and the backend's idle clock reset (GPU is now free)
                await upstream.aclose()
                self.router.release(state)
                state.mark_completed()

        return StreamingResponse(
            gen(),
            status_code=upstream.status_code,
            media_type=content_type,
            headers={"x-proxy-backend": state.name, "x-request-id": rid},
        )

    def _forward_headers(self, request: Request, state) -> dict:
        headers = {
            k: v for k, v in request.headers.items()
            if k.lower() not in HOP_BY_HOP
        }
        if state.api_key:
            headers["authorization"] = f"Bearer {state.api_key}"
        return headers

    @staticmethod
    async def _wants_stream(body: bytes) -> bool:
        """Detect streaming from the JSON body — how OpenAI-compatible
        clients actually signal it (more reliable than headers)."""
        if not body:
            return False
        try:
            payload_begin = None

            # async progressive parsing to avoid parsing large payloads
            async for i in ijson.parse_async(apyio.BytesIO(body)):
                if i[0] == '' and i[2] is None:
                    payload_begin = i[1]
                    continue
                if payload_begin == "start_map" and i[0] == "stream" and i[1] == "boolean":
                    return bool(i[2])
        except Exception as e:
            logger.debug("Failed to parse streaming flag from body: %s", e)
            # not JSON we can parse; fall back to byte heuristic
            return b'"stream": true' in body or b'"stream":true' in body
        return False

    @staticmethod
    async def _extract_model(body: bytes) -> Optional[str]:
        """Pull the OpenAI ``model`` field out of a POST body, or None.

        Used only for backend pinning (exact id match); the body is
        forwarded to the backend unmodified either way.
        """
        if not body:
            return None
        try:
            payload_begin = None
            model_name = None

            # async progressive parsing to avoid parsing large payloads
            async for i in ijson.parse_async(apyio.BytesIO(body)):
                if not i[0] and i[2] is None:
                    payload_begin = i[1]
                    continue
                if i[0] == 'model' and i[1] == "string" and payload_begin == "start_map":
                    model_name = i[2]
                    break
        except Exception as e:
            logger.debug("Failed to extract model from body: %s", e)
            return None
        return model_name

    # ------------------------------------------------------------------
    # model discovery
    # ------------------------------------------------------------------

    async def fanout_get(self, url_path: str) -> list:
        """GET url_path on every fresh backend; return parsed JSON bodies."""
        fresh = self.registry.fresh_states()

        async def one(state):
            try:
                r = await self.client.get(f"{state.url}{url_path}", timeout=5.0)
                if r.status_code == 200:
                    return r.json()
            except Exception as e:
                logger.debug("fanout %s failed for %s: %s", url_path, state.name, e)
            return None

        results = await asyncio.gather(*(one(s) for s in fresh))
        return [r for r in results if r is not None]

    async def merged_models(self) -> dict:
        """Union of /v1/models across all fresh backends, deduplicated by id."""
        merged: dict = {}
        for resp in await self.fanout_get("/v1/models"):
            for m in resp.get("data", []):
                if m.get("id"):
                    merged[m["id"]] = m
        return {"object": "list", "data": list(merged.values())}
