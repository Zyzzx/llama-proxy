#!/usr/bin/env python3
"""Dev helper: fake llama-server for local testing of llm-proxy.

Usage: python scripts/fake_llama_server.py --port 8080 --model test-model [--busy 1]
"""
import argparse
import json
import sys

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
import uvicorn


def build_app(model_id: str, busy: int):
    app = FastAPI()
    state = {"busy": busy}

    @app.get("/metrics")
    async def metrics():
        # current llama.cpp metric names (--metrics); 'busy' is exposed
        # as requests_processing, deferred stays 0
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
        return {"object": "list", "data": [{"id": model_id, "object": "model"}]}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        payload = await request.json()
        if payload.get("stream"):
            async def gen():
                for tok in ["Hello", " from", " fake", " llama!"]:
                    chunk = {"id": "cmpl-1", "object": "chat.completion.chunk",
                             "choices": [{"index": 0, "delta": {"content": tok},
                                          "finish_reason": None}]}
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield 'data: [DONE]\n\n'
            return StreamingResponse(gen(), media_type="text/event-stream")
        return {"id": "cmpl-1", "object": "chat.completion",
                "choices": [{"message": {"role": "assistant", "content": "Hello from fake llama!"}}],
                "model": model_id}

    return app


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--model", default="test-model")
    p.add_argument("--busy", type=int, default=1)
    args = p.parse_args()
    uvicorn.run(build_app(args.model, args.busy), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    sys.exit(main())
