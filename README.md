# llm-proxy

Config-separated, slot-aware reverse proxy for multiple
[llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server`
instances. Production-grade, single purpose, zero supply-chain surface
beyond `fastapi` + `httpx`.

Routing uses llama.cpp's own `llama_slots_busy` Prometheus metric — the
**actual** queue depth on each GPU — instead of TCP connection counts, so
keep-alive connections can't distort load balancing and streaming is
handled explicitly.

```
                        ┌──────────────────────────────┐
 clients (OpenAI SDK,   │        llm-proxy             │      llama-server
 LangChain, ...)  ────► │  ┌──────────┐  ┌──────────┐  │  ┌──► gpu-a:8080  (4 slots)
  one stable URL        │  │ poller   │  │  router  │  │  ├──► gpu-b:8080  (4 slots)
                        │  │ 1 s tick │  │least busy│  │  └──► gpu-c:8080  (8 slots)
                        │  └──────────┘  └──────────┘  │
                        └──────────────────────────────┘
```

## Why this and not Nginx / LiteLLM

| Factor | Nginx `least_conn` | This proxy |
|---|---|---|
| Routing signal | TCP connections | `llama_slots_busy` (real queue depth) |
| Keep-alive distortion | idle conns read as "busy" | none |
| Streaming | native but blind to content | explicit SSE relay, body-based stream detection |
| Health awareness | passive (connection failures) | active polling + staleness guard (~3 s failover) |
| Attack surface | large | ~900 LOC Python, auditable in an afternoon |
| Multi-provider / key management | no | no (by design — see below) |

Use LiteLLM instead if you need multi-provider routing or persistent
key/user management. Use Nginx if your team has zero Python ops
experience. This proxy is the middle ground: **one GPU fleet, one
endpoint, zero third-party routing logic.**

## Design decisions

1. **Background polling, never per-request.** Metrics refresh every
   `poll_interval` (default 1 s) regardless of traffic. Per-request
   polling would add 50–200 ms to every inference call.
2. **Staleness guard.** A backend whose `/metrics` is older than
   `stale_after` (default 3 s) is excluded from selection. Dead GPUs
   stop receiving traffic in ~3 s, no connection errors required.
3. **Streaming detection via body inspection.** Parses
   `"stream": true` from the JSON payload — how OpenAI-compatible
   clients actually signal streaming — with a byte heuristic fallback.
4. **No persistent state.** No Redis, no DB. Restart = clean slate,
   which is acceptable because LLM routing needs no durable state.
   (Consequence: run **one instance per process**; see scaling below.)
5. **Single shared `httpx.AsyncClient`.** One connection pool across all
   backends; no TCP handshake per request.
6. **Transparent passthrough.** Request and response bodies are
   forwarded byte-for-byte. Clients change exactly one thing: the base
   URL. Every OpenAI SDK, LangChain, Chatbox, etc. works unmodified.
7. **In-flight tracking for burst traffic.** The metrics sample is up to
   `poll_interval` old, so the proxy also counts requests *it* has
   dispatched and not yet completed. `pick()` atomically reserves an
   in-flight slot per backend (no await between decision and reserve —
   safe on the single-threaded loop), and routing keys on
   `busy + in_flight`. A burst of 6 requests on 5 single-slot GPUs
   spreads 1/1/1/1/1+1 instead of piling 6 onto one GPU.
8. **Both llama.cpp metric generations.** New builds expose
   `llamacpp:requests_processing` (+`_deferred`); old builds expose
   `llama_slots_busy`. Busy = total pending work
   (processing + deferred), so a GPU with a queue is seen as loaded.

## Project layout

```
├── config/
│   ├── example.yaml          # annotated starter config
│   └── production.yaml       # template for a 3-GPU fleet
├── deploy/
│   ├── llm-proxy.service     # systemd unit (hardened)
│   ├── Dockerfile            # slim, non-root
│   └── nginx.example.conf    # optional TLS + multi-instance front
├── src/llm_proxy/
│   ├── cli.py                # entrypoint (llm-proxy --config ...)
│   ├── config.py             # YAML + ${ENV} expansion + validation
│   ├── state.py              # backend state + metrics poller
│   ├── router.py             # selection strategy + capacity policy
│   └── app.py                # FastAPI passthrough, auth, health
└── tests/                    # unit + end-to-end (fake backends)
```

## Installation

```bash
pip install .          # or: pip install -e .
llm-proxy --config config/example.yaml --check   # validate config
llm-proxy --config config/production.yaml        # run
```

Python 3.9+. The proxy machine needs **no GPU, no llama.cpp, no shared
filesystem** — just HTTP reachability to the backends. A Raspberry Pi
or a $5 VPS is fine; latency to the backends is the only real
constraint (~1 RTT of TTFT cost, negligible on a LAN).

## Configuration

Everything lives in one YAML file. `${VAR}` / `${VAR:-default}` are
expanded from the environment, so the same file works in systemd,
docker-compose and K8s. See [`config/example.yaml`](config/example.yaml)
for the annotated version.

```yaml
server:
  host: 0.0.0.0
  port: 4000
  log_level: info
  request_timeout: 600      # s, max for one proxied request
  poll_interval: 1.0        # s between /metrics refreshes
  poll_timeout: 2.0         # s, single metrics fetch
  stale_after: 3.0          # s, exclude backend older than this
  access_log: true
  merge_models: true        # /v1/models = union across backends
  api_keys:                 # optional: require Bearer auth from clients
    - ${PROXY_API_KEY}

routing:
  strategy: idle_aware      # least_busy | round_robin | idle_aware
  full_policy: overflow     # overflow | wait | reject
  wait_timeout: 10.0        # s (full_policy: wait only)

backends:
  - name: gpu-a
    url: http://10.0.0.11:8080
    slots: 4                # llama-server --parallel value
    weight: 1.0             # tie-break preference
    api_key: ${LLAMA_API_KEY}   # optional per-backend Bearer token
```

### Backend selection

* `least_busy`: fewest busy slots; ties broken by `weight` (higher
  first), then name (deterministic).
* `round_robin`: rotate over fresh backends (staleness still applies).
* `idle_aware`: **idle time first.** Prefer the GPU that has been resting
  longest so traffic rotates across all backends instead of hammering one.
  A *busy* GPU is idle `0` (a GPU doing work is not resting); a *free* GPU
  is however long it has been free; a *never-used* GPU is infinitely idle.
  Because busy ⇒ idle 0, free GPUs always sort ahead of busy ones — no
  separate capacity check needed. Tie-breaks: least busy, then `weight`,
  then name (unique ⇒ exactly one deterministic winner). The idle clock
  resets when a request **completes** on that backend, and `in_flight`
  drops a just-dispatched GPU to idle 0 immediately, so bursts and
  sequential traffic both rotate.

  ```
  req1→gpu0  req2→gpu1  req3→gpu2  req4→gpu3  req5→gpu4  req6→gpu0 …
  ```

### Pinning a request to a specific GPU (standard `model` field)

Give each GPU a distinct `--alias` in llama-server (same weights are fine):

```bash
llama-server -m model.gguf --alias 9B-0 --metrics ...   # gpu0
llama-server -m model.gguf --alias 9B-1 --metrics ...   # gpu1
```

The proxy records each backend's reported model id, so the OpenAI `model`
field doubles as a backend selector — **exact id match**, no client changes
beyond the model string:

```python
client.chat.completions.create(model="9B-2", messages=[...])   # → gpu2 only
client.chat.completions.create(model="anything-else", ...)     # → normal routing
```

Semantics:
* **match** → candidates restricted to the backends serving that id, then the
  active strategy runs within the subset.
* **busy pinned GPU** → governed by `full_policy`; with `overflow` (default)
  llama.cpp queues it on the pinned GPU (pin always honored).
* **stale pinned GPU** → it's not a fresh candidate, so the pin yields
  nothing and we **fall back** to all fresh backends.
* **no match / no fresh backend at all** → normal routing, or 503 if nothing
  can serve.

### Capacity policy (`routing.full_policy`)

When the chosen backend is at its slot limit:

| policy | behaviour |
|---|---|
| `overflow` (default) | route anyway; llama.cpp queues internally. Never 503s on load. |
| `wait` | wait up to `wait_timeout` s for a free slot, then overflow |
| `reject` | 503 immediately when every fresh backend is full |

`slots` per backend is only used before the first metric sample and for
capacity checks; the routing signal itself is always the live
`llama_slots_busy` value.

## Endpoints

| Path | Purpose |
|---|---|
| `GET /v1/models` | merged model list across fresh backends (or plain passthrough with `merge_models: false`) |
| `POST /v1/chat/completions` | passthrough; streams relayed byte-for-byte |
| `GET/POST /v1/{anything}` | transparent passthrough to the chosen backend |
| `GET /health` | liveness (process up) — always open, even with `api_keys` |
| `GET /ready` | readiness (≥1 fresh backend; 503 otherwise) — always open |
| `GET /proxy/metrics` | per-backend busy/slots/staleness + routing counters (auth-protected when `api_keys` is set: it leaks backend topology) |

Responses carry `x-proxy-backend` (which GPU served the request) and
`x-request-id` headers. Backend errors are forwarded as-is, so clients
see standard OpenAI-style errors.

## Client usage

```python
from openai import OpenAI

client = OpenAI(base_url="http://proxy-host:4000/v1", api_key="any")
r = client.chat.completions.create(
    model="qwen2.5-72b",
    messages=[{"role": "user", "content": "hello"}],
    stream=True,
)
for chunk in r:
    print(chunk.choices[0].delta.content, end="")
```

That is the **only** change versus talking to a single llama-server
directly.

## Security (remote deployments)

* Bind `llama-server` to internal interfaces only; firewall the backend
  ports to the proxy host's IP. Never expose them publicly.
* Set `api_keys` on the proxy when it is internet-facing (401 otherwise).
  `/health` and `/ready` stay open so LB/container health checks work
  without a key; everything else — including `/proxy/metrics` — is
  protected.
* Set per-backend `api_key` (with `llama-server --api-key`) to protect
  proxy→backend traffic.
* For TLS termination, put Nginx/caddy in front (see
  [`deploy/nginx.example.conf`](deploy/nginx.example.conf)); keep
  `proxy_buffering off` for SSE.

## Running multiple instances

In-memory routing state is **per-process**, so:

* `uvicorn workers=1` (the CLI already enforces this). Multiple workers
  would each keep independent, inconsistent views of backend busyness.
* For higher throughput, run N multiple proxy instances (N×4000,
  N×4001, …) behind Nginx `round_robin`. The proxy is lightweight — the
  bottleneck is always the LLM backends — and routing decisions are
  per-request, so a connection-level LB on top is harmless.
* Each instance polls the same backends independently; state divergence
  is bounded by one poll interval and self-heals.

## Operational notes

* **TTFT over the network** increases by ~1× RTT (first chunk travels
  backend → proxy → client); later chunks pipeline. Keep the proxy in
  the same region as the GPUs.
* **Graceful degradation**: with `full_policy: overflow` the proxy never
  rejects on load; requests queue inside llama.cpp. 503s only occur when
  *all* backends are stale (or `full_policy: reject` and all are full).
* **Logs** are line-oriented with a request id:
  `rid=a1b2c3d4 POST chat/completions -> 200 (0.42s, backend=gpu-a)`.
  Ship them to your log aggregator; `/proxy/metrics` gives the routing
  counters for dashboards.
* **Adding/removing a GPU** = edit the config, reload. A backend that
  disappears from the config stops receiving traffic on the next
  restart; a disabled one (`enabled: false`) does the same.

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

Covers config validation/env expansion, all routing strategies and
capacity policies, and end-to-end passthrough (streaming + non-streaming,
model merge, auth, 503 failover) against fake llama-server backends on
real localhost ports.

## When NOT to use this

* Multi-provider routing (OpenAI/Anthropic/…) → LiteLLM.
* Persistent API-key/user management → Kong / APISIX.
* Zero Python ops experience → Nginx `least_conn` with keep-alive
  disabled.
