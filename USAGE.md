# llm-proxy — Usage Guide

A detailed, practical reference for operating **llm-proxy**, the slot-aware
reverse proxy for llama.cpp `llama-server` backends.

* [`README.md`](README.md) — what it is, why it exists, design decisions.
* **This file** — how to install, configure, run, and operate it in production.

The single most important mental model: **clients point at one stable URL
(`http://<proxy>:4000/v1`) and the proxy fans requests out across N
`llama-server` instances, routing on each GPU's *real* queue depth** (the
`llama_slots_busy` / `llamacpp:requests_processing` Prometheus metric), not on
TCP connection counts.

---

## Table of contents

1. [Requirements & installation](#1-requirements--installation)
2. [Prerequisite: backends must expose `/metrics`](#2-prerequisite-backends-must-expose-metrics)
3. [Quick start (5 minutes)](#3-quick-start-5-minutes)
4. [Configuration reference](#4-configuration-reference)
5. [CLI reference](#5-cli-reference)
6. [HTTP API reference](#6-http-api-reference)
7. [Client usage](#7-client-usage)
8. [Routing strategies](#8-routing-strategies)
9. [Capacity policy (`full_policy`)](#9-capacity-policy-full_policy)
10. [Pinning a request to a specific GPU](#10-pinning-a-request-to-a-specific-gpu)
11. [Monitoring & observability](#11-monitoring--observability)
12. [Deployment](#12-deployment)
13. [Security hardening](#13-security-hardening)
14. [Troubleshooting](#14-troubleshooting)
15. [Running the tests](#15-running-the-tests)
16. [When NOT to use this](#16-when-not-to-use-this)

---

## 1. Requirements & installation

**Proxy host requirements** — deliberately minimal:

* Python **3.9+** (3.12 recommended).
* HTTP reachability to every backend. **No GPU, no llama.cpp, no shared
  filesystem** on the proxy host. A Raspberry Pi or a $5 VPS is fine; the only
  real constraint is network latency to the backends (~1 RTT of TTFT cost,
  negligible on a LAN).

**Dependencies** (installed automatically):

| package | purpose |
|---|---|
| `fastapi` | HTTP framework |
| `uvicorn[standard]` | ASGI server |
| `httpx` | async backend client (shared connection pool) |
| `pydantic` | config validation |
| `PyYAML` | config parsing |

```bash
# from the project root
pip install .            # install the `llm-proxy` console script
# or, for development:
pip install -e ".[dev]"  # editable + pytest/pytest-asyncio
```

This exposes a single executable on `PATH`: **`llm-proxy`**
(`llm_proxy.cli:main`).

---

## 2. Prerequisite: backends must expose `/metrics`

The proxy's entire routing signal comes from each backend's Prometheus
`/metrics` endpoint. **If a `llama-server` does not expose `/metrics`, the
proxy cannot read its queue depth and will mark it stale → no traffic.**

Start every `llama-server` with metrics enabled (flag name varies by build):

```bash
# newer llama.cpp builds
llama-server -m model.gguf --port 8080 --parallel 1 --metrics \
             --host 127.0.0.1 [--alias 9B-0]

# older builds that already expose /metrics need nothing extra
```

The proxy understands both metric generations automatically:

| build | busy = |
|---|---|
| new | `llamacpp:requests_processing` + `llamacpp:requests_deferred` |
| old | `llama_slots_busy` |

"Busy" is **total pending work** (generating now + queued), so a GPU with a
queue is correctly seen as loaded.

---

## 3. Quick start (5 minutes)

```bash
# 0) install
pip install .

# 1) start your llama-servers (see §2), e.g. five on localhost:
#    llama-server -m model.gguf --port 8080 --parallel 1 --metrics &
#    ... 8081 ... 8084

# 2) pick a config. For a local single-machine setup, one is provided:
ls config/            # example.yaml  production.yaml  local-test.yaml  local.yaml

# 3) validate the config without starting anything
llm-proxy --config config/local.yaml --check
# → config OK: 5 backend(s), 5 enabled, strategy=idle_aware, full_policy=overflow
#     - gpu0: http://127.0.0.1:8080 slots=1 weight=1.0
#     ...

# 4) run the proxy
llm-proxy --config config/local.yaml
# → Uvicorn running on http://0.0.0.0:4000

# 5) smoke-test
curl -s http://localhost:4000/health          # {"status":"ok"}
curl -s http://localhost:4000/ready           # {"status":"ready","fresh_backends":5,...}
curl -s http://localhost:4000/v1/models       # merged model list

# a real completion (non-streaming)
curl -s http://localhost:4000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"your-model","messages":[{"role":"user","content":"hi"}]}'
```

---

## 4. Configuration reference

Everything lives in **one YAML file**. The proxy is fully driven by it; there
are no hidden defaults you can only set elsewhere. Every key below is optional
except `backends` (and the per-backend `name` and `url`).

> `${ENV_VAR}` and `${ENV_VAR:-default}` are expanded from the environment
> before validation, so **the same file works in systemd, docker-compose, and
> Kubernetes without forking**. A `${VAR}` with no value *and* no default is a
> hard config error (fail-fast).

### `server` — the proxy process

| key | type | default | meaning |
|---|---|---|---|
| `host` | str | `0.0.0.0` | bind address of the proxy |
| `port` | int | `4000` | bind port (1–65535) |
| `log_level` | str | `info` | `debug` \| `info` \| `warning` \| `error` |
| `request_timeout` | float | `600.0` | **max seconds for one proxied request.** Long generations need generous values. This is the *read* timeout for inference calls — independent of the short poll timeouts below. |
| `poll_interval` | float | `1.0` | seconds between `/metrics` refreshes (background, never per-request) |
| `poll_timeout` | float | `2.0` | connect/read timeout for a *single* metrics fetch. Keep short so a slow backend can't stall the polling loop. |
| `stale_after` | float | `3.0` | a backend whose last successful sample is older than this is **excluded from routing**. Dead GPUs stop receiving traffic in ~this many seconds. |
| `api_keys` | list[str] | `[]` | when non-empty, clients must send `Authorization: Bearer <key>` matching one of these. `/health` and `/ready` stay open (for LB/container probes); **everything else — including `/proxy/metrics` — is protected.** |
| `access_log` | bool | `true` | emit a per-request access line with request id, backend, busy count |
| `merge_models` | bool | `true` | `/v1/models` returns the **union** of models seen from all backends (deduped by id). `false` → plain passthrough to one backend. |

### `routing` — selection & capacity

| key | type | default | meaning |
|---|---|---|---|
| `strategy` | str | `least_busy` | `least_busy` \| `round_robin` \| `idle_aware` — see [§8](#8-routing-strategies) |
| `full_policy` | str | `overflow` | `overflow` \| `wait` \| `reject` — what to do when the chosen backend is at slot capacity. See [§9](#9-capacity-policy-full_policy). |
| `wait_timeout` | float | `10.0` | seconds to wait for a free slot (only used when `full_policy: wait`) |
| `wait_poll_interval` | float | `0.2` | how often to re-check while waiting |

### `backends` — the llama-server fleet (list)

Each entry describes one `llama-server`. **At least one** is required, names
must be unique, and at least one must be `enabled`.

| key | type | default | meaning |
|---|---|---|---|
| `name` | str | — (required) | identifier, must match `[A-Za-z0-9._-]+`. Shown in logs and the `x-proxy-backend` header. |
| `url` | str | — (required) | **bare `http(s)://host:port` — NO trailing `/v1`.** The proxy appends `/v1/...` itself and polls `/metrics` at the backend root. A trailing `/v1` is a validation error. |
| `slots` | int | `1` | the backend's nominal concurrency = its `llama-server --parallel` value. Used only for capacity checks *before the first metric sample*; live routing always uses the real `llama_slots_busy` value. Set it to match `--parallel`. |
| `weight` | float | `1.0` | multiplicative preference (must be > 0). `2.0` makes this backend twice as likely to be picked at equal busyness; used as a tie-breaker in `least_busy`/`idle_aware`. |
| `api_key` | str \| null | `null` | sent to *this* backend as `Authorization: Bearer <key>` (matches `llama-server --api-key`). Use `${LLAMA_API_KEY}` to inject. |
| `enabled` | bool | `true` | set `false` to take a backend out of rotation without deleting it. |

### A complete annotated example

```yaml
server:
  host: 0.0.0.0
  port: 4000
  log_level: info
  request_timeout: 600
  poll_interval: 1.0
  poll_timeout: 2.0
  stale_after: 3.0
  access_log: true
  merge_models: true
  api_keys:
    - ${PROXY_API_KEY}          # clients must present this

routing:
  strategy: idle_aware
  full_policy: overflow

backends:
  - name: gpu-a
    url: http://10.0.0.11:8080  # bare host:port, no /v1
    slots: 4                    # = llama-server --parallel
    weight: 1.0
    api_key: ${LLAMA_API_KEY}   # = llama-server --api-key
  - name: gpu-b
    url: http://10.0.0.12:8080
    slots: 8
    weight: 1.5                 # preferred tie-breaker
```

> See the ready-made files: [`config/example.yaml`](config/example.yaml)
> (annotated starter), [`config/local.yaml`](config/local.yaml) (single-machine,
> 5 backends, no auth), [`config/production.yaml`](config/production.yaml)
> (3-GPU fleet template with env-injected secrets).

---

## 5. CLI reference

```
llm-proxy [OPTIONS]
```

| flag | default | meaning |
|---|---|---|
| `-c`, `--config` | `/etc/llm-proxy/prod.yaml` | path to the YAML config |
| `--check` | off | validate the config and print a summary, then exit (no server started) |
| `--host` | from config | override `server.host` |
| `--port` | from config | override `server.port` (int) |

Notes:

* The proxy always runs **single-worker** (`uvicorn workers=1`). In-memory
  routing state is per-process; multiple workers would each keep inconsistent
  views of backend busyness. See [§12.4 Multi-instance](#124-multiple-instances--nginx-front).
* `--check` output:

  ```
  config OK: 5 backend(s), 5 enabled, strategy=idle_aware, full_policy=overflow
    - gpu0: http://127.0.0.1:8080 slots=1 weight=1.0
    ...
  ```

* Exit codes: `0` success, `2` config error (message on stderr).

---

## 6. HTTP API reference

All responses from proxied inference carry two extra headers:

* `x-proxy-backend` — which backend (GPU) served the request.
* `x-request-id` — the per-request id also used in logs.

Backend errors are **forwarded as-is** (standard OpenAI-style error bodies), so
clients see normal API errors, not proxy-specific ones.

### Endpoint map

| Path | Methods | Auth* | Purpose |
|---|---|---|---|
| `/v1/models` | GET | yes | merged model list across fresh backends (or plain passthrough if `merge_models: false`) |
| `/v1/chat/completions` | POST | yes | inference; streams relayed byte-for-byte |
| `/v1/{anything}` | GET/POST/DELETE | yes | transparent passthrough to the chosen backend |
| `/health` | GET | **no** | liveness — process is up → `{"status":"ok"}` |
| `/ready` | GET | **no** | readiness — ≥1 fresh backend, else 503 |
| `/proxy/metrics` | GET | yes | per-backend state + routing counters (protected: it leaks topology) |
| `/{anything}` | GET/POST/DELETE | yes | convenience fallback for clients that omit `/v1` |

\* "Auth" = requires a Bearer key **only when** `server.api_keys` is set. With
no `api_keys`, everything is open.

### The `/v1/{path}` and `/{path}` routes

* `/v1/<path>` → forwarded to the chosen backend as `<backend>/v1/<path>`.
* `/<path>` (client omitted `/v1`) →
  * OpenAI-standard paths (`models`, `chat/completions`, `embeddings`,
    `moderations`, `audio/...`, `images/...`, `files`, `fine_tuning`,
    `assistants`, `threads`, `runs`) are rewritten to `/v1/<path>` on the
    backend;
  * `models` with `merge_models: true` returns the merged union (same as
    `/v1/models`);
  * **everything else** (llama.cpp's root-level `/metrics`, `/slots`, `/health`)
    is forwarded **unchanged** (no `/v1/` prepended).

### Status codes you may see from the proxy itself

| code | meaning |
|---|---|
| `401` | bad/missing Bearer key (only when `api_keys` is set) |
| `502` | chosen backend unreachable / upstream HTTP error |
| `503` | no fresh backend available (all stale), or `full_policy: reject` with all backends full; also from `/ready` when degraded |

### `/ready` response

```json
{"status": "ready", "fresh_backends": 5, "total_backends": 5}   // 200
{"status": "degraded", "fresh_backends": 0, "total_backends": 5} // 503
```

---

## 7. Client usage

The **only** change versus talking to a single `llama-server` directly is the
base URL. Every OpenAI-compatible client (OpenAI SDK, LangChain, Chatbox, …)
works unmodified.

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(base_url="http://proxy-host:4000/v1", api_key="any")  # "any" if no api_keys set

# non-streaming
r = client.chat.completions.create(
    model="qwen2.5-72b",
    messages=[{"role": "user", "content": "hello"}],
)
print(r.choices[0].message.content)

# streaming — relayed byte-for-byte by the proxy
r = client.chat.completions.create(
    model="qwen2.5-72b",
    messages=[{"role": "user", "content": "hello"}],
    stream=True,
)
for chunk in r:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

### curl

```bash
# non-streaming
curl -s http://localhost:4000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"your-model","messages":[{"role":"user","content":"hi"}]}'

# streaming (-N disables curl buffering; the proxy relays SSE chunks as produced)
curl -N http://localhost:4000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"your-model","stream":true,"messages":[{"role":"user","content":"hi"}]}'

# with auth (only if server.api_keys is set)
curl -s http://localhost:4000/v1/models -H "Authorization: Bearer $PROXY_API_KEY"
```

### How streaming is detected

The proxy inspects the **request body** for `"stream": true` (how OpenAI
clients actually signal it) with a byte-heuristic fallback. This is more
reliable than sniffing headers, and it's what triggers the explicit SSE relay
(path vs. buffered response).

---

## 8. Routing strategies

Set via `routing.strategy`. In every strategy a backend whose metrics are older
than `stale_after` is excluded from selection first ("fresh" backends only).

### `least_busy` (default)

Pick the fresh backend with the **fewest busy slots**. Ties broken by `weight`
(higher first), then by `name` (deterministic). Busy here means
`effective_busy = observed busy + in_flight` (see below), so it reacts to a
burst even before the next 1 s metric sample.

### `round_robin`

Rotate deterministically over the fresh backends. Staleness still applies (a
dead backend is skipped, not rotated into).

### `idle_aware`

**Score-based, idle time first.** Prefer the GPU that has been *resting*
longest so traffic rotates across all backends instead of hammering one.

* A **busy** GPU is idle `0` (a GPU doing work is not resting).
* A **free** GPU reports how long it's been free.
* A **never-used** GPU is infinitely idle.

Because busy ⇒ idle 0, free GPUs always sort ahead of busy ones — no separate
capacity key needed. Tie-breaks: least busy, then `weight`, then `name` (unique
⇒ exactly one deterministic winner). The idle clock resets when a request
**completes** on that backend, and `in_flight` drops a just-dispatched GPU to
idle 0 immediately — so both bursts and sequential traffic rotate:

```
req1→gpu0  req2→gpu1  req3→gpu2  req4→gpu3  req5→gpu4  req6→gpu0 …
```

> `config/example.yaml` and `config/local.yaml` use `idle_aware`;
> `config/production.yaml` uses `least_busy`.

### In-flight tracking (all strategies)

The metrics sample is up to `poll_interval` old, so the proxy also counts
requests *it* has dispatched and not yet completed. `pick()` **atomically
reserves** an in-flight slot per backend (no `await` between decision and
reserve — safe on the single-threaded loop), and routing keys on
`busy + in_flight` (an upper bound on true load). A burst of 6 requests on 5
single-slot GPUs spreads `1/1/1/1/1+1` instead of piling 6 onto one GPU. For
streaming requests the slot is released only when the stream truly finishes.

---

## 9. Capacity policy (`full_policy`)

When the chosen backend is at its slot limit (`effective_busy >= slots`):

| policy | behaviour |
|---|---|
| `overflow` (default) | route anyway; llama.cpp queues internally. **Never 503s on load.** Matches llama.cpp's own behaviour. |
| `wait` | wait up to `wait_timeout` s for a free slot, then overflow |
| `reject` | 503 immediately when every fresh backend is full |

`slots` per backend is only used before the first metric sample and for these
capacity checks; the routing *signal* itself is always the live
`llama_slots_busy` value.

---

## 10. Pinning a request to a specific GPU

Give each GPU a distinct `--alias` in `llama-server` (same weights are fine).
The proxy records each backend's reported model id, so the standard OpenAI
`model` field doubles as a **backend selector** — exact id match, no client
changes beyond the model string:

```bash
llama-server -m model.gguf --alias 9B-0 --metrics ...   # gpu0
llama-server -m model.gguf --alias 9B-1 --metrics ...   # gpu1
```

```python
client.chat.completions.create(model="9B-2", messages=[...])   # → gpu2 only
client.chat.completions.create(model="anything-else", ...)     # → normal routing
```

Semantics:

* **match** → candidates restricted to the backends serving that id, then the
  active strategy runs within the subset.
* **busy pinned GPU** → governed by `full_policy`; with `overflow` (default)
  llama.cpp queues it on the pinned GPU (the pin is always honored).
* **stale pinned GPU** → not a fresh candidate, so the pin yields nothing and
  we **fall back** to all fresh backends.
* **no match / no fresh backend at all** → normal routing, or 503 if nothing
  can serve.

---

## 11. Monitoring & observability

### `GET /proxy/metrics`

Per-backend state and routing counters (auth-protected when `api_keys` is set,
because it reveals backend topology). Shape:

```json
{
  "server": {
    "strategy": "idle_aware",
    "full_policy": "overflow",
    "poll_interval": 1.0,
    "stale_after": 3.0
  },
  "backends": [
    {
      "name": "gpu0",
      "url": "http://127.0.0.1:8080",
      "slots": 1,
      "busy": 0,
      "in_flight": 1,
      "free": 0,
      "weight": 1.0,
      "stale": false,
      "age_s": 0.42,
      "idle_s": 12.5,
      "last_error": null,
      "models": ["9B-0"],
      "requests_routed": 128,
      "requests_failed": 0,
      "last_routed_at": 1717000000.0
    }
  ]
}
```

Field notes:

* `busy` — live queue depth from `/metrics` (generating + queued).
* `in_flight` — requests this proxy dispatched and not yet completed.
* `free` — `max(0, slots - (busy + in_flight))`.
* `stale` — true when the last sample is older than `stale_after` (excluded
  from routing).
* `age_s` — seconds since the last successful poll.
* `idle_s` — how long the backend has been free (`0` if busy, `null` if never
  used and free).
* `last_error` — last metrics-poll error (e.g. `metrics HTTP 503`, or "no
  recognized busy metric…") — your first place to look when a backend goes
  stale (see [§14](#14-troubleshooting)).

### Access logs

When `access_log: true`, each request emits a line-oriented record with a
request id that ties the log line to the `x-request-id` response header:

```
rid=a1b2c3d4 POST chat/completions -> 200 (0.42s, backend=gpu-a)
```

Stream dispatches are also logged:

```
rid=0f3e9a21 streaming to gpu2 (busy=0/1)
```

Ship these to your log aggregator; `/proxy/metrics` gives the routing counters
for dashboards.

---

## 12. Deployment

### 12.1 systemd

A hardened unit is provided at [`deploy/llm-proxy.service`](deploy/llm-proxy.service).

```bash
# 1) install + create a dedicated user and config dir
sudo useradd -r -s /usr/sbin/nologin llmproxy
sudo mkdir -p /etc/llm-proxy
sudo cp config/production.yaml /etc/llm-proxy/prod.yaml
sudo chown llmproxy:llmproxy /etc/llm-proxy/prod.yaml

# 2) put secrets in an env file (one KEY=value per line)
sudo tee /etc/llm-proxy/env >/dev/null <<'EOF'
PROXY_API_KEY=sk-...
LLAMA_API_KEY=...
EOF
sudo chmod 600 /etc/llm-proxy/env

# 3) install + enable the unit
sudo cp deploy/llm-proxy.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now llm-proxy

# 4) verify
systemctl status llm-proxy
curl -s http://127.0.0.1:4000/ready
```

The unit runs as `User=llmproxy`, reads `EnvironmentFile=/etc/llm-proxy/env`,
`Restart=on-failure`, and applies hardening (`NoNewPrivileges`, `PrivateTmp`,
`ProtectSystem=full`, `ProtectHome`). Adjust `ExecStart` if your install path or
config path differs.

### 12.2 Docker

[`deploy/Dockerfile`](deploy/Dockerfile) builds a slim, non-root image.

```bash
docker build -f deploy/Dockerfile -t llm-proxy .

# mount your config + inject secrets via environment
docker run -d --name llm-proxy \
  -p 4000:4000 \
  -v /path/to/prod.yaml:/etc/llm-proxy/prod.yaml:ro \
  -e PROXY_API_KEY=sk-... -e LLAMA_API_KEY=... \
  llm-proxy --config /etc/llm-proxy/prod.yaml
```

The image has a `HEALTHCHECK` that probes `/health` every 10 s. The container
user is `proxy` (uid 10001).

### 12.3 Kubernetes (brief)

Because of `${ENV}` expansion, no forking is needed: keep the YAML in a
**ConfigMap** and secrets in a **Secret**, then reference the secret values as
env vars on the pod. The proxy reads `--config` from the mounted file and pulls
secrets from the environment at startup. Bind to `0.0.0.0` inside the pod and
expose port `4000` via a Service; use `/ready` for readiness probes and
`/health` for liveness (both are open without auth).

### 12.4 Multiple instances + Nginx front

In-memory routing state is **per-process**, so:

* Each instance runs `workers=1` (the CLI enforces this). For higher
  throughput, run **N proxy instances** (ports `4000`, `4001`, …) behind an LB.
  The proxy is lightweight — the bottleneck is always the LLM backends — and
  routing decisions are per-request, so a connection-level load balancer on top
  is harmless.
* Each instance polls the same backends independently; state divergence is
  bounded by one poll interval and self-heals.

[`deploy/nginx.example.conf`](deploy/nginx.example.conf) provides TLS + a two-
instance upstream. The critical line for streaming:

```nginx
proxy_buffering off;      # SSE chunks reach the client as produced
proxy_read_timeout 600s;
proxy_send_timeout 600s;
```

---

## 13. Security hardening

For **remote / internet-facing** deployments:

* **Bind `llama-server` to internal interfaces only** and firewall the backend
  ports to the proxy host's IP. Never expose them publicly.
* **Set `server.api_keys`** when the proxy is internet-facing (401 otherwise).
  `/health` and `/ready` stay open for LB/container health checks; everything
  else — including `/proxy/metrics` — is protected.
* **Set per-backend `api_key`** (with `llama-server --api-key`) to protect
  proxy→backend traffic.
* **TLS termination**: put Nginx/Caddy in front (see
  [`deploy/nginx.example.conf`](deploy/nginx.example.conf)); keep
  `proxy_buffering off` for SSE.
* Inject all secrets via the environment (`${PROXY_API_KEY}`,
  `${LLAMA_API_KEY}`) — **never commit real keys** to a config file.

For a **single trusted machine** (e.g. [`config/local.yaml`](config/local.yaml))
you can run keyless, but if that box is reachable by untrusted parties, add
`server.api_keys` and/or firewall port `4000`.

---

## 14. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `/ready` returns 503 / inference 503s "all backends stale" | A backend's `/metrics` is failing or older than `stale_after`. Check `last_error` in `/proxy/metrics`. Most often: the server wasn't started with metrics enabled (see [§2](#2-prerequisite-backends-must-expose-metrics)) or is unreachable. |
| `last_error: "no recognized busy metric in /metrics"` | The backend serves `/metrics` but not a recognized busy metric (`llamacpp:requests_processing` / `llama_slots_busy`). Newer/older build mismatch — confirm the flag exposes the expected metric. |
| `config error: backend url must not end in /v1` | Remove the trailing `/v1` from `backends[].url`. The proxy appends it. |
| `config error: environment variable 'X' is not set and has no default` | Export `X`, or give a default: `${X:-fallback}`. |
| `401 invalid or missing API key` | You set `server.api_keys`; send `Authorization: Bearer <key>`. (Or remove `api_keys` for trusted local use.) |
| Streaming appears to "buffer" / arrive late | A front proxy is buffering SSE. Set `proxy_buffering off;` in Nginx (see [§12.4](#124-multiple-instances--nginx-front)). The proxy itself relays byte-for-byte. |
| All traffic piles onto one GPU | You're on `round_robin`/`least_busy` and that GPU is genuinely least loaded, or others are stale. Try `idle_aware` to force rotation; verify other backends are fresh in `/proxy/metrics`. |
| A request is 502 "backend unreachable" | The chosen backend dropped the connection / errored. It will be retried on a fresh backend next request; check that server's logs. |
| TTFT feels high over the network | Expected ~1× RTT (first chunk travels backend→proxy→client). Keep the proxy in the same region/LAN as the GPUs. |

**General first step:** `curl -s http://<proxy>:4000/proxy/metrics` and read
each backend's `stale`, `age_s`, and `last_error`. That single endpoint tells
you whether the problem is reachability, metrics parsing, or routing.

---

## 15. Running the tests

```bash
pip install -e ".[dev]"
pytest -q
```

Covers config validation + env expansion, all routing strategies and capacity
policies, and end-to-end passthrough (streaming + non-streaming, model merge,
auth, 503 failover) against fake `llama-server` backends on real localhost
ports. Test files: `tests/test_config.py`, `test_state.py`, `test_router.py`,
`test_app.py`.

---

## 16. When NOT to use this

* **Multi-provider routing** (OpenAI/Anthropic/…) → use **LiteLLM**.
* **Persistent API-key / user management** → use **Kong** / **APISIX**.
* **Zero Python ops experience** → use **Nginx `least_conn`** with keep-alive
  disabled.

This proxy is the middle ground: **one GPU fleet, one endpoint, zero
third-party routing logic.** If you need those other features, reach for the
tool built for them.
