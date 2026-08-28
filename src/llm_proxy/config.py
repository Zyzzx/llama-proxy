"""Configuration loading and validation.

The proxy is fully driven by a YAML file (see ``config/example.yaml``).
Secrets and hostnames may be injected from the environment with the
``${VAR}`` / ``${VAR:-default}`` syntax, so the same file works in
``docker-compose``, systemd, and Kubernetes without forking.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger("llm_proxy.config")

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class ConfigError(Exception):
    """Raised when the configuration file is missing or invalid."""


# ---------------------------------------------------------------------------
# env expansion
# ---------------------------------------------------------------------------

def _expand_env(value: Any) -> Any:
    """Recursively expand ${VAR} and ${VAR:-default} in strings."""
    if isinstance(value, str):
        def repl(m: "re.Match[str]") -> str:
            name, default = m.group(1), m.group(2)
            env = os.environ.get(name)
            if env is not None:
                return env
            if default is not None:
                return default
            raise ConfigError(
                f"environment variable '{name}' is not set and has no default"
            )
        return _ENV_RE.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------

class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = Field(4000, ge=1, le=65535)
    log_level: str = "info"
    # Seconds before a proxied request is aborted. Long generations need
    # generous values; the *poll* timeout below is separate and short.
    request_timeout: float = Field(600.0, gt=0)
    # Connect/read timeout for the health poller (keep short: a slow
    # backend must not stall the polling loop).
    poll_timeout: float = Field(2.0, gt=0)
    # Seconds between /metrics refreshes.
    poll_interval: float = Field(1.0, gt=0)
    # A backend whose metrics are older than this is considered stale and
    # is not used for routing decisions.
    stale_after: float = Field(3.0, gt=0)
    # Optional bearer auth for *clients* calling the proxy. When a list of
    # keys is given, requests must carry Authorization: Bearer <key>.
    api_keys: List[str] = Field(default_factory=list)
    # Extra per-request log fields (request id, backend, busy count) are
    # always emitted at INFO level when access logging is on.
    access_log: bool = True
    # If true, /v1/models returns the union of models seen from all
    # backends (deduplicated), instead of whatever one backend answers.
    merge_models: bool = True


class BackendConfig(BaseModel):
    name: str
    url: str
    # Nominal concurrency of the backend (llama-server --parallel). Used
    # only for the capacity fallback before the first metric sample.
    slots: int = Field(1, ge=1)
    # Multiplicative preference. 2.0 makes this backend twice as likely
    # to be picked at equal busyness.
    weight: float = Field(1.0, gt=0)
    # Sent to the backend as Authorization: Bearer <key> (llama-server
    # --api-key). Use ${LLAMA_API_KEY} to inject from the environment.
    api_key: Optional[str] = None
    enabled: bool = True

    @field_validator("url")
    @classmethod
    def _validate_url(cls, v: str) -> str:
        if not re.match(r"^https?://[^\s/]+", v):
            raise ValueError(f"backend url must start with http:// or https://: {v!r}")
        v = v.rstrip("/")
        if v.endswith("/v1"):
            raise ValueError(
                f"backend url must not end in /v1 — the proxy appends the "
                f"/v1/ prefix itself and polls /metrics at the backend root: {v!r}"
            )
        return v

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        if not re.match(r"^[A-Za-z0-9._-]+$", v):
            raise ValueError(f"backend name must match [A-Za-z0-9._-]+: {v!r}")
        return v


class RoutingConfig(BaseModel):
    # least_busy: pick the backend with the fewest busy slots (ties broken
    # by weight, then by name for determinism).
    # round_robin: rotate over enabled backends (staleness still applies).
    # idle_aware: score-based, idle time first — prefer the GPU that has
    # been resting longest (a busy GPU is idle=0; a free GPU is however
    # long it has been free; never-used is infinitely idle), so traffic
    # rotates across all backends instead of hammering one.
    strategy: str = "least_busy"
    # When the least-busy backend is at (or over) its slot capacity:
    #   overflow  - route anyway; llama.cpp queues the request internally
    #   wait      - wait up to wait_timeout seconds for a free slot
    #   reject    - answer 503 immediately
    full_policy: str = "overflow"
    wait_timeout: float = Field(10.0, ge=0)
    wait_poll_interval: float = Field(0.2, gt=0)

    @field_validator("strategy")
    @classmethod
    def _validate_strategy(cls, v: str) -> str:
        if v not in ("least_busy", "round_robin", "idle_aware"):
            raise ValueError(f"unknown routing strategy: {v!r}")
        return v

    @field_validator("full_policy")
    @classmethod
    def _validate_policy(cls, v: str) -> str:
        if v not in ("overflow", "wait", "reject"):
            raise ValueError(f"unknown full_policy: {v!r}")
        return v


class AppConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    backends: List[BackendConfig]

    @model_validator(mode="after")
    def _check_backends(self) -> "AppConfig":
        if not self.backends:
            raise ValueError("at least one backend must be configured")
        seen = set()
        for b in self.backends:
            if b.name in seen:
                raise ValueError(f"duplicate backend name: {b.name!r}")
            seen.add(b.name)
        if not any(b.enabled for b in self.backends):
            raise ValueError("all backends are disabled; enable at least one")
        return self

    @property
    def enabled_backends(self) -> List[BackendConfig]:
        return [b for b in self.backends if b.enabled]


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load_config(path: str | os.PathLike) -> AppConfig:
    """Load, env-expand and validate the YAML configuration file."""
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"config file not found: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError(f"config root must be a mapping: {p}")
    data = _expand_env(raw)
    try:
        cfg = AppConfig(**data)
    except Exception as e:  # pydantic ValidationError and friends
        raise ConfigError(f"invalid configuration in {p}: {e}") from e
    logger.info(
        "loaded config from %s: %d backend(s), %d enabled",
        p, len(cfg.backends), len(cfg.enabled_backends),
    )
    return cfg
