"""In-memory backend state and the metrics poller.

Design (from the reference document):
  * background polling, never per-request — polling adds zero latency to
    inference calls;
  * a staleness guard — a backend that stops answering /metrics is
    excluded from selection after ``stale_after`` seconds;
  * no persistent state — restart is a clean slate, which is acceptable
    because LLM routing needs no durable state.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import httpx

from .config import AppConfig

logger = logging.getLogger("llm_proxy.state")


@dataclass
class BackendState:
    name: str
    url: str
    slots: int
    weight: float
    api_key: Optional[str]
    busy: int = 0
    last_update: float = 0.0
    last_error: Optional[str] = None
    models: List[str] = field(default_factory=list)
    # requests this proxy has dispatched and not yet completed. The
    # metrics sample is up to poll_interval old, so in-flight dispatches
    # are invisible to it — routing uses busy + in_flight (an upper
    # bound on true load) to spread burst traffic across backends.
    in_flight: int = 0
    # monotonic time of the last request COMPLETED on this backend
    # (0 = never used). Idle-preference routing keys on this.
    last_completed_at: float = 0.0
    # counters for /proxy/metrics
    requests_routed: int = 0
    requests_failed: int = 0
    last_routed_at: float = 0.0

    def mark_completed(self) -> None:
        """Call when a request finishes on this backend; resets its idle
        clock (a completed request is when the GPU is genuinely free)."""
        self.last_completed_at = time.monotonic()

    def idle_seconds(self) -> Optional[float]:
        """How long this backend has been FREE (no active requests).

        A busy GPU is, by definition, not idle — it reports 0. A free GPU
        reports how long it has been resting; None means never used (and
        currently free), which the router treats as infinitely idle.
        """
        if self.effective_busy > 0:
            return 0.0
        if not self.last_completed_at:
            return None
        return time.monotonic() - self.last_completed_at

    @property
    def age(self) -> Optional[float]:
        """Seconds since the last successful poll; None if never polled."""
        if not self.last_update:
            return None
        return time.monotonic() - self.last_update

    def is_stale(self, stale_after: float) -> bool:
        """Stale if the last successful sample is older than stale_after,
        or if we have at least one failed poll and no sample yet (the first
        poll cycle runs at startup, so this is decided within ~1 s)."""
        if self.last_update:
            return time.monotonic() - self.last_update >= stale_after
        return self.last_error is not None

    @property
    def effective_busy(self) -> int:
        """Upper bound on true load: observed busy + our in-flight
        dispatches (which the metrics sample may not have seen yet)."""
        return self.busy + self.in_flight

    def free_slots(self) -> int:
        return max(0, self.slots - self.effective_busy)

    def at_capacity(self) -> bool:
        return self.effective_busy >= self.slots

    def snapshot(self, stale_after: float) -> dict:
        last_routed = None
        if self.last_routed_at:
            # convert the monotonic stamp to epoch seconds for readability
            last_routed = time.time() - (time.monotonic() - self.last_routed_at)
        idle = self.idle_seconds()
        return {
            "name": self.name,
            "url": self.url,
            "slots": self.slots,
            "busy": self.busy,
            "in_flight": self.in_flight,
            "free": self.free_slots(),
            "weight": self.weight,
            "stale": self.is_stale(stale_after),
            "age_s": round(self.age, 2) if self.age is not None else None,
            "idle_s": round(idle, 2) if idle is not None else None,
            "last_error": self.last_error,
            "models": list(self.models),
            "requests_routed": self.requests_routed,
            "requests_failed": self.requests_failed,
            "last_routed_at": last_routed,
        }


class BackendRegistry:
    """Thread-safe (single-loop) registry of backend states."""

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.states: Dict[str, BackendState] = {
            b.name: BackendState(
                name=b.name,
                url=b.url,
                slots=b.slots,
                weight=b.weight,
                api_key=b.api_key,
            )
            for b in cfg.enabled_backends
        }
        self.lock = asyncio.Lock()

    def fresh_states(self) -> List[BackendState]:
        stale_after = self.cfg.server.stale_after
        return [s for s in self.states.values() if not s.is_stale(stale_after)]

    def all_states(self) -> List[BackendState]:
        return list(self.states.values())


# ---------------------------------------------------------------------------
# poller
# ---------------------------------------------------------------------------

# metrics that indicate load, by llama.cpp version:
#   new builds: llamacpp:requests_processing (generating now)
#               llamacpp:requests_deferred   (waiting for a free slot)
#   old builds: llama_slots_busy             (slots generating now)
_BUSY_METRIC_NAMES = (
    "llamacpp:requests_processing",
    "llamacpp:requests_deferred",
    "llama_slots_busy",
)


def _parse_busy_metrics_sync(text: str) -> Optional[int]:
    """Extract the busy count from llama.cpp Prometheus text output.

    Busy = total pending work on the backend:
      new builds: requests_processing + requests_deferred
      old builds: llama_slots_busy
    Returns None when neither metric family is present.
    """
    # tolerate a JSON-string-wrapped body (defensive; real backends
    # serve plain Prometheus text)
    if text.startswith('"') and text.endswith('"'):
        try:
            import json
            text = json.loads(text)
        except Exception:
            pass

    values: Dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        name, sep, raw = line.rpartition(" ")
        if sep and name in _BUSY_METRIC_NAMES:
            try:
                values[name] = float(raw)
            except ValueError:
                pass

    if "llamacpp:requests_processing" in values:
        return int(values["llamacpp:requests_processing"]
                   + values.get("llamacpp:requests_deferred", 0.0))
    if "llama_slots_busy" in values:
        return int(values["llama_slots_busy"])
    return None

# async for parse_busy_metrics
# loading json, parsing and iterating might run too long for the loop
async def _parse_busy_metrics(text: str) -> Optional[int]:
    return await asyncio.to_thread(_parse_busy_metrics_sync, text)

class MetricsPoller:
    """Polls /metrics on every enabled backend every ``poll_interval`` s."""

    def __init__(self, registry: BackendRegistry, client: httpx.AsyncClient):
        self.registry = registry
        self.client = client
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="metrics-poller")

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass

    async def _run(self) -> None:
        interval = self.registry.cfg.server.poll_interval
        while not self._stop.is_set():
            started = time.monotonic()
            await self.refresh_once()
            elapsed = time.monotonic() - started
            # sleep out the remainder of the interval
            try:
                await asyncio.wait_for(self._stop.wait(), max(0.0, interval - elapsed))
            except asyncio.TimeoutError:
                pass

    async def refresh_once(self) -> None:
        cfg = self.registry.cfg
        timeout = cfg.server.poll_timeout
        now = time.monotonic()

        async def fetch(state: BackendState) -> None:
            try:
                r = await self.client.get(f"{state.url}/metrics", timeout=timeout)
                if r.status_code != 200:
                    state.last_error = f"metrics HTTP {r.status_code}"
                    if r.status_code == 501:
                        logger.error("metrics poll failed for %s: HTTP %s -- %s", state.name, r.status_code, r.json().get("error", {}).get("message"))
                    return
                busy = await _parse_busy_metrics(r.text)
                if busy is None:
                    state.last_error = (
                        "no recognized busy metric in /metrics "
                        "(expected llamacpp:requests_processing or "
                        "llama_slots_busy)"
                    )
                    return
                state.busy = busy
                state.last_update = now
                state.last_error = None
                # opportunistic model discovery for the /v1/models merge
                if cfg.server.merge_models:
                    try:
                        mr = await self.client.get(f"{state.url}/v1/models", timeout=timeout)
                        if mr.status_code == 200:
                            data = mr.json()
                            state.models = [m.get("id") for m in data.get("data", []) if m.get("id")]
                    except Exception:
                        pass
            except Exception as e:
                state.last_error = f"{type(e).__name__}: {e}"
                logger.debug("metrics poll failed for %s: %s", state.name, e)

        async with self.registry.lock:
            states = list(self.registry.states.values())
        await asyncio.gather(*(fetch(s) for s in states))
