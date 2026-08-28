"""Backend selection.

Strategies
----------
least_busy   pick the freshest backend with the fewest busy slots; ties
             broken by weight (higher first), then name (deterministic).
round_robin  rotate over fresh backends.
idle_aware   score-based, idle time first: prefer the GPU that has been
             resting longest (busy = idle 0, free = time free, never-used
             = infinite) so traffic rotates across all backends instead
             of hammering one.

Capacity policy (``routing.full_policy``)
-----------------------------------------
overflow   route anyway — llama.cpp queues internally (default, matches
           the reference design; never rejects on our side)
wait       wait up to ``wait_timeout`` s for a free slot, then overflow
reject     503 immediately when every fresh backend is at capacity
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from typing import Optional, Tuple

from .config import AppConfig
from .state import BackendRegistry, BackendState

logger = logging.getLogger("llm_proxy.router")


class NoBackendAvailable(Exception):
    """No fresh backend is available for routing."""


class BackendFull(Exception):
    """All fresh backends are at capacity and full_policy is 'reject'."""


class Router:
    def __init__(self, registry: BackendRegistry, cfg: AppConfig):
        self.registry = registry
        self.cfg = cfg
        self._rr_index = 0

    async def pick(self, model: Optional[str] = None) -> BackendState:
        """Select a backend AND reserve one in-flight slot on it.

        The reservation happens inside this method with no await between
        the decision and the increment — on a single-threaded event loop
        that makes pick+reserve atomic, so a burst of concurrent requests
        spreads across backends instead of all reading the same (stale)
        view. Call release() exactly once per successful pick.

        Model-based pinning: if ``model`` exactly matches the reported id
        of one or more fresh backends, candidates are restricted to those
        (the standard OpenAI ``model`` field doubles as a backend selector
        when each GPU is given a distinct ``--alias``). If the model
        matches no fresh backend — e.g. the pinned GPU is stale — the
        restriction is dropped and all fresh backends are eligible
        (fall back). A busy pinned GPU is then governed by ``full_policy``
        within the pinned set, so ``overflow`` lets llama.cpp queue it on
        the GPU the client asked for.
        """
        fresh = self.registry.fresh_states()
        if model:
            pinned = [s for s in fresh if model in s.models]
            if pinned:
                fresh = pinned
        if not fresh:
            raise NoBackendAvailable(
                "all backends stale or unresponsive "
                f"(stale_after={self.cfg.server.stale_after}s)"
            )

        strategy = self.cfg.routing.strategy
        if strategy == "round_robin":
            state = self._pick_rr(fresh)
        elif strategy == "idle_aware":
            state = self._pick_idle_aware(fresh)
        else:
            state = self._pick_least_busy(fresh)

        policy = self.cfg.routing.full_policy
        if state.at_capacity() and policy == "reject":
            # with least_busy, a full pick implies every backend is full;
            # with round_robin a full pick does not — fall back to one
            # with room before refusing.
            with_room = [s for s in fresh if not s.at_capacity()]
            if not with_room:
                raise BackendFull("all backends at slot capacity")
            state = min(with_room, key=lambda s: (s.effective_busy, -s.weight, s.name))
        elif state.at_capacity() and policy == "wait":
            state = await self._wait_for_room()

        state.in_flight += 1
        return state

    async def _wait_for_room(self) -> BackendState:
        """Wait up to wait_timeout for a free slot, then overflow."""
        deadline = time.monotonic() + self.cfg.routing.wait_timeout
        poll = self.cfg.routing.wait_poll_interval
        state = None
        while time.monotonic() < deadline:
            await asyncio.sleep(poll)
            fresh = self.registry.fresh_states()
            if not fresh:
                raise NoBackendAvailable("backends became stale while waiting")
            state = self._pick_least_busy(fresh)
            if not state.at_capacity():
                return state
        # timed out: overflow rather than fail the request
        logger.warning(
            "wait_timeout exceeded (%.1fs); overflowing to %s (busy=%d/%d)",
            self.cfg.routing.wait_timeout, state.name, state.busy, state.slots,
        )
        return state

    def release(self, state: BackendState) -> None:
        """Release the in-flight slot reserved by pick()."""
        state.in_flight = max(0, state.in_flight - 1)

    # ------------------------------------------------------------------

    def _pick_least_busy(self, fresh: list) -> BackendState:
        # effective_busy = observed busy + our in-flight dispatches, so a
        # burst of requests spread across backends instead of piling onto
        # whichever one the (<=poll_interval stale) metrics show first.
        return min(fresh, key=lambda s: (s.effective_busy, -s.weight, s.name))

    def _pick_idle_aware(self, fresh: list) -> BackendState:
        """Score-based selection, idle time first.

        Sort key (min; lower = preferred):
          1. -idle          — the most-idle GPU wins. A busy GPU reports
                              idle=0 (a GPU doing work is not resting), a
                              free GPU reports how long it has been free,
                              never-used reports infinity. Because busy
                              implies idle=0, free GPUs always sort ahead
                              of busy ones — no separate capacity key needed.
          2. effective_busy — tie-break (e.g. all busy): least loaded wins.
          3. -weight        — configured preference.
          4. name           — unique, so the order is total: min() returns
                              exactly one backend, deterministically.
        """
        def key(s: BackendState):
            idle = s.idle_seconds()
            idle_val = math.inf if idle is None else idle
            return (-idle_val, s.effective_busy, -s.weight, s.name)

        return min(fresh, key=key)

    def _pick_rr(self, fresh: list) -> BackendState:
        idx = self._rr_index % len(fresh)
        self._rr_index += 1
        return fresh[idx]

    def record_result(self, state: BackendState, ok: bool) -> None:
        state.last_routed_at = time.monotonic()
        if ok:
            state.requests_routed += 1
        else:
            state.requests_failed += 1
