import asyncio

import pytest

from llm_proxy.config import AppConfig
from llm_proxy.router import BackendFull, NoBackendAvailable, Router
from llm_proxy.state import BackendRegistry, BackendState


def make_cfg(strategy="least_busy", full_policy="overflow", stale_after=3.0):
    return AppConfig(
        server={"stale_after": stale_after, "poll_interval": 1.0},
        routing={"strategy": strategy, "full_policy": full_policy,
                 "wait_timeout": 5.0, "wait_poll_interval": 0.01},
        backends=[
            {"name": "a", "url": "http://a:8080", "slots": 4},
            {"name": "b", "url": "http://b:8080", "slots": 4},
        ],
    )


def make_router(**kw):
    cfg = make_cfg(**kw)
    reg = BackendRegistry(cfg)
    for s in reg.all_states():
        s.last_update = _now()
    return reg, Router(reg, cfg)


def _now():
    import time
    return time.monotonic()


def fresh(reg, name, busy=0, slots=None):
    s = reg.states[name]
    s.busy = busy
    s.last_update = _now()
    if slots:
        s.slots = slots
    return s


def set_idle(reg, name, seconds):
    """Mark backend as having completed a request `seconds` ago."""
    s = reg.states[name]
    s.last_completed_at = _now() - seconds
    return s


def stale(reg, name, age=10.0):
    s = reg.states[name]
    s.last_update = _now() - age
    return s


async def test_least_busy_picks_fewest_busy():
    reg, router = make_router()
    fresh(reg, "a", busy=3)
    fresh(reg, "b", busy=1)
    assert (await router.pick()).name == "b"


async def test_tie_broken_by_weight_then_name():
    cfg = make_cfg()
    cfg.backends[1].weight = 2.0
    reg = BackendRegistry(cfg)
    for s in reg.all_states():
        s.last_update = _now()
    router = Router(reg, cfg)
    fresh(reg, "a", busy=1)
    fresh(reg, "b", busy=1)
    assert (await router.pick()).name == "b"  # higher weight


async def test_stale_excluded():
    reg, router = make_router(stale_after=3.0)
    fresh(reg, "a", busy=0)
    stale(reg, "b", age=10.0)
    assert (await router.pick()).name == "a"


async def test_all_stale_raises():
    reg, router = make_router(stale_after=3.0)
    stale(reg, "a")
    stale(reg, "b")
    with pytest.raises(NoBackendAvailable):
        await router.pick()


async def test_round_robin_rotates():
    reg, router = make_router(strategy="round_robin")
    fresh(reg, "a")
    fresh(reg, "b")
    picks = [(await router.pick()).name for _ in range(4)]
    assert picks == ["a", "b", "a", "b"]


async def test_reject_policy_503_when_all_full():
    reg, router = make_router(full_policy="reject")
    fresh(reg, "a", busy=4)
    fresh(reg, "b", busy=4)
    with pytest.raises(BackendFull):
        await router.pick()


async def test_reject_policy_prefers_backend_with_room():
    reg, router = make_router(strategy="round_robin", full_policy="reject")
    fresh(reg, "a", busy=4)
    fresh(reg, "b", busy=2)
    # rr may pick full 'a' first; policy must fall back to 'b'
    assert (await router.pick()).name == "b"


async def test_wait_policy_waits_for_free_slot():
    reg, router = make_router(full_policy="wait")
    fresh(reg, "a", busy=4)
    fresh(reg, "b", busy=4)

    async def free_later():
        await asyncio.sleep(0.05)
        reg.states["a"].busy = 2

    task = asyncio.create_task(free_later())
    picked = await router.pick()
    await task
    assert picked.name == "a"


async def test_wait_policy_timeout_overflows():
    reg, router = make_router(full_policy="wait")
    cfg = router.cfg
    cfg.routing.wait_timeout = 0.05
    fresh(reg, "a", busy=4)
    fresh(reg, "b", busy=4)
    picked = await router.pick()  # should overflow, not raise
    assert picked.name in ("a", "b")


async def test_in_flight_affects_selection():
    """Metrics are stale (both show busy=0), but the proxy has already
    dispatched 3 in-flight requests to 'a' — new traffic must go to 'b'."""
    reg, router = make_router()
    fresh(reg, "a", busy=0)
    fresh(reg, "b", busy=0)
    reg.states["a"].in_flight = 3
    assert (await router.pick()).name == "b"


async def test_pick_reserves_and_release_frees():
    """pick() atomically reserves an in-flight slot; release() frees it.
    A burst of picks must therefore spread across backends."""
    reg, router = make_router()
    fresh(reg, "a", busy=0)
    fresh(reg, "b", busy=0)
    picked = [await router.pick() for _ in range(4)]
    names = [s.name for s in picked]
    assert names.count("a") == 2 and names.count("b") == 2, names
    for s in picked:
        router.release(s)
    assert reg.states["a"].in_flight == 0
    assert reg.states["b"].in_flight == 0


async def test_in_flight_capacity():
    reg, router = make_router()
    fresh(reg, "a", busy=0)
    fresh(reg, "b", busy=0)
    reg.states["a"].in_flight = 4  # slots=4 -> effectively full
    assert reg.states["a"].at_capacity()
    assert (await router.pick()).name == "b"


async def test_overflow_policy_never_raises():
    reg, router = make_router(full_policy="overflow")
    fresh(reg, "a", busy=4)
    fresh(reg, "b", busy=4)
    picked = await router.pick()
    assert picked.busy == 4


# ---------------------------------------------------------------------------
# idle_aware strategy
# ---------------------------------------------------------------------------

async def test_idle_aware_prefers_most_idle():
    reg, router = make_router(strategy="idle_aware")
    fresh(reg, "a"); set_idle(reg, "a", 10)    # free, idle 10 s
    fresh(reg, "b"); set_idle(reg, "b", 100)   # free, idle 100 s
    assert (await router.pick()).name == "b"


async def test_idle_aware_never_used_wins():
    reg, router = make_router(strategy="idle_aware")
    fresh(reg, "a"); set_idle(reg, "a", 1000)  # used long ago
    fresh(reg, "b")                            # never used -> infinite idle
    assert reg.states["b"].idle_seconds() is None
    assert (await router.pick()).name == "b"


async def test_idle_aware_busy_reports_zero_and_loses():
    """A busy GPU is idle=0 even if its last completion was long ago, so a
    free GPU (however recently used) wins."""
    reg, router = make_router(strategy="idle_aware")
    fresh(reg, "a"); set_idle(reg, "a", 5)              # free, idle 5 s
    fresh(reg, "b"); set_idle(reg, "b", 100)            # busy, last done 100 s ago
    reg.states["b"].in_flight = 1
    assert reg.states["b"].idle_seconds() == 0.0
    assert (await router.pick()).name == "a"


async def test_idle_aware_all_busy_picks_least_busy():
    reg, router = make_router(strategy="idle_aware")
    fresh(reg, "a", busy=1); set_idle(reg, "a", 50)
    fresh(reg, "b", busy=3); set_idle(reg, "b", 50)
    # both busy -> idle 0 -> tie -> least effective_busy wins
    assert (await router.pick()).name == "a"


async def test_idle_aware_tie_is_deterministic_name():
    """Two identical never-used backends: the unique name makes the order
    total, so min() returns exactly one, deterministically (alphabetical).
    Releasing between picks keeps the starting state identical each time."""
    reg, router = make_router(strategy="idle_aware")
    fresh(reg, "a")  # never used
    fresh(reg, "b")  # never used
    for _ in range(3):
        p = await router.pick()
        assert p.name == "a"  # same state -> same pick, always
        router.release(p)


async def test_idle_aware_rotates_after_dispatch():
    """After dispatching to the most-idle GPU, in_flight drops it to
    idle=0, so the next request moves to a different GPU."""
    reg, router = make_router(strategy="idle_aware")
    fresh(reg, "a")  # never used
    fresh(reg, "b")  # never used
    first = await router.pick()      # 'a' (name tie-break)
    second = await router.pick()     # 'a' now idle=0 -> 'b'
    assert first.name == "a"
    assert second.name == "b"
    router.release(first)
    router.release(second)


# ---------------------------------------------------------------------------
# model-based pinning (exact id match)
# ---------------------------------------------------------------------------

async def test_model_pin_restricts_candidates():
    reg, router = make_router(strategy="idle_aware")
    fresh(reg, "a"); reg.states["a"].models = ["model-a"]
    fresh(reg, "b"); reg.states["b"].models = ["model-b"]
    assert (await router.pick(model="model-b")).name == "b"
    assert (await router.pick(model="model-a")).name == "a"
    # unknown model -> no pin -> normal routing over all
    assert (await router.pick(model="unknown")).name in ("a", "b")


async def test_model_pin_falls_back_when_pinned_stale():
    """Pinned GPU is stale -> it's not in `fresh`, pin yields nothing, so we
    fall back to all fresh backends instead of erroring."""
    reg, router = make_router(strategy="idle_aware", stale_after=3.0)
    fresh(reg, "a"); reg.states["a"].models = ["model-a"]
    stale(reg, "b", age=10.0); reg.states["b"].models = ["model-b"]
    # 'model-b' is only served by the stale 'b' -> fall back to 'a'
    assert (await router.pick(model="model-b")).name == "a"


async def test_model_pin_busy_honors_full_policy_overflow():
    """A busy pinned GPU is still the pinned GPU: overflow routes to it
    (llama.cpp queues) rather than silently substituting another backend."""
    reg, router = make_router(strategy="idle_aware", full_policy="overflow")
    fresh(reg, "a", busy=1); reg.states["a"].models = ["model-a"]
    fresh(reg, "b", busy=0); reg.states["b"].models = ["model-b"]
    # pin to busy 'a' -> still 'a' (overflow), not 'b'
    assert (await router.pick(model="model-a")).name == "a"


def test_idle_seconds_reflects_state():
    cfg = make_cfg()
    reg = BackendRegistry(cfg)
    s = reg.states["a"]
    # never used + free -> None
    assert s.idle_seconds() is None
    # completed 30 s ago, free -> ~30
    s.last_completed_at = _now() - 30
    assert 29 <= s.idle_seconds() <= 31
    # busy -> 0 regardless of last completion
    s.in_flight = 1
    assert s.idle_seconds() == 0.0


def test_state_helpers():
    cfg = make_cfg()
    reg = BackendRegistry(cfg)
    s = reg.states["a"]
    s.busy, s.slots = 3, 4
    assert s.free_slots() == 1
    assert not s.at_capacity()
    s.busy = 4
    assert s.at_capacity()
    snap = s.snapshot(stale_after=3.0)
    assert snap["name"] == "a" and snap["free"] == 0
