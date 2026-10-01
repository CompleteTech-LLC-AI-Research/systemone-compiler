import sys
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime, timezone

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "ai-gateway"))
import paced_routes as r  # noqa: E402
from typewright.backends import ManagedBackend, Response  # noqa: E402
from typewright.errors import BudgetExceeded, BackendError  # noqa: E402


def test_retry_after_numbers_dates_and_invalid_values():
    assert r.retry_after("60") == 60
    now = datetime(2026, 9, 25, tzinfo=timezone.utc).timestamp()
    assert r.retry_after("Fri, 25 Sep 2026 00:01:00 GMT", now) == 60
    for value in ("bad", "-1", "NaN", "inf", None):
        assert r.retry_after(value) is None


def test_shared_cooldown_and_slow_route_does_not_block_ready_route():
    now = [100.0]
    configs = {
        "v1": {"concurrency": 1, "interval": 0.1, "group": "vercel"},
        "v2": {"concurrency": 1, "interval": 0.1, "group": "vercel"},
        "beat": {"concurrency": 1, "interval": 60, "group": "beat", "after_completion": True},
        "direct": {"concurrency": 1, "interval": 0.05, "group": "direct"},
    }
    s = r.Scheduler(configs, lambda: now[0])
    assert s.take_ready() == "v1"
    s.release("v1", 60, True)
    assert s.take_ready() == "beat"
    now[0] += 3
    s.release("beat")
    assert s.next["beat"] == 163
    assert s.take_ready() == "direct"
    assert s.take_ready() is None
    s.release("direct")
    now[0] = 160
    assert s.take_ready() == "v2"


def test_wrapped_http_errors_keep_status_and_cooldown():
    response = httpx.Response(429, headers={"retry-after": "60"})
    cause = r.HTTPFailure(response)
    wrapped = BackendError("sanitized")
    wrapped.__cause__ = cause
    assert r.failure(wrapped) == (True, 60, 429)
    for code in (400, 401, 402, 403, 404):
        assert r.failure(r.HTTPFailure(httpx.Response(code)))[0] is False
    pytest.importorskip("typesafe_sdk")
    from typesafe_sdk._core.errors import TypeSafeAPIError
    import httpx2

    native = TypeSafeAPIError(429, {}, httpx2.Headers({"retry-after-ms": "61000"}))
    wrapped.__cause__ = native
    assert r.failure(wrapped) == (True, 61, 429)


def test_retry_reserves_budget_and_never_scores_errors():
    configs = {name: {"concurrency": 1, "interval": 0.001, "group": name} for name in ("a", "b", "c")}
    counts = []

    class Fake:
        synthetic = False

        def __init__(self, name):
            self.identity = name

        def evaluate(self, p, s):
            counts.append(self.identity)
            if self.identity == "a":
                raise r.HTTPFailure(httpx.Response(429, headers={"retry-after": "60"}))
            return Response(answers={}, model=r.PIN, usage={})

        def close(self):
            pass

    factories = {
        name: lambda n, name=name: ManagedBackend(Fake(name), max_calls=n, cache=None) for name in configs
    }
    b = r.PacedBackend(2, None, configs, r.Scheduler(configs), factories)
    try:
        assert b.evaluate(SimpleNamespace(model=r.PIN, questions={}), {}).model == r.PIN
        assert counts == ["a", "b"]
        assert b.budget.used == 2 and b.responses == 1 and b.transient_retries == 1
        with pytest.raises(BudgetExceeded):
            b.evaluate(SimpleNamespace(model=r.PIN, questions={}), {})
        assert counts == ["a", "b"]
    finally:
        b.close()


def test_concurrency_cap_under_contention():
    from concurrent.futures import ThreadPoolExecutor
    import threading
    import time

    configs = {"one": {"concurrency": 2, "interval": 0.001, "group": "one"}}
    s = r.Scheduler(configs)
    peak = [0]
    lock = threading.Lock()

    def call(_):
        name = s.acquire(lambda *args: None)
        with lock:
            peak[0] = max(peak[0], s.active[name])
        time.sleep(0.005)
        s.release(name)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(call, range(24)))
    assert peak[0] <= 2
    assert s.active["one"] == 0


def test_alias_rejects_model_drift():
    backend = r.ManagedAlias(
        SimpleNamespace(identity="fixture", synthetic=False, config={"model": "jev-1.13-free"}),
        max_calls=1,
        cache=None,
    )
    with pytest.raises(BackendError):
        backend._validate_identity(
            SimpleNamespace(model=r.PIN), Response(answers={}, model="other", usage={})
        )


def test_question_quota_persists_and_exhaustion_skips_route(tmp_path):
    now = [100.0]
    configs = {
        "classifier": {
            "concurrency": 2,
            "interval": 0.1,
            "group": "classifier",
            "questions_per_second": 50,
            "questions_daily": 30,
        },
        "direct": {"concurrency": 1, "interval": 0.05, "group": "direct"},
    }
    path = tmp_path / "quota.json"
    s = r.Scheduler(configs, lambda: now[0], path)
    assert s.take_ready(28) == "classifier"
    assert s.next["classifier"] == 100.56
    s.release("classifier")
    now[0] += 1
    reloaded = r.Scheduler(configs, lambda: now[0], path)
    assert reloaded.take_ready(3) == "direct"
    assert reloaded.quota["classifier"]["used"] == 28


def test_retry_limit_is_three_attempts():
    configs = {n: {"concurrency": 1, "interval": 0.001, "group": n} for n in ("a", "b", "c")}

    class Fake:
        synthetic = False
        identity = "fixture"

        def evaluate(self, p, s):
            raise r.HTTPFailure(httpx.Response(503))

        def close(self):
            pass

    b = r.PacedBackend(
        10,
        None,
        configs,
        r.Scheduler(configs),
        {n: lambda maximum: ManagedBackend(Fake(), max_calls=maximum, cache=None) for n in configs},
    )
    try:
        with pytest.raises(r.HTTPFailure):
            b.evaluate(SimpleNamespace(model=r.PIN, questions={}), {})
        assert b.budget.used == 3 and b.responses == 0
    finally:
        b.close()


def test_weighted_sliding_minute_limit(tmp_path):
    configs = {
        "classifier": {
            "concurrency": 2,
            "interval": 0.1,
            "group": "classifier",
            "questions_daily": 20000,
            "questions_per_minute": 3000,
        }
    }
    now = [1.0]
    s = r.Scheduler(configs, lambda: now[0], tmp_path / "quota.json")
    assert s.take_ready(2990) == "classifier"
    s.release("classifier")
    now[0] += 1
    assert s.take_ready(11) is None
    assert s.take_ready(10) == "classifier"


def test_http_520_is_transient():
    assert r.failure(r.HTTPFailure(httpx.Response(520)))[0] is True


def test_continuous_dispatch_refills_before_slowest_finishes():
    import threading
    from concurrent.futures import ThreadPoolExecutor

    release = threading.Event()
    refilled = threading.Event()
    configs = {"fixture": {"concurrency": 8, "interval": 0.001, "group": "fixture"}}
    backend = r.PacedBackend(20, None, configs, r.Scheduler(configs), {})

    def work(index):
        if index == 0:
            assert release.wait(5)
        if index == 8:
            refilled.set()
        return index

    try:
        with ThreadPoolExecutor(max_workers=1) as observer:
            future = observer.submit(backend.map, work, range(16))
            try:
                assert refilled.wait(2), "A completed slot was not refilled while request zero waited"
            finally:
                release.set()
            assert future.result(timeout=5) == list(range(16))
    finally:
        backend.close()


def test_only_exhausted_calibration_is_deferred():
    backend = SimpleNamespace(budget=SimpleNamespace(used=10, maximum=10))
    assert r.calibration_budget_exhausted(BudgetExceeded("fixture"), "calibration", backend)
    assert not r.calibration_budget_exhausted(BudgetExceeded("fixture"), "selection", backend)
    assert not r.calibration_budget_exhausted(BackendError("fixture"), "calibration", backend)
    backend.budget.used = 9
    assert not r.calibration_budget_exhausted(BudgetExceeded("fixture"), "calibration", backend)


def test_calibration_factory_uses_only_native_and_keeps_budget(tmp_path):
    from typewright.backends import MockBackend

    launch = SimpleNamespace(
        make_backend=lambda paid, n: ManagedBackend(MockBackend(), max_calls=n, cache=None)
    )
    backend = r.make_calibration_backend(13, launch, allow_paid=True)
    try:
        assert backend.budget.maximum == 13
        assert list(backend.configs) == ["direct"]
        assert backend.workers == 8
    finally:
        backend.launch = None
        backend.close()


def test_cooldown_is_capped_so_a_hostile_retry_after_cannot_freeze_a_group():
    now = [100.0]
    configs = {"v1": {"concurrency": 1, "interval": 0.1, "group": "vercel"},
               "v2": {"concurrency": 1, "interval": 0.1, "group": "vercel"}}
    s = r.Scheduler(configs, lambda: now[0])
    assert s.take_ready() == "v1"
    s.release("v1", 10 ** 9, True)
    assert s.groups["vercel"] == 100.0 + r.MAX_COOLDOWN_SECONDS
    now[0] += r.MAX_COOLDOWN_SECONDS + 1
    assert s.take_ready() == "v2"


@pytest.mark.parametrize("consent", [None, False, "true", 1, 0])
def test_paid_backends_refuse_without_explicit_boolean_consent(consent):
    from typewright.errors import ConfigurationError
    called = []
    launch = SimpleNamespace(make_backend=lambda paid, n: called.append(paid))
    kwargs = {} if consent is None else {"allow_paid": consent}
    with pytest.raises(ConfigurationError, match="explicit consent"):
        r.make_backend(5, launch, **kwargs)
    with pytest.raises(ConfigurationError, match="explicit consent"):
        r.make_calibration_backend(5, launch, **kwargs)
    assert called == []


def test_consent_is_forwarded_to_the_direct_backend_not_hard_coded():
    from typewright.backends import MockBackend
    received = []

    def make(paid, n):
        received.append(paid)
        return ManagedBackend(MockBackend(), max_calls=n, cache=None)

    backend = r.make_calibration_backend(3, SimpleNamespace(make_backend=make), allow_paid=True)
    try:
        backend.launch = None
        backend.factories["direct"](3).close()
    finally:
        backend.close()
    assert received == [True]


def test_every_gateway_route_including_one_and_two_needs_its_enable_flag():
    assert list(r.configurations({})) == ["direct"]
    assert list(r.configurations({"AI_GATEWAY_ROUTE_1_ENABLED": "yes"})) == ["direct"]
    enabled = r.configurations({"AI_GATEWAY_ROUTE_1_ENABLED": "true", "AI_GATEWAY_ROUTE_2_ENABLED": "true"})
    assert list(enabled) == ["direct", "gateway-1", "gateway-2"]
    only_two = r.configurations({"AI_GATEWAY_ROUTE_2_ENABLED": "true"})
    assert list(only_two) == ["direct", "gateway-2"]
