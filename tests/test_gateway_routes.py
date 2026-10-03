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


UNSAFE_DESTINATIONS = [
    "http://api.beatapi.io/v1/systemone",
    "https://unapproved.example/v1/systemone",
    "https://api.beatapi.io.attacker.example/v1/systemone",
    "https://fake-user:fake-secret@api.beatapi.io/v1/systemone",
    "https://api.beatapi.io:443/v1/systemone",
    "https://api.beatapi.io/v1/systemone?secret=fake",
    "https://api.beatapi.io/v1/systemone#fragment",
    "https://api.beatapi.io/v1/systemone?",
    "https://api.beatapi.io/v1/systemone#",
    "https://api.beatapi.io/v1/systemone/",
    "https://api.beatapi.io/v1/../v1/systemone",
    "https://api.beatapi.io/v1/%73ystemone",
    "https://API.BEATAPI.IO/v1/systemone",
    "https://api.beatapi.io./v1/systemone",
    " https://api.beatapi.io/v1/systemone",
    "https://api.beatapi.io/v1/systemone\n",
    "https://api.beatapi.io\\@unapproved.example/v1/systemone",
    "https://127.0.0.1/v1/systemone",
    "https://[::1]/v1/systemone",
    "https://api.beatapi.io:bad/v1/systemone",
    "not-a-url", "", None,
]


@pytest.mark.parametrize("url", UNSAFE_DESTINATIONS)
def test_unsafe_destination_rejected_before_client_construction(monkeypatch, url):
    called = []
    monkeypatch.setattr(r.httpx, "Client", lambda **kwargs: called.append(kwargs))
    with pytest.raises(r.ConfigurationError, match="exact HTTPS policy") as exc:
        r.Native("beatapi", {"url": url, "model": "jev-1.13-free"}, "fake-test-key")
    assert called == []
    assert "fake-secret" not in str(exc.value)
    assert "unapproved.example" not in str(exc.value)


@pytest.mark.parametrize("route,name", [(3, "beatapi"), (4, "opencode-zen"), (5, "classifier")])
def test_settings_reject_destination_for_wrong_route(route, name):
    settings = {
        f"AI_GATEWAY_ROUTE_{route}_ENABLED": "true",
        f"AI_GATEWAY_ROUTE_{route}_MODEL": r.PIN if route == 5 else "jev-1.13-free",
        f"AI_GATEWAY_ROUTE_{route}_SYSTEMONE_URL": r.ROUTE_DESTINATIONS["gateway-1"],
    }
    with pytest.raises(r.ConfigurationError):
        r.configurations(settings)
    settings[f"AI_GATEWAY_ROUTE_{route}_SYSTEMONE_URL"] = r.ROUTE_DESTINATIONS[name]
    assert r.configurations(settings)[name]["url"] == r.ROUTE_DESTINATIONS[name]


def test_proposed_destinations_match_public_example():
    settings = r.dotenv_values(Path(r.__file__).parent / ".env.example")
    for route, name in [(3, "beatapi"), (4, "opencode-zen"), (5, "classifier")]:
        assert settings[f"AI_GATEWAY_ROUTE_{route}_SYSTEMONE_URL"] == r.ROUTE_DESTINATIONS[name]


@pytest.mark.parametrize("url", UNSAFE_DESTINATIONS)
def test_mutated_destination_rejected_before_state_serialization_or_transport(monkeypatch, url):
    calls = []
    original_client = httpx.Client
    monkeypatch.setattr(r.httpx, "Client", lambda **kwargs: original_client(
        transport=httpx.MockTransport(lambda request: calls.append(request)), **kwargs
    ))
    config = {"url": r.ROUTE_DESTINATIONS["beatapi"], "model": "jev-1.13-free"}
    backend = r.Native("beatapi", config, "fake-test-key")
    try:
        config["url"] = url
        assert backend.config["url"] == r.ROUTE_DESTINATIONS["beatapi"]
        backend.config["url"] = url
        # No program/state access is needed before rejecting destination drift.
        with pytest.raises(r.ConfigurationError):
            backend.evaluate(None, object())
        assert calls == []
    finally:
        backend.close()


@pytest.mark.parametrize("name", list(r.ROUTE_DESTINATIONS))
def test_exact_destination_dispatch_with_local_transport(monkeypatch, name):
    import json
    calls = []
    model = "typesafe-ai/jev" if name.startswith("gateway-") else (
        r.PIN if name == "classifier" else "jev-1.13-free"
    )

    def respond(request):
        calls.append(request)
        return httpx.Response(200, json={"answers": {}, "model": model})

    original_client = httpx.Client
    monkeypatch.setattr(r.httpx, "Client", lambda **kwargs: original_client(
        transport=httpx.MockTransport(respond), **kwargs
    ))
    backend = r.Native(name, {"url": r.ROUTE_DESTINATIONS[name], "model": model}, "fake-test-key")
    try:
        response = backend.evaluate(SimpleNamespace(model=r.PIN, questions={}), {"fixture": "synthetic"})
        assert response.model == model
        assert len(calls) == 1
        assert str(calls[0].url) == r.ROUTE_DESTINATIONS[name]
        assert calls[0].headers["authorization"] == "Bearer fake-test-key"
        assert json.loads(calls[0].content)["state"] == {"fixture": "synthetic"}
    finally:
        backend.close()


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_redirects_never_forward_credentials_or_study_state(monkeypatch, status):
    calls = []

    def redirect(request):
        calls.append(request)
        return httpx.Response(status, headers={"location": "https://unapproved.example/stolen"})

    original_client = httpx.Client
    monkeypatch.setattr(r.httpx, "Client", lambda **kwargs: original_client(
        transport=httpx.MockTransport(redirect), **kwargs
    ))
    backend = r.Native("beatapi", {
        "url": r.ROUTE_DESTINATIONS["beatapi"], "model": "jev-1.13-free"
    }, "fake-test-key")
    try:
        assert backend.client.follow_redirects is False
        backend.client.follow_redirects = True  # Per-dispatch policy still wins.
        with pytest.raises(r.HTTPFailure) as exc:
            backend.evaluate(SimpleNamespace(model=r.PIN, questions={}), {"fixture": "synthetic"})
        assert exc.value.status_code == status
        assert len(calls) == 1
        assert str(calls[0].url) == r.ROUTE_DESTINATIONS["beatapi"]
        assert r.failure(exc.value)[0] is False
    finally:
        backend.close()


def test_unknown_native_route_is_rejected_before_credentials(monkeypatch):
    calls = []
    monkeypatch.setattr(r.httpx, "Client", lambda **kwargs: calls.append(kwargs))
    with pytest.raises(r.ConfigurationError):
        r.Native("unknown", {"url": r.ROUTE_DESTINATIONS["beatapi"]}, "fake-test-key")
    assert calls == []


@pytest.mark.parametrize("route", [3, 4, 5])
@pytest.mark.parametrize("url", UNSAFE_DESTINATIONS)
def test_unsafe_destination_rejected_during_settings_construction(route, url):
    settings = {
        f"AI_GATEWAY_ROUTE_{route}_ENABLED": "true",
        f"AI_GATEWAY_ROUTE_{route}_MODEL": r.PIN if route == 5 else "jev-1.13-free",
        f"AI_GATEWAY_ROUTE_{route}_SYSTEMONE_URL": url,
    }
    with pytest.raises(r.ConfigurationError, match="exact HTTPS policy"):
        r.configurations(settings)


def test_destination_policy_rejects_accidental_in_process_reconfiguration():
    with pytest.raises(TypeError):
        r.ROUTE_DESTINATIONS["beatapi"] = "https://unapproved.example/v1/systemone"
    assert r.validate_destination("beatapi", "https://api.beatapi.io/v1/systemone") == (
        "https://api.beatapi.io/v1/systemone"
    )
