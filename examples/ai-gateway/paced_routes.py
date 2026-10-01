"""Study-only paced native routes. No cached inputs or changes to request budgets."""

import math
import random
import threading
import time
import json
import os
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import httpx
from dotenv import dotenv_values
from s1compiler.backends import ManagedBackend, Response
from s1compiler.errors import BackendError
from s1compiler.io import atomic_json

from parallel_backend import RoutedParallel

PIN = "jev-1.13.0"
# A hostile or buggy Retry-After must not freeze a whole route group indefinitely.
MAX_COOLDOWN_SECONDS = 600


def retry_after(value, now=None):
    if not value:
        return None
    try:
        seconds = float(value)
        return seconds if math.isfinite(seconds) and seconds >= 0 else None
    except (ValueError, TypeError):
        try:
            return max(0, parsedate_to_datetime(value).timestamp() - (time.time() if now is None else now))
        except (ValueError, TypeError, OverflowError):
            return None


class HTTPFailure(BackendError):
    def __init__(self, response):
        super().__init__(f"Provider request failed (HTTP {response.status_code}).")
        self.status_code = response.status_code
        self.retry_after = retry_after(response.headers.get("retry-after"))


def failure(exc):
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        response = getattr(exc, "response", None)
        status = getattr(exc, "status_code", getattr(exc, "status", None))
        if status is None and response is not None:
            status = response.status_code
        if status is not None:
            delay = getattr(exc, "retry_after", None)
            headers = getattr(exc, "headers", None)
            if headers is None and response is not None:
                headers = response.headers
            if delay is None and headers is not None:
                delay = retry_after(headers.get("retry-after"))
                milliseconds = retry_after(headers.get("retry-after-ms"))
                if milliseconds is not None:
                    delay = max(delay or 0, milliseconds / 1000)
            return status in {408, 429, 500, 502, 503, 504, 520, 529}, delay, status
        if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)) or type(exc).__name__ in {
            "TypeSafeAPITimeoutError",
            "TypeSafeAPIConnectionError",
        }:
            return True, None, None
        exc = exc.__cause__ or exc.__context__
    return False, None, None


class Scheduler:
    def __init__(self, configs, clock=time.monotonic, quota_path=None):
        self.configs = configs
        self.clock = clock
        self.condition = threading.Condition()
        self.active = {name: 0 for name in configs}
        self.next = {name: 0.0 for name in configs}
        self.groups = {}
        self.order = list(configs)
        self.throttles = {name: 0 for name in configs}
        self.quota_path = quota_path
        self.quota = json.loads(quota_path.read_text()) if quota_path and quota_path.exists() else {}

    def take_ready(self, questions=1):
        """Caller holds condition. Rotate fairly among eligible routes only."""
        now = self.clock()
        for name in self.order:
            c = self.configs[name]
            if self.active[name] >= c["concurrency"]:
                continue
            if now < max(self.next[name], self.groups.get(c["group"], 0)):
                continue
            if c.get("questions_daily"):
                day = datetime.now(timezone.utc).date().isoformat()
                entry = self.quota.get(name, {})
                used = entry.get("used", 0) if entry.get("day") == day else 0
                if used + questions > c["questions_daily"]:
                    continue
                wall_now = time.time()
                recent = [item for item in entry.get("recent", []) if item[0] > wall_now - 60]
                if sum(item[1] for item in recent) + questions > c.get("questions_per_minute", float("inf")):
                    continue
                self.quota[name] = {
                    "day": day,
                    "used": used + questions,
                    "recent": recent + [[wall_now, questions]],
                }
                if self.quota_path:
                    atomic_json(self.quota_path, self.quota)
            self.active[name] += 1
            self.next[name] = now + max(
                c["interval"], questions / c["questions_per_second"] if c.get("questions_per_second") else 0
            )
            self.groups[c["group"]] = now + c.get("group_interval", 0)
            self.order.remove(name)
            self.order.append(name)
            return name
        return None

    def acquire(self, event, questions=1):
        last = self.clock()
        with self.condition:
            while True:
                name = self.take_ready(questions)
                if name is not None:
                    return name
                self.condition.wait(0.1)
                if self.clock() - last >= 10:
                    event("rate_wait", 0)
                    last = self.clock()

    def release(self, name, cooldown=0, throttled=False):
        with self.condition:
            c = self.configs[name]
            self.active[name] -= 1
            if c.get("after_completion"):
                self.next[name] = max(self.next[name], self.clock() + c["interval"])
            if cooldown:
                group = c["group"]
                self.groups[group] = max(self.groups.get(group, 0),
                                         self.clock() + min(cooldown, MAX_COOLDOWN_SECONDS))
            self.throttles[name] += int(throttled)
            self.condition.notify_all()


class Native:
    synthetic = False

    def __init__(self, name, config, key):
        self.identity = name
        self.config = config
        self.provider_attempts = 0
        self.providers = set()
        self.client = httpx.Client(headers={"Authorization": "Bearer " + key}, timeout=60)

    def evaluate(self, program, state):
        if program.model != PIN:
            raise BackendError("Unapproved program model.")
        payload = {
            "model": self.config["model"],
            "state": state,
            "questions": {k: q.wire() for k, q in program.questions.items()},
        }
        if self.identity.startswith("gateway-"):
            payload["providerOptions"] = {"gateway": {"only": ["typesafe-ai"]}}
        start = time.perf_counter()
        response = self.client.post(self.config["url"], json=payload)
        if response.status_code != 200:
            raise HTTPFailure(response)
        body = response.json()
        routing = body.get("provider_metadata", {}).get("gateway", {}).get("routing", {})
        attempts = routing.get("totalProviderAttemptCount", 1)
        if type(attempts) is not int or attempts < 1:
            raise BackendError("Invalid gateway attempt accounting.")
        self.provider_attempts += attempts
        self.providers.add(routing.get("finalProvider", self.identity))
        return Response(
            answers=body["answers"],
            model=body["model"],
            usage=body.get("usage") or {},
            latency_ms=(time.perf_counter() - start) * 1000,
        )

    def close(self):
        self.client.close()


class ManagedAlias(ManagedBackend):
    def evaluate(self, program, state):
        before = self.backend.provider_attempts
        try:
            return super().evaluate(program, state)
        finally:
            for _ in range(max(0, self.backend.provider_attempts - before - 1)):
                self.budget.reserve()

    def _validate_identity(self, program, response):
        if program.model != PIN or response.model != self.backend.config["model"] or response.synthetic:
            raise BackendError("Provider/model identity differs from approved mapping.")

    def accounting(self):
        result = super().accounting()
        result.update(
            provider_attempts_reported=self.backend.provider_attempts,
            actual_providers=sorted(self.backend.providers),
        )
        return result


class PacedBackend(RoutedParallel):
    def __init__(self, limit, launch, configs, scheduler, factories):
        super().__init__(limit, None, launch=launch, workers=8)
        self.configs, self.scheduler, self.factories = configs, scheduler, factories
        self.last_telemetry = 0.0

    def telemetry(self, final=False):
        if self.launch is None:
            return
        with self.lock:
            now = time.monotonic()
            if not final and now - self.last_telemetry < 10:
                return
            self.last_telemetry = now
            try:
                directory = self.launch.RUN / "rate-telemetry"
                directory.mkdir(exist_ok=True)
                atomic_json(
                    directory / f"{os.getpid()}-{self.started:.6f}.json",
                    {
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                        "arm": self.launch.progress.get("current_arm"),
                        "final": final,
                        "accounting": self.accounting(),
                    },
                )
            except OSError:
                # Reporting failure must not destroy paid in-flight work.
                pass

    def event(self, kind, attempt):
        super().event(kind, attempt)
        self.telemetry()

    def close(self):
        super().close()
        self.telemetry(final=True)

    def map(self, fn, rows):
        from concurrent.futures import wait, FIRST_COMPLETED

        iterator = iter(enumerate(rows))
        pending = {}
        results = {}

        def submit_next():
            item = next(iterator, None)
            if item is None:
                return False
            index, row = item
            pending[self.pool.submit(fn, row)] = index
            return True

        for _ in range(self.workers):
            if not submit_next():
                break
        while pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            errors = []
            for future in completed:
                index = pending.pop(future)
                try:
                    results[index] = future.result()
                except Exception as exc:
                    errors.append(exc)
            if errors:
                # Drain already-reserved work; never enqueue after observed failure.
                for future in pending:
                    try:
                        future.result()
                    except Exception:
                        pass
                raise errors[0]
            for _ in completed:
                submit_next()
        return [results[index] for index in range(len(results))]

    def route_client(self, name):
        if not hasattr(self.local, "routes"):
            self.local.routes = {}
        if name not in self.local.routes:
            client = self.factories[name](self.budget.maximum)
            client.budget = self.budget
            client.route_responses = 0
            client.route_attempts = 0
            client.route_errors = {}
            original = client.backend.evaluate

            def measured(program, state):
                client.route_attempts += 1
                try:
                    return original(program, state)
                except Exception as exc:
                    _, _, status = failure(exc)
                    label = str(status) if status is not None else type(exc).__name__
                    client.route_errors[label] = client.route_errors.get(label, 0) + 1
                    raise

            client.backend.evaluate = measured
            with self.lock:
                self.clients.append(client)
            self.local.routes[name] = client
        return self.local.routes[name]

    def evaluate(self, program, state):
        for attempt in range(3):
            name = self.scheduler.acquire(self.event, len(program.questions))
            cooldown, throttled = 0, False
            with self.lock:
                self.active += 1
                self.peak = max(self.peak, self.active)
            try:
                self.event("attempt", attempt)
                client = self.route_client(name)
                response = client.evaluate(program, state)
                with self.lock:
                    client.route_responses += 1
                    self.responses += 1
                    if self.launch:
                        self.launch.progress["jev_responses_received"] += 1
                return response
            except Exception as exc:
                transient, delay, status = failure(exc)
                if not transient:
                    raise
                cooldown = max(delay or 0, (5, 15, 30)[attempt] + random.uniform(0, 1))
                throttled = status == 429
                if attempt == 2:
                    raise
                self.event("retry", attempt)
            finally:
                with self.lock:
                    self.active -= 1
                self.scheduler.release(name, cooldown, throttled)

    def accounting(self):
        result = super().accounting()
        result["route_limits"] = self.configs
        result["rate_limit_responses_process_total"] = dict(self.scheduler.throttles)
        result["question_quota"] = dict(self.scheduler.quota)
        elapsed = result["elapsed_seconds"]
        result["reported_tokens_per_second"] = (
            (result["reported_input_tokens"] + result["reported_output_tokens"]) / elapsed
            if elapsed
            else None
        )
        for client in self.clients:
            route = result["routes"][client.identity]
            route["requests_attempted"] = route.get("requests_attempted", 0) + client.route_attempts
            errors = route.setdefault("errors", {})
            for name, count in client.route_errors.items():
                errors[name] = errors.get(name, 0) + count
        for route in result["routes"].values():
            route["requests_per_second"] = route["requests_attempted"] / elapsed if elapsed else None
            route["reported_tokens_per_second"] = (
                (route["input_tokens"] + route["output_tokens"]) / elapsed if elapsed else None
            )
        return result


def configurations(settings):
    if settings.get("AI_GATEWAY_ROUTE_6_ENABLED") == "true":
        raise BackendError("Pollinations is not approved: preflight returned HTTP 403.")
    configs = {"direct": {"concurrency": 8, "interval": 0.05, "group": "direct"}}
    for i in (1, 2, 3, 4, 5):
        if i > 2 and settings.get(f"AI_GATEWAY_ROUTE_{i}_ENABLED") != "true":
            continue
        name = f"gateway-{i}" if i < 3 else {3: "beatapi", 4: "opencode-zen", 5: "classifier"}[i]
        interval = float(settings.get(f"AI_GATEWAY_ROUTE_{i}_MIN_INTERVAL_SECONDS", ".1" if i < 3 else "60"))
        if not math.isfinite(interval) or interval <= 0 or (i == 3 and interval < 60):
            raise ValueError("Invalid route interval.")
        if i >= 3 and settings[f"AI_GATEWAY_ROUTE_{i}_MODEL"] != ("jev-1.13-free" if i in (3, 4) else PIN):
            raise BackendError("Route model differs from the verified mapping.")
        configs[name] = {
            "concurrency": 4 if i < 3 else (1 if i == 3 else 2),
            "interval": interval,
            "group": "vercel" if i < 3 else name,
            "group_interval": 0.05 if i < 3 else 0,
            "after_completion": i == 3,
            "model": "typesafe-ai/jev" if i < 3 else settings[f"AI_GATEWAY_ROUTE_{i}_MODEL"],
            "url": "https://ai-gateway.vercel.sh/typesafe/v1/systemone"
            if i < 3
            else settings[f"AI_GATEWAY_ROUTE_{i}_SYSTEMONE_URL"],
            "key_env": f"AI_GATEWAY_API_KEY_{i}",
        }
        if i == 5:
            configs[name].update(questions_per_second=50, questions_per_minute=3000, questions_daily=20000)
    return configs


_scheduler = None


def make_backend(limit, launch):
    global _scheduler
    settings = dotenv_values(launch.ROOT / ".env.local")
    configs = configurations(settings)
    if _scheduler is None:
        _scheduler = Scheduler(configs, quota_path=launch.RUN / "route-question-quota.json")
    elif _scheduler.configs != configs:
        raise BackendError("Route settings changed during the study process.")
    factories = {"direct": lambda n: launch.make_backend(True, n)}
    for name, config in configs.items():
        if name != "direct":
            key = settings[config["key_env"]]
            if not key:
                raise BackendError("Missing route credential.")
            factories[name] = lambda n, name=name, c=config, key=key: ManagedAlias(
                Native(name, c, key), max_calls=n, cache=None
            )
    return PacedBackend(limit, launch, configs, _scheduler, factories)


def calibration_budget_exhausted(exc, phase, backend):
    from s1compiler.errors import BudgetExceeded

    return (
        phase == "calibration"
        and isinstance(exc, BudgetExceeded)
        and backend is not None
        and backend.budget.used == backend.budget.maximum
    )


_calibration_scheduler = None


def make_calibration_backend(limit, launch):
    """Keep exact calibration attempt ceilings; avoid throttled gateway routes."""
    global _calibration_scheduler
    configs = {"direct": {"concurrency": 8, "interval": 0.05, "group": "direct"}}
    if _calibration_scheduler is None:
        _calibration_scheduler = Scheduler(configs)
    return PacedBackend(
        limit, launch, configs, _calibration_scheduler, {"direct": lambda n: launch.make_backend(True, n)}
    )
