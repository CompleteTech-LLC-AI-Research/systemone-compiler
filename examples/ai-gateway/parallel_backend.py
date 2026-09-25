"""Bounded ordered evaluation with isolated clients and one shared request budget."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from s1compiler.backends import Budget


def bounded_retry(call, event):
    event("attempt", 0)
    return call()


class SharedBudget(Budget):
    def __init__(self, maximum, on_reserve):
        super().__init__(maximum)
        self.lock = threading.RLock()
        self.on_reserve = on_reserve

    def reserve(self):
        with self.lock:
            super().reserve()
            self.on_reserve()


class ParallelBackend:
    synthetic = False
    identity = "typesafe-sdk/0.7.0"

    def __init__(self, maximum, factory, launch=None, workers=4, retry=bounded_retry):
        self.lock = threading.RLock()
        self.launch = launch
        self.factory = factory
        self.workers = workers
        self.retry = retry
        self.budget = SharedBudget(maximum, self.reserved)
        self.local = threading.local()
        self.clients = []
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.started = time.monotonic()
        self.responses = 0
        self.transient_retries = 0
        self.active = 0
        self.peak = 0

    def reserved(self):
        with self.lock:
            if self.launch:
                self.launch.progress["jev_requests_attempted"] += 1

    def client(self):
        if not hasattr(self.local, "client"):
            client = self.factory(self.budget.maximum)
            client.budget = self.budget
            with self.lock:
                self.clients.append(client)
            self.local.client = client
        return self.local.client

    def event(self, kind, attempt):
        with self.lock:
            if kind == "retry":
                self.transient_retries += 1
            if self.launch:
                self.launch.checkpoint(
                    execution_concurrency=self.workers,
                    phase_transport_retries=self.transient_retries,
                    phase_peak_in_flight=self.peak,
                    phase_responses=self.responses,
                    phase_elapsed_seconds=round(time.monotonic() - self.started, 1),
                )

    def evaluate(self, program, state):
        client = self.client()

        def call():
            with self.lock:
                self.active += 1
                self.peak = max(self.peak, self.active)
            try:
                response = client.evaluate(program, state)
                with self.lock:
                    self.responses += 1
                    if self.launch:
                        self.launch.progress["jev_responses_received"] += 1
                return response
            finally:
                with self.lock:
                    self.active -= 1

        return self.retry(call, self.event)

    def map(self, fn, rows):
        values = list(rows)
        result = []
        for start in range(0, len(values), self.workers):
            futures = [self.pool.submit(fn, row) for row in values[start : start + self.workers]]
            errors = []
            batch = []
            for future in futures:
                try:
                    batch.append(future.result())
                except Exception as exc:
                    errors.append(exc)
            # All in-flight reservations accounted; no next batch after failure.
            if errors:
                raise errors[0]
            result.extend(batch)
        return result

    def remember_validated(self, *args):
        pass  # All clients explicitly use cache=None.

    def accounting(self):
        with self.lock:
            accounts = [client.accounting() for client in self.clients]
            elapsed = time.monotonic() - self.started
            result = {
                "backend": self.identity,
                "synthetic": False,
                "requests_attempted": self.budget.used,
                "request_limit": self.budget.maximum,
                "dollar_cost": None,
                "models_seen": sorted({m for a in accounts for m in a["models_seen"]}),
                "execution_concurrency": self.workers,
                "peak_in_flight": self.peak,
                "responses_received": self.responses,
                "transport_retries": self.transient_retries,
                "failed_requests_without_response": self.budget.used - self.responses,
                "elapsed_seconds": elapsed,
                "responses_per_second": self.responses / elapsed if elapsed else None,
            }
            for key in (
                "cache_hits",
                "reported_input_tokens",
                "reported_output_tokens",
                "usage_unknown_calls",
            ):
                result[key] = sum(a[key] for a in accounts)
            result["usage_unknown_calls"] += self.budget.used - self.responses
            return result

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
        for client in self.clients:
            client.close()


class RoutedParallel(ParallelBackend):
    def accounting(self):
        result = super().accounting()
        result["routes"] = {}
        for client in self.clients:
            account = client.accounting()
            route = result["routes"].setdefault(
                client.identity,
                {
                    "responses": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "provider_attempts_reported": 0,
                    "actual_providers": [],
                },
            )
            route["responses"] += getattr(client, "route_responses", 0)
            route["input_tokens"] += account["reported_input_tokens"]
            route["output_tokens"] += account["reported_output_tokens"]
            route["provider_attempts_reported"] += account.get("provider_attempts_reported", 0)
            route["actual_providers"] = sorted(
                set(route["actual_providers"]) | set(account.get("actual_providers", []))
            )
        return result
