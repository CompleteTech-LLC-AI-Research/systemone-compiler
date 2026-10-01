from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from typewright.errors import BudgetExceeded
from typewright.resilience import recover_batch


def run(fn, rows, **kwargs):
    with ThreadPoolExecutor(max_workers=2) as executor:
        return recover_batch(fn, rows, executor=executor, workers=2,
                             transient=lambda e: isinstance(e, TimeoutError),
                             sleep=lambda _: None, **kwargs)


def test_recovery_retains_successful_results_and_order():
    calls, reservations = Counter(), []
    def operation(row):
        calls[row] += 1
        if row == 1 and calls[row] < 3:
            raise TimeoutError()
        return row * 10
    assert run(operation, range(6), reserve_retry=lambda: reservations.append(1)) == [0, 10, 20, 30, 40, 50]
    assert calls == Counter({0: 1, 1: 3, 2: 1, 3: 1, 4: 1, 5: 1})
    assert len(reservations) == 2


def test_fatal_in_flight_error_wins_over_timeout():
    barrier = threading.Barrier(2)
    calls = []
    def operation(row):
        calls.append(row)
        barrier.wait(timeout=5)
        if row == 0:
            raise TimeoutError()
        raise ValueError("identity mismatch")
    with pytest.raises(ValueError, match="identity"):
        run(operation, range(4), reserve_retry=lambda: pytest.fail("must not retry"))
    assert sorted(calls) == [0, 1]


def test_outage_is_bounded_and_charged():
    reservations = []
    def operation(_):
        raise TimeoutError()
    with pytest.raises(TimeoutError):
        run(operation, [0], reserve_retry=lambda: reservations.append(1))
    assert len(reservations) == 3


def test_no_retry_after_budget_exhaustion():
    calls = []
    def operation(row):
        calls.append(row)
        raise TimeoutError()
    def reserve():
        raise BudgetExceeded("shared allowance exhausted")
    with pytest.raises(BudgetExceeded):
        run(operation, [0], reserve_retry=reserve)
    assert calls == [0]


def test_cooldown_has_heartbeat_and_no_request_in_flight():
    calls, beats, sleeps = [], [], []
    def operation(row):
        calls.append(row)
        if len(calls) == 1:
            raise TimeoutError()
        return row
    with ThreadPoolExecutor(max_workers=1) as executor:
        assert recover_batch(operation, [1], executor=executor, workers=1,
            transient=lambda e: isinstance(e, TimeoutError), reserve_retry=lambda: None,
            heartbeat=lambda: beats.append(len(calls)),
            sleep=lambda seconds: sleeps.append((seconds, len(calls)))) == [1]
    assert sleeps == [(15, 1)] * 4
    assert len(beats) >= 4
