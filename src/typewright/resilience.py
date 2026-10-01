"""Bounded batch recovery without repeating successful evaluations.

The caller owns request reservations, identity validation and durable search
checkpoints. This module neither grants budget nor turns failures into scores.
"""
from concurrent.futures import FIRST_COMPLETED, wait
import time


def recover_batch(fn, rows, *, executor, workers, transient, reserve_retry,
                  recovery_rounds=3, delay_seconds=60.0, sleep=time.sleep,
                  heartbeat=lambda: None):
    """Drain on failure, pause the whole batch, and retry only failed rows.

    ``reserve_retry`` must charge the existing shared request allowance or raise.
    It runs immediately before each resubmission. The provided function may have
    its own bounded transport retries; those must also charge that allowance.
    Fatal errors always win over transient errors from the same in-flight batch.
    No new work is submitted after an observed failure until recovery succeeds.
    """
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    if type(recovery_rounds) is not int or not 0 <= recovery_rounds <= 3:
        raise ValueError("recovery_rounds must be between zero and three")
    if not 0 < delay_seconds <= 300:
        raise ValueError("delay_seconds must be between zero and 300")
    iterator = iter(enumerate(rows))
    pending, results, failures = {}, {}, {}
    exhausted = False
    rounds = 0
    recovering = False

    def submit(index, row):
        pending[executor.submit(fn, row)] = (index, row)

    while True:
        if recovering and not pending and not failures:
            recovering = False
        while not recovering and not failures and not exhausted and len(pending) < workers:
            item = next(iterator, None)
            if item is None:
                exhausted = True
                break
            submit(*item)
        if not pending:
            if not failures:
                return [results[i] for i in range(len(results))]
            fatal = [exc for _, exc in failures.values() if not transient(exc)]
            if fatal:
                raise fatal[0]
            if rounds >= recovery_rounds:
                raise next(iter(failures.values()))[1]
            # No provider requests in flight during this bounded cooldown.
            remaining = min(300.0, delay_seconds * 2 ** rounds)
            while remaining > 0:
                heartbeat()
                interval = min(15.0, remaining)
                sleep(interval)
                remaining -= interval
            rounds += 1
            recovering = True
            retry_rows, failures = failures, {}
            try:
                for index, (row, _) in sorted(retry_rows.items()):
                    reserve_retry()
                    submit(index, row)
            except BaseException:
                # A budget/storage failure cannot orphan already reserved work.
                for future in pending:
                    try:
                        future.result()
                    except BaseException:
                        pass
                raise
        completed, _ = wait(pending, timeout=15, return_when=FIRST_COMPLETED)
        heartbeat()
        for future in completed:
            index, row = pending.pop(future)
            try:
                results[index] = future.result()
            except Exception as exc:
                failures[index] = (row, exc)
