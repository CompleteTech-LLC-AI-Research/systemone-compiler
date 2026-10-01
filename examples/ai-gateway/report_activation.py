"""One-shot observer: report the first ten minutes after activation.

It never starts inference or touches benchmark data, but it writes new-route-throughput.* into the run
directory and notifies through notify.ps1. Rates sum every rate-telemetry file present, so leftover
telemetry from earlier phases inflates them; they are approximate.
"""

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

RUN = None
NOTIFY = Path(__file__).resolve().with_name("notify.ps1")


def notify(message):
    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(NOTIFY),
            "-Title",
            "Jev throughput report",
            "-Message",
            message,
        ],
        capture_output=True,
        text=True,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    return result.returncode == 0


def main():
    started = None
    initial_completed = None
    while True:
        try:
            status = json.loads((RUN / "status.json").read_text())
            handoff = json.loads((RUN / "handoff-v12-status.json").read_text())
        except (OSError, ValueError):
            time.sleep(15)
            continue
        if handoff["status"] == "blocked":
            notify(
                "Route activation is blocked. Check handoff-v12-status.json; benchmark data has been preserved."
            )
            return
        active = status.get("selection_directory", "").endswith("selected-resume-12")
        if active and started is None:
            started = time.monotonic()
            initial_completed = status["completed_arms"]
            notify(
                "Eight-worker, six-route continuation activated. First measured traffic report follows in ten minutes."
            )
        if started is not None and time.monotonic() - started >= 600:
            accounts = []
            for path in (RUN / "rate-telemetry").glob("*.json"):
                try:
                    accounts.append(json.loads(path.read_text())["accounting"])
                except (OSError, ValueError):
                    continue
            if not accounts:
                notify(
                    "New continuation has no traffic telemetry yet. Check benchmark status; no throughput result can be reported."
                )
                return
            seconds = time.monotonic() - started
            sums = {
                key: sum(a.get(key, 0) for a in accounts)
                for key in (
                    "requests_attempted",
                    "responses_received",
                    "reported_input_tokens",
                    "reported_output_tokens",
                    "transport_retries",
                    "usage_unknown_calls",
                )
            }
            routes = {}
            for account in accounts:
                for name, route in account["routes"].items():
                    totals = routes.setdefault(
                        name, {"requests_attempted": 0, "responses": 0, "input_tokens": 0, "output_tokens": 0}
                    )
                    for key in totals:
                        totals[key] += route.get(key, 0)
            rps = sums["requests_attempted"] / seconds
            tps = (sums["reported_input_tokens"] + sums["reported_output_tokens"]) / seconds
            result = {
                "at": datetime.now(timezone.utc).isoformat(),
                "window_seconds": seconds,
                "accounting": sums,
                "routes": routes,
                "rps": rps,
                "reported_tps": tps,
                "completed_arms": status["completed_arms"],
                "new_completed_arms": status["completed_arms"] - initial_completed,
                "status": status["status"],
                "current_arm": status.get("current_arm"),
                "note": "Approximate first ten minutes; telemetry interval ten seconds. Includes pauses. No new held-out evaluation.",
            }
            (RUN / "new-route-throughput.json").write_text(json.dumps(result, indent=2))
            lines = [
                "# First six-route traffic measurement",
                "",
                f"Observed at {result['at']}; window {seconds:.1f} seconds.",
                "",
                f"- Requests/sec: **{rps:.2f}**",
                f"- Reported input tokens/sec: **{sums['reported_input_tokens'] / seconds:.0f}**",
                f"- Reported output tokens/sec: **{sums['reported_output_tokens'] / seconds:.0f}**",
                f"- Combined reported tokens/sec: **{tps:.0f}**",
                f"- Transport retries: {sums['transport_retries']}; unknown-usage calls: {sums['usage_unknown_calls']}.",
                f"- Completed arms: {status['completed_arms']}; newly completed: {result['new_completed_arms']}.",
                "",
                "| Route | Attempts | Responses | Reported tokens |",
                "|---|---:|---:|---:|",
            ]
            for name, route in routes.items():
                lines.append(
                    f"| {name} | {route['requests_attempted']} | {route['responses']} | {route['input_tokens'] + route['output_tokens']} |"
                )
            lines += [
                "",
                "Earlier four-worker reference: 19.2 RPS over 45 seconds; 16.0 RPS over an arm including pauses. Earlier completed calibration: 81,208 reported TPS at 18.4 RPS. These are different windows and workloads, not a controlled speedup comparison.",
                "",
                result["note"],
                "This is a traffic measurement, not evidence of improved prediction quality.",
            ]
            (RUN / "NEW_ROUTE_THROUGHPUT.md").write_text("\n".join(lines) + "\n")
            submitted = notify(
                f"First 10-minute report: {rps:.1f} RPS, {tps:,.0f} reported TPS. See runs/live-study-20260920/NEW_ROUTE_THROUGHPUT.md."
            )
            (RUN / "route-report-observer-status.json").write_text(
                json.dumps({"status": "complete", "notification_submitted": submitted})
            )
            return
        time.sleep(15)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    RUN = args.run_dir.resolve()
    main()
