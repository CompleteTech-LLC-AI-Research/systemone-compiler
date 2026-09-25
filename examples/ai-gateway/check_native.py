"""Four bounded synthetic requests; never print raw errors or credentials."""

import json
from pathlib import Path
import httpx
from dotenv import dotenv_values

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-paid", action="store_true", required=True)
    parser.parse_args()
    ROOT = Path(__file__).resolve().parents[2]
    settings = dotenv_values(ROOT / ".env.local")
    questions = {
        "refunded": {"type": "noul", "instructions": "Was a refund issued?"},
        "team": {
            "type": "choice",
            "instructions": "Which team handles this?",
            "criteria": {"billing": "Charges and refunds", "technical": "Software bugs"},
        },
        "sentiment": {
            "type": "score",
            "instructions": "How positive is the customer?",
            "criteria": ["Unhappy", "Neutral", "Happy"],
        },
    }
    reports = []
    with httpx.Client(timeout=30) as client:
        for route in (1, 2):
            for model in ("jev-1.13.0", "typesafe-ai/jev"):
                try:
                    response = client.post(
                        "https://ai-gateway.vercel.sh/typesafe/v1/systemone",
                        headers={"Authorization": "Bearer " + settings[f"AI_GATEWAY_API_KEY_{route}"]},
                        json={
                            "model": model,
                            "state": "The agent refunded my duplicate charge. I am happy.",
                            "questions": questions,
                        },
                    )
                    record = {"route": route, "requested_model": model, "http_status": response.status_code}
                    if response.is_success:
                        body = response.json()
                        record.update(
                            returned_model=body.get("model"),
                            answer_fields={k: sorted(v) for k, v in body.get("answers", {}).items()},
                            usage=body.get("usage"),
                            routing=body.get("provider_metadata", {}).get("gateway", {}).get("routing"),
                        )
                except Exception as exc:
                    record = {"route": route, "requested_model": model, "error_type": type(exc).__name__}
                reports.append(record)
    destination = ROOT / "runs/gateway-native-preflight.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "synthetic_probe_inputs": True,
                "study_examples_used": False,
                "requests_attempted": 4,
                "results": reports,
            },
            indent=2,
        )
    )
    print(json.dumps(reports))


if __name__ == "__main__":
    main()
