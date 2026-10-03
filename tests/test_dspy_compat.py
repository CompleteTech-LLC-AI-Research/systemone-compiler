"""DSPy version-range regression tests (issue #90). Offline: the only socket is a loopback test double.

The loopback server below is a TEST DOUBLE standing in for a provider. Nothing here is a Jev result,
a provider call, or evidence about quality.
"""
import json
import inspect
import threading
import tomllib
import warnings
import socket
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib import metadata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def loopback_only_network(monkeypatch):
    """Fail closed if an integration regression tries any external transport."""
    connect = socket.socket.connect
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")

    def guarded(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            assert address[0] in ("127.0.0.1", "::1"), "External network forbidden in offline teacher tests"
        return connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", guarded)


def _optimize():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return data["project"]["optional-dependencies"]


def test_extras_declare_verified_teacher_range_and_native_compile_profile():
    extras = _optimize()
    assert "dspy[litellm]>=3.3.1,<3.5" in extras["optimize"]
    assert "dspy[typesafe]==3.4.0" in extras["compile"]
    assert "typesafe-sdk==0.7.0" in extras["compile"]
    assert "dspy[litellm,typesafe]==3.4.0" in extras["all"]
    assert "typesafe-sdk==0.7.0" in extras["all"]
    for name in ("optimize", "all"):
        assert "gepa==0.1.4" in extras[name], name


@pytest.mark.optional
def test_installed_dspy_is_inside_the_declared_range():
    pytest.importorskip("dspy")
    major_minor = tuple(int(x) for x in metadata.version("dspy").split(".")[:2])
    assert (3, 3) <= major_minor < (3, 5)
    assert metadata.version("gepa") == "0.1.4"


class _Provider(BaseHTTPRequestHandler):
    seen = []

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("content-length", 0)))
        type(self).seen.append((self.path, self.headers.get("authorization"), json.loads(body)))
        out = json.dumps({"id": "t", "object": "chat.completion", "created": 1, "model": "deepseek-flash",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content":
                '[[ ## revised_components_json ## ]]\n{"flag/instructions":"Is it supported?"}\n[[ ## completed ## ]]'}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *args):
        pass


@pytest.mark.optional
def test_teacher_uses_pinned_engine_overrides_and_unchanged_metering_against_loopback_double(monkeypatch):
    pytest.importorskip("dspy")
    pytest.importorskip("litellm")
    _Provider.seen = []
    server = HTTPServer(("127.0.0.1", 0), _Provider)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setenv("S1_TEACHER_API_BASE", f"http://127.0.0.1:{server.server_port}")
        monkeypatch.setenv("S1_TEACHER_API_KEY", "fake-test-key-not-real")
        from typewright.research_teacher import AuditedDSPyTeacher
        teacher = AuditedDSPyTeacher("deepseek/deepseek-flash", expected_response_model="deepseek-flash",
            max_provider_calls=2, allow_paid=True, share_feedback=True, max_calls=2)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            result = teacher.propose_components({"flag/instructions": '"Check this"'}, {}, ["flag/instructions"])
    finally:
        server.shutdown()
        server.server_close()
    assert result == {"flag/instructions": '"Is it supported?"'}
    assert len(_Provider.seen) == 1
    path, authorization, _ = _Provider.seen[0]
    assert path.endswith("/chat/completions") and authorization == "Bearer fake-test-key-not-real"
    account = teacher.accounting()
    assert account["signature_calls"] == account["provider_requests_attempted"] == 1
    assert account["reported_input_tokens"] == 10 and account["reported_output_tokens"] == 5
    assert account["observed_response_models"] == ["deepseek-flash"]
    # Cost metadata comes from LiteLLM's response; DSPy 3.4 native engines would report it unknown.
    assert account["cost_unknown_calls"] == 0 and account["sdk_estimated_cost"] is not None
    assert account["dollar_cost"] is None
    assert teacher.lm.history == [] and teacher.revise.history == []
    assert teacher.lm.cache is False
    assert not any("BaseLM.forward()" in str(item.message) for item in caught)
    if "engine" in inspect.signature(teacher.dspy.LM.__init__).parameters:
        assert "forward" not in vars(teacher.lm)
        assert not hasattr(teacher.lm._engine_spec, "complete_legacy")
