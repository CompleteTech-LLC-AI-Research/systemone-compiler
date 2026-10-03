from __future__ import annotations
import math
import re
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_serializer, model_validator

from .errors import CandidateError, DataError
from .io import atomic_json, fingerprint, load_document

Kind = Literal["choice", "noul", "score"]
Entry = str | dict[str, JsonValue] | list[JsonValue] | None
ID = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)


def validate_shape(kind: str, criteria: Any) -> None:
    if kind == "choice":
        if not isinstance(criteria, dict) or len(criteria) < 2:
            raise ValueError("Choice requires at least two named options.")
        if any(not isinstance(k, str) or not k.strip() for k in criteria):
            raise ValueError("Choice option names must be nonempty strings.")
    elif kind == "score":
        if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
            raise ValueError("Score requires an ordered list of 2–10 level descriptions.")
    elif criteria is not None:
        if not isinstance(criteria, dict) or not set(criteria).issubset({"true", "false"}):
            raise ValueError("Noul criteria must use quoted 'true'/'false' keys.")


class StateField(StrictModel):
    type: Literal["string", "integer", "number", "boolean", "array", "object"] = "string"
    description: str = ""
    required: bool = True


class Decision(StrictModel):
    type: Kind
    goal: str = Field(min_length=3, max_length=12000)
    criteria: dict[str, Entry] | list[Entry] | None = None
    weight: float = Field(default=1.0, gt=0)
    score_tolerance: float = Field(default=0.5, ge=0)

    @model_validator(mode="after")
    def validate_decision(self):
        validate_shape(self.type, self.criteria)
        return self


class UseCase(StrictModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    description: str = ""
    model: str = "jev-1.13.0"
    state: dict[str, StateField]
    decisions: dict[str, Decision]

    @model_validator(mode="after")
    def check_ids(self):
        if not self.state or not self.decisions:
            raise ValueError("At least one state field and decision are required.")
        for key in list(self.state) + list(self.decisions):
            if not ID.fullmatch(key):
                raise ValueError("State and decision identifiers must be lowercase snake_case.")
        return self

    @classmethod
    def load(cls, path: str | Path) -> UseCase:
        return cls.model_validate(load_document(path))


class Question(StrictModel):
    type: Kind
    instructions: Entry
    criteria: dict[str, Entry] | list[Entry] | None = None

    @model_validator(mode="after")
    def check_question(self):
        if not self.instructions:
            raise ValueError("Every question must have a self-contained instruction.")
        validate_shape(self.type, self.criteria)
        return self

    def wire(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)


class Binding(StrictModel):
    kind: Literal["question", "weighted_mean"] = "question"
    question: str | None = None
    weights: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_binding(self):
        if self.kind == "question":
            if not self.question or self.weights:
                raise ValueError("A direct binding requires exactly one question and no weights.")
        elif self.question is not None or not self.weights or any(v <= 0 for v in self.weights.values()):
            raise ValueError("A weighted_mean requires positive weights and no direct question.")
        return self


class Policy(StrictModel):
    noul_threshold: float = Field(default=0.5, gt=0, lt=1)
    min_gate: float = Field(default=0.8, ge=0, le=1)
    force_review: bool = False
    fitting_version: Literal["native-policy/v1"] | None = None
    score_cuts: list[Annotated[float, Field(strict=True, ge=0)]] | None = None
    choice_weights: dict[str, Annotated[float, Field(strict=True, gt=0)]] | None = None

    @model_validator(mode="after")
    def check_fitted_policy(self):
        knobs = self.score_cuts is not None or self.choice_weights is not None
        if knobs != (self.fitting_version is not None):
            raise ValueError("Fitted knobs require an explicit native-policy/v1 version.")
        if self.score_cuts is not None:
            if not self.score_cuts or any(a >= b for a, b in zip(self.score_cuts, self.score_cuts[1:])):
                raise ValueError("Score cuts must be nonempty and strictly ordered.")
        if self.choice_weights is not None:
            if not self.choice_weights or any(v <= 0 for v in self.choice_weights.values()):
                raise ValueError("Choice weights must be positive.")
        return self

    @model_serializer(mode="wrap")
    def serialize_compatible(self, handler):
        # Omit absent extensions even in nested dumps: existing artifact bytes and
        # content hashes retain their original meaning.
        data = handler(self)
        for key in ("fitting_version", "score_cuts", "choice_weights"):
            if data.get(key) is None:
                data.pop(key, None)
        return data


class Program(StrictModel):
    format: Literal["systemone-program/v1"] = "systemone-program/v1"
    name: str
    model: str
    state: dict[str, StateField]
    decisions: dict[str, Decision]
    questions: dict[str, Question]
    bindings: dict[str, Binding]
    policies: dict[str, Policy]
    provenance: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_program(self):
        if not self.state or not self.questions or len(self.questions) > 64:
            raise ValueError("A program needs state and 1–64 questions.")
        if set(self.decisions) != set(self.bindings) or set(self.decisions) != set(self.policies):
            raise ValueError("Each decision requires exactly one binding and policy.")
        for qid in self.questions:
            if not ID.fullmatch(qid):
                raise ValueError("Question identifiers must be lowercase snake_case.")
        used = set()
        for name, binding in self.bindings.items():
            decision = self.decisions[name]
            policy = self.policies[name]
            if policy.score_cuts is not None:
                if decision.type != "score" or len(policy.score_cuts) != len(decision.criteria) - 1:
                    raise ValueError("Score cuts must match the declared Score scale.")
                if any(not 0 <= cut <= len(decision.criteria) - 1 for cut in policy.score_cuts):
                    raise ValueError("Score cuts must lie within the declared scale.")
            if policy.choice_weights is not None:
                if decision.type != "choice" or set(policy.choice_weights) != set(decision.criteria):
                    raise ValueError("Choice weights must match the fixed labels.")
            if binding.kind == "question":
                if binding.question not in self.questions:
                    raise ValueError(f"Unknown question in binding {name}.")
                used.add(binding.question)
                q = self.questions[binding.question]
                if q.type != decision.type:
                    raise ValueError(f"Type mismatch in {name}.")
                if q.type == "choice" and set(q.criteria) != set(decision.criteria):
                    raise ValueError(f"Choice labels cannot change: {name}.")
                if q.type == "score" and len(q.criteria) != len(decision.criteria):
                    raise ValueError(f"Score output range cannot change: {name}.")
            else:
                if decision.type != "score":
                    raise ValueError("Weighted composition is restricted to Score outputs.")
                for qid in binding.weights:
                    if qid not in self.questions or self.questions[qid].type not in {"score", "noul"}:
                        raise ValueError("Composite inputs must be existing Score or Noul questions.")
                    used.add(qid)
        if used != set(self.questions):
            raise ValueError("Unused questions are prohibited; bind or remove them.")
        return self

    @property
    def content_hash(self) -> str:
        return fingerprint(self.model_dump(mode="json", exclude={"provenance"}))

    def save(self, path: str | Path) -> None:
        atomic_json(path, {"program": self.model_dump(mode="json"), "sha256": fingerprint(self.model_dump(mode="json"))})

    @classmethod
    def load(cls, path: str | Path) -> Program:
        data = load_document(path)
        if set(data) != {"program", "sha256"}:
            raise DataError("Expected an artifact envelope with program and sha256.")
        program = cls.model_validate(data["program"])
        if fingerprint(program.model_dump(mode="json")) != data["sha256"]:
            raise DataError("Artifact content checksum mismatch; recompile or verify the file.")
        return program


def assert_contract(program: Program, source: UseCase) -> None:
    if program.name != source.name or program.model != source.model:
        raise CandidateError("The architect may not change the name or target model.")
    if program.decisions != source.decisions:
        raise CandidateError("The architect may not change declared output semantics or weights.")
    if not set(program.state).issubset(source.state):
        raise CandidateError("The architect attempted to introduce an undeclared state field.")
    if any(field != source.state[key] for key, field in program.state.items()):
        raise CandidateError("The architect may select state fields, not redefine their schema.")


def project_state(fields: dict[str, StateField], state: dict[str, Any]) -> dict[str, Any]:
    """Select declared top-level fields. Extra fields are never transmitted."""
    if not isinstance(state, dict):
        raise DataError("State must be a JSON object.")
    selected = {}
    for name, spec in fields.items():
        if name not in state:
            if spec.required:
                raise DataError(f"Required state field is missing: {name}")
            continue
        value = state[name]
        expected = {
            "string": lambda v: isinstance(v, str),
            "integer": lambda v: type(v) is int,
            "number": lambda v: type(v) in (int, float) and math.isfinite(v),
            "boolean": lambda v: type(v) is bool,
            "array": lambda v: isinstance(v, list),
            "object": lambda v: isinstance(v, dict),
        }[spec.type]
        if not expected(value):
            raise DataError(f"State field {name} must have type {spec.type}.")
        selected[name] = value
    fingerprint(selected)  # Verify JSON serializability, including finite nested numbers.
    return selected
