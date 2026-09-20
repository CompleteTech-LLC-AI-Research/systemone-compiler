from __future__ import annotations

from .errors import CandidateError, ConfigurationError
from .io import canonical, json_loads
from .metrics import example_quality, feedback
from .models import Program
from .runtime import Runtime


def components_from_program(program: Program) -> dict[str, str]:
    components = {}
    for qid, q in program.questions.items():
        components[f"{qid}/instructions"] = canonical(q.instructions)
        if isinstance(q.criteria, dict):
            for label, value in q.criteria.items():
                # Index prevents delimiter ambiguity for arbitrary external Choice labels.
                index = list(q.criteria).index(label)
                components[f"{qid}/criterion/{index}"] = canonical(value)
        elif isinstance(q.criteria, list):
            for index, value in enumerate(q.criteria):
                components[f"{qid}/level/{index}"] = canonical(value)
    return components


def program_from_components(base: Program, candidate: dict[str, str]) -> Program:
    expected = components_from_program(base)
    if set(candidate) != set(expected):
        raise CandidateError("A prompt candidate cannot add or delete components.")
    data = base.model_dump(mode="json")
    try:
        for key, text in candidate.items():
            if not isinstance(text, str) or len(text) > 24000:
                raise ValueError("Invalid component text/size")
            value = json_loads(text)
            if value is not None and not isinstance(value, (str, dict, list)):
                raise ValueError("Invalid entry type")
            qid, section, *rest = key.split("/")
            if section == "instructions":
                data["questions"][qid]["instructions"] = value
            elif section == "criterion":
                label = list(base.questions[qid].criteria)[int(rest[0])]
                data["questions"][qid]["criteria"][label] = value
            else:
                data["questions"][qid]["criteria"][int(rest[0])] = value
        return Program.model_validate(data)
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        raise CandidateError("Prompt candidate violates the typed program contract.") from exc


class JevGEPAAdapter:
    """Standalone GEPA's adapter protocol with a DSPy component proposer.

    The optimized task is a native Jev program, not an emulated chat LM.
    Validation examples are scored, but only train traces reach the proposer
    through GEPA's reflective minibatches.
    """
    def __init__(self, program: Program, backend, teacher, *, batch_factory=None):
        self.program, self.backend, self.teacher = program, backend, teacher
        self.batch_factory = batch_factory
        self.rejected = 0

    def evaluate(self, batch, candidate, capture_traces=False):
        if self.batch_factory is None:
            try:
                from gepa.core.adapter import EvaluationBatch
            except ImportError as exc:
                raise ConfigurationError("GEPA is missing; install the optimize extra.") from exc
            factory = EvaluationBatch
        else:
            factory = self.batch_factory
        outputs, scores, traces = [], [], []
        try:
            program = program_from_components(self.program, candidate)
        except CandidateError:
            # Syntax/schema failure is candidate quality, not an infrastructure outage.
            return factory(outputs=[{"error": "invalid_candidate"} for _ in batch],
                scores=[0.0] * len(batch),
                trajectories=[{"error": "Candidate schema invalid; preserve component names and JSON entry types."}
                              for _ in batch] if capture_traces else None)
        runtime = Runtime(program, self.backend)
        for example in batch:
            result = runtime.run(example.state)
            # Authentication, rate-limit, model drift, and malformed server responses
            # are systemic failures: propagate rather than score them as wrong labels.
            outputs.append(result)
            scores.append(example_quality(program, example, result))
            if capture_traces:
                traces.append(feedback(program, example, result, include_state=True))
        return factory(outputs=outputs, scores=scores, trajectories=traces if capture_traces else None)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        if eval_batch.trajectories is None:
            raise CandidateError("Reflection requested without captured training traces.")
        return {key: [{"Inputs": trace.get("input", {}),
                       "Generated Outputs": trace.get("predictions", {}),
                       "Feedback": {k: v for k, v in trace.items() if k not in {"input", "predictions"}}}
                      for trace in eval_batch.trajectories] for key in components_to_update}

    def propose_new_texts(self, candidate, reflective_dataset, components_to_update):
        try:
            proposed = self.teacher.propose_components(candidate, reflective_dataset, components_to_update)
            if set(proposed) != set(components_to_update):
                raise CandidateError("Proposer changed unrequested component keys.")
            merged = {**candidate, **proposed}
            program_from_components(self.program, merged)
            return proposed
        except CandidateError:
            self.rejected += 1
            return {key: candidate[key] for key in components_to_update}


def optimize_gepa(program: Program, train, validation, backend, teacher, *,
                  max_metric_calls: int = 128, seed: int = 7):
    try:
        import gepa
    except ImportError as exc:
        raise ConfigurationError("Install the optimize extra: pip install -e '.[optimize]'.") from exc
    if max_metric_calls < len(validation) + 2:
        raise ConfigurationError("GEPA metric budget must exceed the initial validation-set evaluation.")
    adapter = JevGEPAAdapter(program, backend, teacher)
    # No checkpoint loading/pickle, no third-party experiment tracking, no raw
    # trace files. DSPy supplies propose_new_texts, so reflection_lm is not needed.
    result = gepa.optimize(
        seed_candidate=components_from_program(program), trainset=train, valset=validation,
        adapter=adapter, max_metric_calls=max_metric_calls, reflection_minibatch_size=min(3, len(train)),
        seed=seed, use_merge=False, skip_perfect_score=False, display_progress_bar=False,
        raise_on_exception=True,
    )
    return program_from_components(program, result.best_candidate), {
        "engine": "gepa", "metric_call_budget": max_metric_calls,
        "invalid_mutations_rejected": adapter.rejected,
    }
