"""No-key reference interpreter for the hierarchy ADR fixtures.

This documents observable routing semantics. It is not the production graph
runtime; later issues must validate/lower a versioned frozen artifact.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from typewright.backends import ManagedBackend, MockBackend
from typewright.io import load_document
from typewright.models import Decision, Program, StateField, UseCase, project_state
from typewright.runtime import Runtime


class ContractError(ValueError):
    pass


def require(condition, category):
    if not condition:
        raise ContractError(category)


def interface(graph):
    return ({key: StateField.model_validate(value) for key, value in graph['inputs'].items()},
            {key: Decision.model_validate(value) for key, value in graph['outputs'].items()})


def stage_outputs(stage, definitions):
    if stage['kind'] == 'leaf':
        return Program.model_validate(stage['program']).decisions
    require(stage['kind'] == 'subgraph' and stage['definition'] in definitions, 'unknown_definition')
    return interface(definitions[stage['definition']])[1]


def field_type(decision, field):
    if field == 'value':
        return {'choice': 'string', 'noul': 'boolean', 'score': 'number'}[decision.type]
    if field == 'p_true' and decision.type == 'noul':
        return 'number'
    if field == 'probabilities' and decision.type in {'choice', 'score'}:
        return 'object'
    if field in {'gate_score', 'vendor_confidence'}:
        return 'number'
    if field == 'review_required':
        return 'boolean'
    raise ContractError('invalid_reference_field')


def ref_type(ref, inputs, stages, definitions):
    require(isinstance(ref, dict), 'invalid_reference')
    names = set(ref) - {'default'}
    if names == {'root'}:
        require(ref['root'] in inputs, 'undeclared_root_field')
        return inputs[ref['root']].type, None
    require(names == {'stage', 'decision', 'field'}, 'invalid_reference')
    require(ref['stage'] in stages, 'unknown_stage')
    outputs = stage_outputs(stages[ref['stage']], definitions)
    require(ref['decision'] in outputs, 'unknown_decision')
    decision = outputs[ref['decision']]
    return field_type(decision, ref['field']), decision


def references(stage):
    result = list(stage['inputs'].values())
    if 'when' in stage:
        require(set(stage['when']) == {'all'} and stage['when']['all'], 'invalid_condition')
        result += [predicate['ref'] for predicate in stage['when']['all']]
    return result


def ordered_stages(graph, definitions):
    require(set(graph) == {'inputs', 'outputs', 'stages', 'final'}, 'unknown_graph_field')
    stages = {stage['id']: stage for stage in graph['stages']}
    require(len(stages) == len(graph['stages']), 'duplicate_stage')
    inputs, outputs = interface(graph)
    dependencies = {}
    for name, stage in stages.items():
        require(set(stage) <= {'id', 'kind', 'program', 'definition', 'inputs', 'when',
                               'after', 'on_review'}, 'unknown_stage_field')
        require(stage.get('on_review', 'defer') in {'defer', 'continue_marked'}, 'invalid_review_policy')
        require(('program' in stage) != ('definition' in stage), 'invalid_stage_kind')
        require(stage['kind'] in {'leaf', 'subgraph'}, 'invalid_stage_kind')
        if stage['kind'] == 'subgraph':
            require(stage['definition'] in definitions, 'unknown_definition')
        stage_interface = (Program.model_validate(stage['program']).state if stage['kind'] == 'leaf'
                           else interface(definitions[stage['definition']])[0])
        require(set(stage['inputs']) <= set(stage_interface) and
                all(not field.required or key in stage['inputs'] for key, field in stage_interface.items()),
                'stage_input_contract')
        for field, ref in stage['inputs'].items():
            actual, _ = ref_type(ref, inputs, stages, definitions)
            require(actual == stage_interface[field].type or
                    (actual == 'integer' and stage_interface[field].type == 'number'),
                    'input_type_mismatch')
            if 'default' in ref:
                project_state({field: stage_interface[field]}, {field: ref['default']})
        deps = set(stage.get('after', []))
        for ref in references(stage):
            if 'stage' in ref:
                deps.add(ref['stage'])
        require(deps <= stages.keys(), 'unknown_stage')
        dependencies[name] = deps
        for predicate in stage.get('when', {}).get('all', []):
            actual, decision = ref_type(predicate['ref'], inputs, stages, definitions)
            op = predicate.get('op')
            value = predicate.get('value')
            require(op in {'eq', 'in', 'lt', 'lte', 'gt', 'gte'}, 'invalid_condition')
            if op in {'lt', 'lte', 'gt', 'gte'}:
                require(actual in {'number', 'integer'} and type(value) in (int, float)
                        and math.isfinite(value),
                        'condition_type_mismatch')
            elif op == 'in':
                require(isinstance(value, list) and bool(value) and all(
                    (actual == 'boolean' and type(item) is bool) or
                    (actual == 'string' and type(item) is str) or
                    (actual in {'number', 'integer'} and type(item) in (int, float) and math.isfinite(item))
                    for item in value), 'condition_type_mismatch')
            else:
                require((actual == 'boolean' and type(value) is bool) or
                        (actual == 'string' and type(value) is str) or
                        (actual in {'number', 'integer'} and type(value) in (int, float) and
                         math.isfinite(value)), 'condition_type_mismatch')
            if decision and decision.type == 'choice' and op in {'eq', 'in'}:
                values = value if op == 'in' else [value]
                require(all(v in decision.criteria for v in values), 'unknown_choice_label')
    done = []
    pending = set(stages)
    while pending:
        ready = sorted(name for name in pending if dependencies[name] <= set(done))
        require(bool(ready), 'cycle')
        name = ready[0]
        done.append(name)
        pending.remove(name)
    require(set(graph['final']) == set(outputs), 'missing_final_output')
    for key, mapping in graph['final'].items():
        require(mapping.get('on_missing') == 'review_required' and mapping.get('candidates'),
                'invalid_final_mapping')
        for candidate in mapping['candidates']:
            require(candidate['stage'] in stages, 'unknown_stage')
            child = stage_outputs(stages[candidate['stage']], definitions)
            require(candidate['decision'] in child, 'unknown_decision')
            source = child[candidate['decision']]
            target = outputs[key]
            require(source.type == target.type, 'final_type_mismatch')
            if target.type == 'choice':
                labels = set(source.criteria)
                if candidate.get('distribution_scope') == 'branch_conditional':
                    mapping_labels = candidate.get('label_map', {})
                    require(set(mapping_labels) == labels and
                            set(mapping_labels.values()) <= set(target.criteria) and
                            len(set(mapping_labels.values())) == len(labels), 'invalid_label_map')
                else:
                    require(labels == set(target.criteria), 'final_choice_labels')
            if target.type == 'score':
                require(len(source.criteria) == len(target.criteria), 'final_score_scale')
    return [stages[name] for name in done]


def validate(doc):
    require(set(doc) == {'format', 'source', 'limits', 'definitions', 'graph'}, 'unknown_source_field')
    require(doc['format'] == 'systemone-hierarchy-source/v1', 'unsupported_format')
    source = UseCase.model_validate(doc['source'])
    require(doc['graph']['inputs'] == source.model_dump(mode='json')['state'] and
            doc['graph']['outputs'] == source.model_dump(mode='json')['decisions'], 'root_contract_drift')
    limits = doc['limits']
    require(set(limits) == {'max_expanded_nodes', 'max_depth', 'max_native_calls_per_example'} and
            0 < limits['max_expanded_nodes'] <= 64 and 0 < limits['max_depth'] <= 8 and
            0 < limits['max_native_calls_per_example'] <= 64, 'invalid_limits')
    definitions = doc['definitions']
    used_definitions = set()
    def visit(graph, stack, depth):
        require(depth <= limits['max_depth'], 'depth_limit')
        stages = ordered_stages(graph, definitions)
        nodes = calls = 0
        for stage in stages:
            nodes += 1
            if stage['kind'] == 'leaf':
                program = Program.model_validate(stage['program'])
                require(program.model == source.model, 'model_drift')
                calls += 1
            else:
                definition = stage['definition']
                require(definition not in stack, 'recursive_definition')
                used_definitions.add(definition)
                subnodes, subcalls = visit(definitions[definition], stack + (definition,), depth + 1)
                nodes += subnodes
                calls += subcalls
        return nodes, calls
    nodes, calls = visit(doc['graph'], (), 1)
    require(used_definitions == set(definitions), 'unused_definition')
    require(nodes <= limits['max_expanded_nodes'] and calls <= limits['max_native_calls_per_example'],
            'expansion_limit')
    return source


def read_ref(ref, inputs, values):
    try:
        if 'root' in ref:
            return inputs[ref['root']]
        return values[ref['stage']][ref['decision']][ref['field']]
    except KeyError:
        if 'default' in ref:
            return ref['default']
        raise


def allows(condition, inputs, values):
    for predicate in condition.get('all', []):
        try:
            actual = read_ref(predicate['ref'], inputs, values)
        except (KeyError, TypeError):
            return False
        op, target = predicate['op'], predicate['value']
        accepted = {'eq': lambda: actual == target, 'in': lambda: actual in target,
                    'lt': lambda: actual < target, 'lte': lambda: actual <= target,
                    'gt': lambda: actual > target, 'gte': lambda: actual >= target}[op]()
        if not accepted:
            return False
    return True


def execute_graph(graph, definitions, inputs, backend, prefix=''):
    values, statuses, reviews, executed, skipped = {}, {}, {}, [], []
    for stage in ordered_stages(graph, definitions):
        name = stage['id']
        qualified = prefix + name
        if not allows(stage.get('when', {}), inputs, values):
            statuses[name] = 'skipped'
            skipped.append(qualified)
            continue
        stage_interface = (Program.model_validate(stage['program']).state if stage['kind'] == 'leaf'
                           else interface(definitions[stage['definition']])[0])
        stage_input_deps = {ref['stage'] for field, ref in stage['inputs'].items()
                            if 'stage' in ref and 'default' not in ref and
                            stage_interface[field].required}
        condition_deps = {predicate['ref']['stage']
                          for predicate in stage.get('when', {}).get('all', [])
                          if 'stage' in predicate['ref']}
        required_deps = stage_input_deps | set(stage.get('after', []))
        if any(statuses[dep] != 'completed' or
               (reviews.get(dep) and
                next(s for s in graph['stages'] if s['id'] == dep).get('on_review', 'defer') == 'defer')
               for dep in required_deps | condition_deps):
            statuses[name] = 'review_blocked'
            skipped.append(qualified)
            continue
        try:
            mapped = {}
            for field, ref in stage['inputs'].items():
                try:
                    mapped[field] = read_ref(ref, inputs, values)
                except KeyError:
                    if stage_interface[field].required:
                        raise
        except KeyError:
            statuses[name] = 'review_blocked'
            skipped.append(qualified)
            continue
        if stage['kind'] == 'leaf':
            program = Program.model_validate(stage['program'])
            result = Runtime(program, backend).run(mapped)
            values[name] = result['decisions']
            executed.append(qualified)
            reviews[name] = any(item['review_required'] for item in result['decisions'].values())
        else:
            result = execute_graph(definitions[stage['definition']], definitions, mapped,
                                   backend, prefix=qualified + '/')
            values[name] = result['decisions']
            executed.extend(result['executed'])
            skipped.extend(result['skipped'])
            reviews[name] = result['status'] == 'review_required'
        reviews[name] = reviews[name] or any(reviews.get(dep, False)
                                            for dep in required_deps | condition_deps)
        statuses[name] = 'completed'
    finals = {}
    final_review = False
    for public_name, mapping in graph['final'].items():
        available = [candidate for candidate in mapping['candidates']
                     if statuses[candidate['stage']] == 'completed']
        require(len(available) <= 1, 'ambiguous_final')
        if not available:
            continue
        candidate = available[0]
        selected = dict(values[candidate['stage']][candidate['decision']])
        if candidate.get('distribution_scope') == 'branch_conditional':
            selected['value'] = candidate['label_map'][selected['value']]
        selected['distribution_scope'] = candidate['distribution_scope']
        selected['review_required'] = bool(selected['review_required'] or reviews[candidate['stage']])
        final_review = final_review or selected['review_required']
        finals[public_name] = selected
    return {'status': 'completed' if len(finals) == len(graph['outputs']) and not final_review
            else 'review_required',
            'decisions': finals, 'executed': executed, 'skipped': skipped}


def run(doc, state):
    source = validate(doc)
    projected = project_state(source.state, state)
    backend = ManagedBackend(MockBackend(), max_calls=doc['limits']['max_native_calls_per_example'])
    try:
        result = execute_graph(doc['graph'], doc['definitions'], projected, backend)
        result['accounting'] = backend.accounting()
        result['synthetic'] = True
        return result
    finally:
        backend.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='No-key hierarchy contract preview')
    parser.add_argument('fixture', type=Path)
    args = parser.parse_args()
    fixture = load_document(args.fixture)
    cases = load_document(Path(__file__).parent / 'cases.json')
    state = cases[args.fixture.name]['state']
    print(json.dumps(run(fixture, state), indent=2))
