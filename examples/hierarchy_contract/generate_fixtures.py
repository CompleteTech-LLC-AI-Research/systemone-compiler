"""Generate synthetic hierarchy contract fixtures from validated flat programs."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from s1compiler.architect import template_program
from s1compiler.models import Decision, StateField, UseCase

HERE = Path(__file__).resolve().parent
MODEL = 'jev-1.13.0'


def S(description=''):
    return StateField(type='string', description=description)


def B():
    return StateField(type='boolean')


def N():
    return StateField(type='number')


def choice(labels, goal):
    return Decision(type='choice', goal=goal,
                    criteria={name: description for name, description in labels.items()})


def noul(goal):
    return Decision(type='noul', goal=goal,
                    criteria={'true': 'urgent outage or confirmed risk', 'false': 'routine request or no risk'})


def score(goal):
    return Decision(type='score', goal=goal, criteria=['low routine risk', 'medium concern', 'high urgent risk'])


def usecase(name, state, decisions):
    return UseCase(name=name, model=MODEL, state=state, decisions=decisions).model_dump(mode='json')


def program(name, state, decisions):
    source = UseCase(name=name, model=MODEL, state=state, decisions=decisions)
    result = template_program(source)
    for policy in result.policies.values():
        policy.min_gate = 0.0  # Synthetic routes do not depend on mock confidence.
    return result.model_dump(mode='json')


def root(field):
    return {'root': field}


def ref(stage, decision, field='value'):
    return {'stage': stage, 'decision': decision, 'field': field}


def stage(name, leaf, inputs, when=None, after=None):
    result = {'id': name, 'kind': 'leaf', 'program': leaf, 'inputs': inputs}
    if when is not None:
        result['when'] = when
    if after:
        result['after'] = after
    return result


def when_eq(stage_name, decision, value):
    return {'all': [{'ref': ref(stage_name, decision), 'op': 'eq', 'value': value}]}


def final(*candidates):
    return {'candidates': list(candidates), 'on_missing': 'review_required'}


def candidate(name, decision, label_map=None):
    result = {'stage': name, 'decision': decision}
    if label_map:
        result['label_map'] = label_map
        result['distribution_scope'] = 'branch_conditional'
    else:
        result['distribution_scope'] = 'full_contract'
    return result


def fixture(source, graph, *, definitions=None, sample_state, expected):
    return {'format': 'systemone-hierarchy-source/v1', 'source': source,
            'limits': {'max_expanded_nodes': 64, 'max_depth': 8, 'max_native_calls_per_example': 64},
            'definitions': definitions or {}, 'graph': graph,
            '_fixture': {'synthetic': True, 'state': sample_state, 'expected': expected}}


def build():
    message = {'message': S('Synthetic ticket text')}
    signal = stage('signal', program('signal', message, {'urgent': noul('Is this an urgent outage?')}),
                   {'message': root('message')})
    priority = stage('priority', program('priority', {'message': S(), 'urgent': B()},
                                         {'priority': score('What is the ticket priority?')}),
                     {'message': root('message'), 'urgent': ref('signal', 'urgent')})
    chain_source = usecase('priority_chain', message, {'priority': score('What is the ticket priority?')})
    chain_graph = {'inputs': chain_source['state'], 'outputs': chain_source['decisions'],
                   'stages': [signal, priority], 'final': {'priority': final(candidate('priority', 'priority'))}}
    chain = fixture(chain_source, chain_graph, sample_state={'message': 'urgent outage'},
                    expected={'status': 'completed', 'executed': ['signal', 'priority']})

    resolution = choice({'refund': 'refund invoice charge', 'bug_fix': 'fix broken software',
                         'general': 'general response'}, 'Which response category applies?')
    route = choice({'billing': 'invoice charge refund', 'technical': 'software crash bug',
                    'other': 'press partnership'}, 'Which specialist should inspect the ticket?')
    router = stage('router', program('router', message, {'department': route}),
                   {'message': root('message')})
    billing_labels = {'refund': 'refund invoice charge', 'general': 'general response'}
    technical_labels = {'bug_fix': 'fix broken software', 'general': 'general response'}
    billing = stage('billing', program('billing', message,
                    {'resolution': choice(billing_labels, 'Which billing response applies?')}),
                    {'message': root('message')}, when_eq('router', 'department', 'billing'))
    technical = stage('technical', program('technical', message,
                      {'resolution': choice(technical_labels, 'Which technical response applies?')}),
                      {'message': root('message')}, when_eq('router', 'department', 'technical'))
    specialist_source = usecase('specialist_routing', message, {'resolution': resolution})
    specialist_graph = {'inputs': specialist_source['state'], 'outputs': specialist_source['decisions'],
                        'stages': [router, billing, technical],
                        'final': {'resolution': final(candidate('billing', 'resolution',
                                                         {label: label for label in billing_labels}),
                                                      candidate('technical', 'resolution',
                                                         {label: label for label in technical_labels}))}}
    specialist = fixture(specialist_source, specialist_graph,
                         sample_state={'message': 'refund invoice charge'},
                         expected={'status': 'completed', 'executed': ['router', 'billing'],
                                   'skipped': ['technical'], 'distribution_scope': 'branch_conditional'})

    diamond_state = {'diff': S('Synthetic diff summary')}
    breaking = stage('breaking', program('breaking', diamond_state,
                     {'breaking': noul('Does the diff remove a public contract?')}),
                     {'diff': root('diff')})
    operational = stage('operational', program('operational', diamond_state,
                        {'severity': score('How broad is the operational impact?')}),
                        {'diff': root('diff')})
    combined = stage('combined', program('combined', {'breaking': B(), 'severity': N()},
                      {'risk': score('What is the combined risk level?')}),
                     {'breaking': ref('breaking', 'breaking'),
                      'severity': ref('operational', 'severity')})
    diamond_source = usecase('advisory_diamond', diamond_state,
                             {'risk': score('What is the combined risk level?')})
    diamond_graph = {'inputs': diamond_source['state'], 'outputs': diamond_source['decisions'],
                     'stages': [breaking, operational, combined],
                     'final': {'risk': final(candidate('combined', 'risk'))}}
    diamond = fixture(diamond_source, diamond_graph, sample_state={'diff': 'urgent outage'},
                      expected={'status': 'completed', 'executed': ['breaking', 'operational', 'combined']})

    check_state = {'note': S('Synthetic note')}
    check_decisions = {'review': noul('Does this note require risk review?')}
    check = {'inputs': {k: v.model_dump(mode='json') for k, v in check_state.items()},
             'outputs': {k: v.model_dump(mode='json') for k, v in check_decisions.items()},
             'stages': [stage('check', program('check', check_state, check_decisions),
                              {'note': root('note')})],
             'final': {'review': final(candidate('check', 'review'))}}
    nested_state = {'first_note': S(), 'second_note': S()}
    nested_decisions = {'first_review': noul('Does the first note require review?'),
                        'second_review': noul('Does the second note require review?')}
    nested_source = usecase('nested_reuse', nested_state, nested_decisions)
    nested_graph = {'inputs': nested_source['state'], 'outputs': nested_source['decisions'],
                    'stages': [{'id': 'first', 'kind': 'subgraph', 'definition': 'check_note',
                                'inputs': {'note': root('first_note')}},
                               {'id': 'second', 'kind': 'subgraph', 'definition': 'check_note',
                                'inputs': {'note': root('second_note')}}],
                    'final': {'first_review': final(candidate('first', 'review')),
                              'second_review': final(candidate('second', 'review'))}}
    nested = fixture(nested_source, nested_graph, definitions={'check_note': check},
                     sample_state={'first_note': 'urgent outage', 'second_note': 'routine request'},
                     expected={'status': 'completed', 'executed': ['first/check', 'second/check'],
                               'qualified_instances': ['first/check', 'second/check']})

    invalid_cycle = copy.deepcopy(chain)
    invalid_cycle['graph']['stages'][0]['after'] = ['priority']
    invalid_cycle['_fixture']['expected'] = {'error': 'cycle'}
    invalid_unbound = copy.deepcopy(chain)
    invalid_unbound['graph']['stages'][0]['inputs']['message'] = root('undeclared')
    invalid_unbound['_fixture']['expected'] = {'error': 'undeclared_root_field'}
    invalid_incomplete = copy.deepcopy(specialist)
    invalid_incomplete['graph']['final'] = {}
    invalid_incomplete['_fixture']['expected'] = {'error': 'missing_final_output'}
    return {'chain.json': chain, 'conditional.json': specialist, 'diamond.json': diamond,
            'nested.json': nested, 'invalid_cycle.json': invalid_cycle,
            'invalid_unbound.json': invalid_unbound, 'invalid_incomplete.json': invalid_incomplete}


if __name__ == '__main__':
    HERE.mkdir(parents=True, exist_ok=True)
    cases = {}
    for name, data in build().items():
        cases[name] = data.pop('_fixture')
        with (HERE / name).open('w', encoding='utf8', newline='\n') as stream:
            stream.write(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
        print(name)
    with (HERE / 'cases.json').open('w', encoding='utf8', newline='\n') as stream:
        stream.write(json.dumps(cases, indent=2, ensure_ascii=False) + '\n')
