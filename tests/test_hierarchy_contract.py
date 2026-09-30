"""Reference examples pin routes and invalid boundaries before graph runtime work."""
import copy
import json
from pathlib import Path
import sys

import pytest

FIXTURES = Path(__file__).resolve().parents[1] / 'examples' / 'hierarchy_contract'
sys.path.insert(0, str(FIXTURES))
from preview import ContractError, run, validate  # noqa: E402


@pytest.fixture(scope='module')
def cases():
    return json.loads((FIXTURES / 'cases.json').read_text(encoding='utf8'))


@pytest.mark.parametrize('name', ['chain.json', 'conditional.json', 'diamond.json', 'nested.json'])
def test_valid_reference_graphs_use_expected_native_stages(name, cases):
    fixture = json.loads((FIXTURES / name).read_text(encoding='utf8'))
    expected = cases[name]['expected']
    result = run(fixture, cases[name]['state'])
    assert result['synthetic'] is True
    assert result['status'] == expected['status']
    assert result['executed'] == expected['executed']
    assert result['skipped'] == expected.get('skipped', [])
    assert result['accounting']['requests_attempted'] == len(expected['executed'])
    if 'distribution_scope' in expected:
        assert result['decisions']['resolution']['distribution_scope'] == expected['distribution_scope']
        assert set(result['decisions']['resolution']['probabilities']) != set(fixture['source']['decisions']['resolution']['criteria'])
    if 'qualified_instances' in expected:
        assert result['executed'] == expected['qualified_instances']


@pytest.mark.parametrize('name', ['invalid_cycle.json', 'invalid_unbound.json', 'invalid_incomplete.json'])
def test_invalid_graphs_fail_before_a_backend_can_be_constructed(name, cases, monkeypatch):
    fixture = json.loads((FIXTURES / name).read_text(encoding='utf8'))
    import preview
    def forbid_backend(*args, **kwargs):
        raise AssertionError('Backend construction occurred before graph validation')
    monkeypatch.setattr(preview, 'ManagedBackend', forbid_backend)
    with pytest.raises(ContractError, match=cases[name]['expected']['error']):
        run(fixture, cases[name]['state'])


def test_unmatched_branch_returns_review_without_invented_output(cases):
    fixture = json.loads((FIXTURES / 'conditional.json').read_text(encoding='utf8'))
    result = run(fixture, {'message': 'press partnership'})
    assert result['status'] == 'review_required'
    assert result['decisions'] == {}
    assert result['executed'] == ['router']
    assert result['accounting']['requests_attempted'] == 1


def test_second_specialist_path_keeps_public_choice_labels():
    fixture = json.loads((FIXTURES / 'conditional.json').read_text(encoding='utf8'))
    result = run(fixture, {'message': 'software crash bug'})
    assert result['status'] == 'completed'
    assert result['executed'] == ['router', 'technical']
    assert result['skipped'] == ['billing']
    selected = result['decisions']['resolution']
    assert selected['value'] == 'bug_fix'
    assert selected['value'] in fixture['source']['decisions']['resolution']['criteria']
    assert selected['distribution_scope'] == 'branch_conditional'


def test_parent_review_cannot_be_erased_by_a_confident_child(cases):
    fixture = json.loads((FIXTURES / 'conditional.json').read_text(encoding='utf8'))
    router = fixture['graph']['stages'][0]
    router['program']['policies']['department']['force_review'] = True
    router['on_review'] = 'continue_marked'
    result = run(fixture, cases['conditional.json']['state'])
    assert result['executed'] == ['router', 'billing']
    assert result['status'] == 'review_required'
    assert result['decisions']['resolution']['review_required'] is True
    router['on_review'] = 'defer'
    result = run(fixture, cases['conditional.json']['state'])
    assert result['status'] == 'review_required'
    assert result['decisions'] == {}
    assert result['accounting']['requests_attempted'] == 1


def test_recursive_subgraph_and_model_drift_fail_closed(cases):
    nested = json.loads((FIXTURES / 'nested.json').read_text(encoding='utf8'))
    recursive = copy.deepcopy(nested)
    recursive['definitions']['check_note']['stages'][0] = {
        'id': 'again', 'kind': 'subgraph', 'definition': 'check_note',
        'inputs': {'note': {'root': 'note'}}}
    recursive['definitions']['check_note']['final']['review']['candidates'][0]['stage'] = 'again'
    with pytest.raises(ContractError, match='recursive_definition'):
        validate(recursive)
    chain = json.loads((FIXTURES / 'chain.json').read_text(encoding='utf8'))
    chain['graph']['stages'][0]['program']['model'] = 'other-model'
    with pytest.raises(ContractError, match='model_drift'):
        validate(chain)
