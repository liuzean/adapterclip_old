"""Reference identity and sequential scheduling contracts; not auto-executed."""
import ast
import sys
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'models_test'))
from new_d_without_rma import check_identities, parse_args


def test_default_campaign_is_six_datasets_each_all_shots():
    args=parse_args([])
    assert args.datasets==['Visa','BTAD','MPDD','SDD','DTD','part_glass']
    assert args.shots==[1,2,4]


def test_duplicate_references_and_order_are_not_reduced_to_sets(tmp_path):
    a,b=str(tmp_path/'a.png'),str(tmp_path/'b.png')
    expected=[('class',a),('class',a),('class',b)]
    assert all(r['Passed'] for r in check_identities(expected,expected,'Reference'))
    changed=[('class',a),('class',b),('class',a)]
    assert not all(r['Passed'] for r in check_identities(changed,expected,'Reference'))
    with pytest.raises(ValueError,match='count differs'):
        check_identities(expected[:2],expected,'Reference')


def test_query_labels_are_part_of_identity(tmp_path):
    p=str(tmp_path/'a.png')
    assert not check_identities([('class',p,0)],[('class',p,1)],'Query')[0]['Passed']


def test_entry_delegates_prediction_to_original_test_and_has_no_rma_constructor():
    tree=ast.parse((ROOT/'models_test/new_d_without_rma.py').read_text(encoding='utf-8'))
    calls=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
    assert 'original.test' in calls and not any('ReferenceMatchingAdapter' in n for n in calls)
    assert 'run_child' in calls and 'cleanup' in calls
