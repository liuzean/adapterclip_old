"""Training-profile contracts; authored for explicit future execution only."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'models_test'))
from train_rma import parse_args, TRAINING_PROFILES


def test_legacy_profile_preserves_existing_experiment_nine():
    args=parse_args([])
    assert args.training_profile=='legacy_rma'
    assert all(getattr(args,k)==v for k,v in TRAINING_PROFILES['legacy_rma'].items())
    assert args.sampling_method=='sample_without_replacement'


def test_adaptclip_profile_resolves_all_seven_settings_together():
    args=parse_args(['--training_profile','adaptclip'])
    assert (args.batch_size,args.accumulation,args.optimizer)==(8,1,'adam')
    assert args.train_reference_shots==[1] and args.sampling_method=='choice'
    assert args.learning_rate==.001 and args.betas==[.5,.999] and args.weight_decay==0
    assert args.profile_overrides==[]


def test_explicit_override_is_recorded_instead_of_silently_relabelled():
    args=parse_args(['--training_profile','adaptclip','--batch_size','4'])
    assert args.batch_size==4 and args.profile_overrides==['batch_size']
    assert args.optimizer=='adam' and args.train_reference_shots==[1]


@pytest.mark.parametrize('flags', [['--betas','1','0.999'],['--betas','nan','0.9'],
                                  ['--train_reference_shots','1','1']])
def test_invalid_profile_values_are_rejected(flags):
    with pytest.raises(SystemExit):
        parse_args(flags)
