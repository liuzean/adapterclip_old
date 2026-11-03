"""Read-only audit and copy of completed original-flow DATASET test prefixes."""
import math
import shutil
from pathlib import Path

from fusion_ablation_utils import read_csv, read_json
from rma_utils import FIELDS, digest


def checked_rows(path):
    if b'\x00' in Path(path).read_bytes():
        raise ValueError(f'Zero-filled/corrupt result file: {path}')
    rows = read_csv(path)
    if not rows:
        raise ValueError(f'Empty result file: {path}')
    return rows


def audit_round(directory, flow, checkpoint_hash, audit, dataset, shot):
    """A round can only be reused with complete state, identity and metric checks."""
    state = read_json(directory/'run_status.json')
    if not all(state.get(k) is True for k in ('complete','baseline_parity_passed','frozen_integrity_passed')):
        raise ValueError('Round is incomplete or failed parity.')
    if state.get('flow') != flow or state.get('seed') != 10 or state.get('k_shots') != shot:
        raise ValueError('Round flow/seed/shot changed.')
    config = read_json(directory/'experiment_config.json')
    if config['joint_sha256'] != checkpoint_hash or config['training_flow'] != flow or config['disable_rma'] is not False:
        raise ValueError('Round checkpoint/flow changed.')
    if read_json(directory/'input_audit.json') != audit:
        raise ValueError('Historical inputs changed.')
    source_config = read_json(Path(audit['source'])/'experiment_config.json')
    for key in ('dataset','seed','k_shots','test_data_path','pretrained_model','checkpoint_path','features_list',
                'image_size','batch_size','sigma','n_ctx','vl_reduction','pq_mid_dim','pq_context',
                'visual_learner','textual_learner','pq_learner','fusion_type'):
        if config[key] != source_config[key]:
            raise ValueError(f'Evaluation protocol changed: {dataset}/{shot}/{key}')
    for name in ('baseline_reference_comparison.csv','baseline_score_checks.csv','baseline_map_checks.csv'):
        if not all(row['Passed']=='True' for row in checked_rows(directory/name)):
            raise ValueError(f'Failed recorded check: {name}')
    manifest = checked_rows(directory/'sample_manifest.csv')
    original_manifest = checked_rows(Path(audit['source'])/'sample_manifest.csv')
    if manifest != original_manifest:
        raise ValueError('Query identities changed.')
    references = checked_rows(directory/'references.csv')
    expected_refs = []
    for line in (Path(audit['source'])/'reference_samples.txt').read_text(encoding='utf-8-sig').splitlines():
        category, value = line.split('\t',1)
        path = Path(value)
        if not path.is_absolute():
            path = Path(source_config['test_data_path'])/path
        expected_refs.append((category,str(path.resolve()).casefold()))
    if [(r['Category'],str(Path(r['Image Path']).resolve()).casefold()) for r in references] != expected_refs:
        raise ValueError('Reference identities changed.')
    scores = checked_rows(directory/'image_scores.csv')
    if len(scores) != state['sample_count'] or len(scores) != len(manifest):
        raise ValueError('Image score coverage changed.')
    candidate = f'S_{flow}'
    for score,sample in zip(scores,manifest):
        if (score['Category'] != sample['Category'] or score['Image Path'] != sample['Image Path']
                or int(score['Label']) != int(sample['Sample Type']=='Anomaly')):
            raise ValueError('Image score identity/order changed.')
        if not all(math.isfinite(float(score[f'{c} Image Score'])) for c in ('baseline',candidate)):
            raise ValueError('Nonfinite image scores.')
    metrics = checked_rows(directory/'metrics_combined.csv')
    categories = {r['Category'] for r in manifest}
    keys = [(r['Category'],r['Condition']) for r in metrics]
    expected = {(cat,c) for cat in categories|{'Mean'} for c in ('baseline',candidate)}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('Incomplete/duplicate category metrics.')
    if not all(math.isfinite(float(row[f])) and 0 <= float(row[f]) <= 100 for row in metrics for f in FIELDS):
        raise ValueError('Invalid six-metric result.')
    for condition in ('baseline',candidate):
        mean_row = next(r for r in metrics if r['Category']=='Mean' and r['Condition']==condition)
        group = [r for r in metrics if r['Category']!='Mean' and r['Condition']==condition]
        if any(abs(sum(float(r[f]) for r in group)/len(group)-float(mean_row[f])) > 1e-8 for f in FIELDS):
            raise ValueError('Category macro mean mismatch.')


def reusable_datasets(previous, args, jobs, audit, root, entry_stem):
    """Stop at first incomplete dataset and rerun ALL its requested shots.

    Thus corrupt DTD-1 state causes DTD 1/2/4 and all later datasets to rerun.
    Earlier valid datasets are copied to a fresh session; old files stay intact.
    """
    previous = Path(previous).resolve()
    previous.relative_to((root/'results'/entry_stem).resolve())
    campaign = read_json(previous/'campaign_config.json')
    checkpoint_hash = digest(args.checkpoint)
    if (campaign['flow'] != args.flow or campaign['checkpoint_sha256'] != checkpoint_hash
            or Path(campaign['checkpoint']).resolve() != args.checkpoint.resolve()
            or campaign['datasets'] != args.datasets or campaign['shots'] != args.shots or campaign['seed'] != 10):
        raise ValueError('Resume test session has a different checkpoint, flow or scope.')
    reused, reason = [], None
    for dataset in args.datasets:
        group = [(d,s) for d,s in jobs if d == dataset]
        try:
            for d,s in group:
                audit_round(previous/d/f'{s}shot_seed10',args.flow,checkpoint_hash,audit[d][str(s)],d,s)
        except (OSError,ValueError,KeyError,TypeError) as error:
            reason = dict(dataset=dataset,reason=str(error))
            break
        reused.extend(group)
    return reused,reason


def copy_round(previous, session, dataset, shot):
    """Copy to a new directory; never mutate or rename the previous result."""
    source = Path(previous)/dataset/f'{shot}shot_seed10'
    destination = session/dataset/f'{shot}shot_seed10'
    shutil.copytree(source,destination)
    for path in source.rglob('*'):
        if path.is_file() and digest(path) != digest(destination/path.relative_to(source)):
            raise RuntimeError('Copied test result failed checksum verification.')
