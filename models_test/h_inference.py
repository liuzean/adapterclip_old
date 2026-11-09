"""Small H/E inference API; no historical scores used to produce predictions."""
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from fusion_ablation_utils import path_key, read_csv, read_json
from rma_utils import digest

WEIGHT_SHA = '2ec4a3c9164f27fbad061b8e814346986a5289960c63dd8d0097e6c1ebc60704'
COEFFICIENT = .25
LOCAL_FRACTION = .125


def recorded_inputs(reference_run, dataset, shot):
    """Use exact normal TRAIN references and original test query order."""
    folder = Path(reference_run)/dataset/f'{shot}shot_seed10'
    status = read_json(folder/'run_status.json')
    if not all(status.get(key) is True for key in ('complete', 'H_six_metric_parity_passed', 'H_pixel_identity_passed')):
        raise ValueError('Reference round must be completed H real-inference verification.')
    for name in ('references.csv','sample_manifest.csv','input_audit.json'):
        if digest(folder/name) != status['H_output_sha256'][name]:
            raise ValueError('Recorded input ledger changed after H verification.')
    config = read_json(folder/'experiment_config.json')
    if (config['checkpoint_sha256'] != WEIGHT_SHA or config['weight_epoch'] != 14 or config['seed'] != 10
            or config['k_shots'] != shot or config['image_size'] != 518 or config['sigma'] != 4
            or config['features_list'] != [6,12,18,24]
            or not all(config[key] for key in ('visual_learner','textual_learner','pq_learner','pq_context'))):
        raise ValueError('Reference protocol differs from fixed H.')
    root = Path(config['test_data_path']); meta_path = root/'meta.json'
    audit = read_json(folder/'input_audit.json')
    if digest(meta_path) != audit['metadata_sha256']:
        raise ValueError('Target metadata changed since H verification.')
    meta = read_json(meta_path); classes = list(meta['test'])
    normal = {path_key(root/row['img_path']):row for rows in meta['train'].values() for row in rows if int(row['anomaly']) == 0}
    refs, paths = [], read_csv(folder/'references.csv')
    for row in paths:
        path = Path(row['Image Path']); path.resolve().relative_to(root.resolve())
        item = normal.get(path_key(path))
        if item is None or item['cls_name'] != row['Category'] or not path.is_file():
            raise ValueError('Reference is not the recorded normal TRAIN image.')
        refs.append(item)
    if Counter(row['Category'] for row in paths) != Counter({c:shot for c in classes}):
        raise ValueError('Reference order/multiplicity/shot count differs.')
    manifest = read_csv(folder/'sample_manifest.csv')
    queries = [row for category in classes for row in meta['test'][category]]
    if len(manifest) != len(queries):
        raise ValueError('Query count changed.')
    for actual, expected in zip(queries, manifest):
        if (actual['cls_name'] != expected['Category'] or path_key(root/actual['img_path']) != path_key(expected['Image Path'])
                or ('Anomaly' if int(actual['anomaly']) else 'Normal') != expected['Sample Type']):
            raise ValueError('Query identity/order/label changed.')
        if not (root/actual['img_path']).is_file() or (int(actual['anomaly']) and not (root/actual['mask_path']).is_file()):
            raise FileNotFoundError(actual['img_path'])
    return dict(config=config, classes=classes, references=refs, reference_paths=paths,
                manifest=manifest, history=folder, meta=meta_path)


def load_model(checkpoint, inputs, output, workers, *, expected_sha=WEIGHT_SHA,
               expected_epoch=14, expected_matching='learned'):
    """Load a verified four-adapter payload; defaults remain fixed original H.

    The matched control supplies its own verified best hash/epoch/matching mode;
    this does not loosen test_h's fixed epoch14/learned checkpoint contract.
    """
    import torch
    import adaptcliplib
    from adaptcliplib import TextualAdapter, VisualAdapter, PQAdapter
    from dataset import Dataset
    from fusion_ablation_utils import RecordedPromptDataset
    from rma_model import ReferenceMatchingAdapter
    from rma_scratch_common import validate_scratch
    from rma_utils import freeze_base
    from test import build_prompt_memory, resolve_pretrained_model_type
    from tools import get_transform, setup_seed
    if not torch.cuda.is_available():
        raise RuntimeError('Existing CLIP text adapter requires CUDA.')
    setup_seed(10); checkpoint = Path(checkpoint)
    if digest(checkpoint) != expected_sha:
        raise ValueError('Checkpoint identity differs from the declared experiment.')
    manifest = read_json(checkpoint.parent/'model_manifest.json')
    config = SimpleNamespace(**inputs['config'])
    if digest(config.pretrained_model) != manifest['pretrained_sha256']:
        raise ValueError('CLIP pretrained weight identity differs.')
    if resolve_pretrained_model_type(str(config.pretrained_model)) != 'ViT-L/14@336px':
        raise ValueError('Requires original CLIP ViT-L/14@336px projected768 interface.')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    validate_scratch(payload, manifest['base_sha256'], manifest['pretrained_sha256'], inputs['config'], 'new')
    if payload['epoch'] != expected_epoch or payload['model_spec']['matching'] != expected_matching:
        raise ValueError('Checkpoint epoch/matching mode differs from the declared experiment.')
    device='cuda'; model,_=adaptcliplib.load(config.pretrained_model,device=device)
    model.visual.DAPM_replace(DPAM_layer=20)
    textual=TextualAdapter(model.to('cpu'),config.image_size,config.n_ctx)
    visual=VisualAdapter(config.image_size,14,input_dim=768,reduction=config.vl_reduction)
    pq=PQAdapter(config.image_size,14,context=config.pq_context,input_dim=768,
                 mid_dim=config.pq_mid_dim,layers_num=len(config.features_list))
    rma=ReferenceMatchingAdapter(**payload['model_spec'])
    for name,module in (('textual_learner',textual),('visual_learner',visual),('pq_learner',pq),('rma',rma)):
        module.load_state_dict(payload[name],strict=True)
    del payload
    for module in (model,textual,visual,pq,rma):
        module.to(device).eval().requires_grad_(False)
    textual.prepare_static_text_feature(model)
    transform,target_transform=get_transform(image_size=config.image_size)
    prompts=RecordedPromptDataset(config.test_data_path,inputs['references'],inputs['classes'],transform,target_transform)
    queries=Dataset(root=config.test_data_path,transform=transform,target_transform=target_transform,
                    dataset_name=config.dataset,k_shots=config.k_shots,save_dir=str(output),mode='test',seed=10)
    reference_loader=torch.utils.data.DataLoader(prompts,batch_size=config.batch_size,shuffle=False)
    loader=torch.utils.data.DataLoader(queries,batch_size=config.batch_size,shuffle=False,num_workers=workers)
    with torch.no_grad():
        ctx,tokens=textual(); text=model.encode_text_learn(ctx,tokens).float()
        memory=build_prompt_memory(model,reference_loader,device,queries.obj_list,queries.view_list,config.features_list,20)
    snapshot=freeze_base([None,model,textual,visual,pq]); rma_snapshot=freeze_base([None,rma])
    return SimpleNamespace(config=config,model=model,adapters=(textual,visual,pq),text=text,rma=rma,
                           memory=memory,loader=loader,device=device,snapshots=(snapshot,rma_snapshot))


def select_score(e_score, native, mode):
    from rma_score_coordination_utils import mix_scores
    if mode not in ('H','E'):
        raise ValueError('Mode must be H or E.')
    return e_score if mode == 'E' else mix_scores(e_score,native,COEFFICIENT)


def predict(state, images, classes, paths, mode='H'):
    """Return one final [B,H,W] map, [B] chosen score, and scalar score parts.

    Reuse existing adapter/fusion operations. No history label or score input.
    G0_L1's complete final score is SE; localization ALWAYS follows E12.5%.
    """
    from fusion_ablation import infer_batch
    from rma_fixed_fusion_utils import gated_output
    from rma_output_ablation_utils import route_outputs
    from test import prompt_association
    features,patches=state.model.encode_image(images.to(state.device),state.config.features_list,DPAM_layer=20)
    pf,pp=prompt_association(*state.memory,classes)
    shared=(features,pf,*patches,*pp);versions=[value._version for value in shared]
    old,maps,scores,_,_,native,extra=infer_batch(state.config,features,patches,pf,pp,*state.adapters,state.text,paths)
    g,local=state.rma(patches,pp)
    raw,routed,scored,_=route_outputs(old,maps,scores,native,extra,g,local,state.config.sigma)
    _,pixel,_,_=gated_output(old,native,extra,local,None,LOCAL_FRACTION,state.config.sigma,
                            (raw['G0_L1'],routed['G0_L1'],scored['G0_L1']))
    e_score=scored['G0_L1']; native=native.detach().cpu()
    score=select_score(e_score,native,mode)
    if versions != [value._version for value in shared]:
        raise ValueError('Shared CLIP features mutated.')
    return pixel,score,dict(E=e_score,G0=native)
