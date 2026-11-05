"""Four fixed global/local switches; no torch import in parent schedulers."""
from fusion_ablation_utils import read_csv, read_json, write_csv
from rma_utils import FIELDS, MetricsBuffer

CONDITIONS=('G0_L0','G1_L0','G0_L1','G1_L1')
UNIQUE_MAPS=('baseline','G0_L0','G1_L1')
PIXEL_SOURCE={'G0_L0':'G0_L0','G1_L0':'G0_L0','G0_L1':'G1_L1','G1_L1':'G1_L1'}


def save_sample(args,source,output,sample,image,mask,raw,maps,index):
    """Reuse original baseline checks/matrix writer with accurate new figure labels."""
    from pathlib import Path
    from types import SimpleNamespace
    from fusion_ablation_utils import save_panels
    from test_rma_joint import save_example
    options=SimpleNamespace(**dict(vars(args),save_visualizations=False))
    checks=save_example(options,source,output,sample,image,mask,raw,maps,index)
    if args.save_visualizations:
        relative=Path(sample['Category'])/sample['Sample Type']/sample['Sample Key']
        arrays={n:v[index].detach().cpu().numpy() for n,v in maps.items()}
        original={n:v[index].detach().cpu().numpy() for n,v in raw.items()}
        save_panels(output/'visualizations'/relative.with_suffix('.png'),image,mask.cpu().numpy(),
                    original,arrays,['baseline',*CONDITIONS],3,'Original A / new D RMA global-local switches')
    return checks


def route_outputs(old,maps,scores,fixed,extra,global_logits,local_logits,sigma):
    """G0=three-native global mean, G1=four; L0=three map, L1=four.

    Inputs are the new-D three-adapter infer_batch outputs, never original A.
    Global logits [B,2], local logits [B,2,H,W]. Return raw/maps/scores dicts
    with exact endpoint aliases; each image score uses its own local map max.
    """
    from adaptcliplib import fusion_fun
    from rma_utils import fuse_rma
    raw1,map1,score11=fuse_rma(old,extra,fixed,global_logits,local_logits,sigma)
    native1=(3*fixed+global_logits.softmax(-1)[:,1])/4
    max0=maps['baseline'].flatten(1).max(1).values.to(fixed.device)
    max1=map1.flatten(1).max(1).values.to(fixed.device)
    raw={'G0_L0':old['baseline'],'G1_L0':old['baseline'],'G0_L1':raw1,'G1_L1':raw1}
    evaluated={'G0_L0':maps['baseline'],'G1_L0':maps['baseline'],'G0_L1':map1,'G1_L1':map1}
    image={'G0_L0':scores['baseline'],'G1_L0':fusion_fun([native1,max0],fusion_type='harmonic_mean').cpu(),
           'G0_L1':fusion_fun([fixed,max1],fusion_type='harmonic_mean').cpu(),'G1_L1':score11}
    return raw,evaluated,image,dict(G0=fixed,G1=native1,L0_Max=max0,L1_Max=max1)


class FactorBuffer(MetricsBuffer):
    """Cache baseline and TWO candidate map arrays; evaluate all four image scores."""
    def __init__(self,category):
        super().__init__(category,conditions=UNIQUE_MAPS)
        self.extra_scores={n:[] for n in ('G1_L0','G0_L1')}

    def append(self,mask,label,maps,scores,index):
        import torch
        for n,source in PIXEL_SOURCE.items():
            if maps[n] is not maps[source] and not torch.equal(maps[n],maps[source]):
                raise ValueError('Global switch changed pixel outputs.')
        super().append(mask,label,maps,scores,index)
        for name in self.extra_scores:
            if not scores[name][index].isfinite().all():
                raise ValueError('Nonfinite mixed-condition image score.')
            self.extra_scores[name].append(scores[name][index].detach().cpu().clone())

    def finish(self,device):
        import torch
        from branch_diagnostics_utils import evaluate_scores
        from tools import Evaluator
        labels=torch.tensor(self.labels).int()
        pixels,images=super().finish(device)
        lookup={r['Condition']:r for r in pixels}
        for name in self.extra_scores:
            pixels.append(dict(lookup[PIXEL_SOURCE[name]],Condition=name))
            score=torch.stack(self.extra_scores[name]);self.extra_scores[name].clear()
            dummy=torch.zeros((len(labels),1,1))
            evaluator=Evaluator(device,metrics=[f'I-{m}' for m in ('AUROC','AP','F1max')])
            images.append(dict(Category=self.category,Condition=name,**evaluate_scores(
                evaluator,self.category,dummy,dummy,labels,score,device,'image')))
        return pixels,images


def audit_history(source,audit,full,off):
    """Validate endpoint identities and actual evaluation protocol before inference."""
    for directory in (full,off):
        state=read_json(directory/'run_status.json')
        if state.get('complete') is not True or read_json(directory/'input_audit.json')!=audit:
            raise ValueError('Historical endpoint is incomplete or has different audited inputs.')
        references=read_csv(directory/'references.csv')
        if [(r['Category'],r['Image Path']) for r in references]!=list(source['reference_paths']):
            raise ValueError('Historical reference identity/order/multiplicity changed.')
        if read_csv(directory/'sample_manifest.csv')!=source['manifest']:
            raise ValueError('Historical query identities changed.')
        config=read_json(directory/'experiment_config.json')
        for key in ('dataset','seed','k_shots','test_data_path','pretrained_model','features_list','image_size',
                    'batch_size','sigma','n_ctx','vl_reduction','pq_mid_dim','pq_context','visual_learner',
                    'textual_learner','pq_learner','fusion_type'):
            if config[key]!=source['config'][key]:
                raise ValueError(f'Historical endpoint protocol differs: {key}')
    full_config,off_config=[read_json(d/'experiment_config.json') for d in (full,off)]
    if (full_config['joint_sha256']!=off_config['checkpoint_sha256'] or off_config['weight_epoch']!=14
            or off_config['rma_used'] is not False or off_config['align_used'] is not True):
        raise ValueError('Historical endpoint weight identity or RMA/ALIGN mode differs.')


def endpoint_metric_checks(output,table,full,off):
    checks=[]
    current={(r['Category'],r['Condition']):r for r in table}
    for directory,candidate,historical in [(full,'G1_L1','S_new'),(off,'G0_L0','new_D_without_RMA')]:
        previous=[r for r in read_csv(directory/'metrics_combined.csv') if r['Condition']==historical]
        if {r['Category'] for r in previous}!={c for c,n in current if n==candidate}:
            raise ValueError('Historical endpoint category coverage differs.')
        for row in previous:
            for f in FIELDS:
                delta=float(current[(row['Category'],candidate)][f])-float(row[f])
                checks.append(dict(Category=row['Category'],Condition=candidate,Metric=f,Delta=delta,Passed=abs(delta)<=.001))
    write_csv(output/'endpoint_metric_checks.csv',list(checks[0]),checks)
    if not all(r['Passed'] for r in checks):
        raise ValueError('All-on/all-off endpoint metrics differ from historical results.')


def summarize(session,jobs,datasets):
    combined=[];deltas=[];effects=[];retention=[]
    pairs=[('global_at_L0','G1_L0','G0_L0'),('global_at_L1','G1_L1','G0_L1'),
           ('local_at_G0','G0_L1','G0_L0'),('local_at_G1','G1_L1','G1_L0')]
    for dataset,shot in jobs:
        values=read_csv(session/dataset/f'{shot}shot_seed10'/'metrics_combined.csv')
        combined.extend(dict(Dataset=dataset,Shot=shot,**r) for r in values)
        means={r['Condition']:r for r in values if r['Category']=='Mean'}
        for name in CONDITIONS:
            for comparator in ('baseline','G1_L1'):
                difference={f:float(means[name][f])-float(means[comparator][f]) for f in FIELDS}
                deltas.append(dict(Dataset=dataset,Shot=shot,Condition=name,Comparator=comparator,**difference,Net=sum(difference.values())))
        for effect,left,right in pairs:
            difference={f:float(means[left][f])-float(means[right][f]) for f in FIELDS}
            effects.append(dict(Dataset=dataset,Shot=shot,Effect=effect,Left=left,Right=right,**difference,Net=sum(difference.values())))
    for filename,keep_mean in [('metrics_summary.csv',True),('metrics_by_category.csv',False)]:
        write_csv(session/filename,['Dataset','Shot','Category','Condition',*FIELDS],
                  [r for r in combined if (r['Category']=='Mean')==keep_mean])
    for comparator,filename in [('baseline','deltas_vs_original.csv'),('G1_L1','deltas_vs_new_D.csv')]:
        group=[r for r in deltas if r['Comparator']==comparator]
        write_csv(session/filename,list(group[0]),group)
    write_csv(session/'factor_effects.csv',list(effects[0]),effects)
    for shot in sorted({s for _,s in jobs}):
        for name in CONDITIONS:
            group=[r for r in deltas if r['Shot']==shot and r['Condition']==name and r['Comparator']=='baseline']
            positive=sum(r['Net']>0 for r in group);net=sum(r['Net'] for r in group)
            complete={r['Dataset'] for r in group}==set(datasets) and len(datasets)==6
            retention.append(dict(Shot=shot,Condition=name,PositiveDatasets=positive,CompletedDatasets=len(group),
                Net=net,SixDatasetsComplete=complete,Retain=bool(complete and positive>=3 and net>0)))
    write_csv(session/'retention_by_shot.csv',list(retention[0]),retention)
