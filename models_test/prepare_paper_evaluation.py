r"""外部评估准备（不是新算法实验，不启动SubspaceAD/DictAS）。

python .\models_test\prepare_paper_evaluation.py
导出六数据集18条件的同参考/同查询清单、六指标协议、原A/H/K双输出及待填外部表。
后续外部结果可用 --external_csv 收集；导入数值不等于通过同协议复现验证。
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from fusion_ablation_utils import new_directory, path_key, project_path, read_csv, write_csv
from pqa_local_control_utils import DATASETS, DUAL_RUN, REFERENCE_RUN, target_inputs
from rma_control_io import write_json
from rma_utils import FIELDS, digest

RESULT_ROOT = ROOT/'results'/Path(__file__).stem
EXTERNAL_METHODS = ('SubspaceAD', 'DictAS')
CSV_FIELDS = ['Method','Dataset','Shot','Seed',*FIELDS,'ProtocolStatus','Source']


def sample_key(category, relative_path):
    return hashlib.sha256((category+'|'+relative_path.replace('\\','/')).encode('utf-8')).hexdigest()[:24]


def validate_external(rows):
    expected = {(m,d,s) for m in EXTERNAL_METHODS for d in DATASETS for s in (1,2,4)}
    seen = set()
    cleaned = []
    for row in rows:
        key = row['Method'], row['Dataset'], int(row['Shot'])
        if key not in expected or key in seen or int(row['Seed']) != 10:
            raise ValueError('Unknown/duplicate method, dataset, shot or wrong seed.')
        seen.add(key)
        filled = [str(row.get(f,'')).strip() for f in FIELDS]
        if not any(filled): continue
        if not all(filled): raise ValueError('Each imported condition must contain all six metrics.')
        values = {f:float(row[f]) for f in FIELDS}
        if not all(math.isfinite(v) and 0 <= v <= 100 for v in values.values()):
            raise ValueError('External metrics must be finite on a 0-100 scale.')
        cleaned.append(dict(Method=key[0],Dataset=key[1],Shot=key[2],Seed=10,**values,
            ProtocolStatus='unverified_import',Source=row.get('Source','')))
    return cleaned


def complete_dataset_means(rows):
    """Never turn a partially reproduced dataset set into a six-dataset mean."""
    result = []
    for method,shot in sorted({(r['Method'],int(r['Shot'])) for r in rows}):
        group = [r for r in rows if r['Method']==method and int(r['Shot'])==shot]
        if len(group)!=6 or {r['Dataset'] for r in group}!=set(DATASETS): continue
        result.append(dict(Method=method,Shot=shot,Seed=10,
            **{f:sum(float(r[f]) for r in group)/6 for f in FIELDS}))
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--external_csv',help='Optional completed template; values remain explicitly unverified.')
    a = p.parse_args(argv)
    imported = validate_external(read_csv(project_path(a.external_csv,ROOT))) if a.external_csv else []
    # All preflight work is read-only; create a new result folder only after it passes.
    records = []
    for dataset in DATASETS:
        for shot in (1,2,4):
            records.append((dataset,shot,target_inputs(REFERENCE_RUN,DUAL_RUN,dataset,shot)))
    output = new_directory(RESULT_ROOT,'six_datasets_seed10')
    own_rows, template, inputs_hashes, jobs = [], [], {}, []
    for dataset,shot,(inputs,history,table,_,original) in records:
        child = output/dataset/f'{shot}shot_seed10';child.mkdir(parents=True)
        root = Path(inputs['config']['test_data_path'])
        meta = json.loads(Path(inputs['meta']).read_text(encoding='utf-8'))
        metadata = {path_key(root/r['img_path']):r for c in inputs['classes'] for r in meta['test'][c]}
        references = [dict(Category=r['cls_name'],ImagePath=str(root/r['img_path']),RelativePath=r['img_path'],
                           Label=0,ReferenceOrder=i) for i,r in enumerate(inputs['references'])]
        queries = []
        for i,row in enumerate(inputs['manifest']):
            item = metadata[path_key(row['Image Path'])]
            label = int(item['anomaly'])
            queries.append(dict(QueryIndex=i,SampleKey=sample_key(row['Category'],item['img_path']),Category=row['Category'],
                ImagePath=row['Image Path'],RelativePath=item['img_path'],Label=label,
                MaskPath=str(root/item['mask_path']) if label else '',MaskRelativePath=item['mask_path'] if label else ''))
        write_csv(child/'references.csv',list(references[0]),references)
        write_csv(child/'queries.csv',list(queries[0]),queries)
        write_json(child/'job.json',dict(dataset=dataset,shot=shot,seed=10,root=str(root),categories=inputs['classes'],
            reference_file='references.csv',query_file='queries.csv',reference_identity_source=str(history),
            target_labels_may_tune=False,normal_reference_augmentation='Record method-native policy; use only these normal references.',
            encoder='Method-native frozen encoder; record pretrained weight identity and input preprocessing.',
            evaluator_output_size=[inputs['config']['image_size']]*2,
            evaluator_mask='Existing Dataset/get_transform mask processing, anomaly mask >0.5; normal mask zero.',
            sample_map_contract='One finite anomaly map per SampleKey; higher means more anomalous; preserve query order.',
            image_score_contract='One finite image score per SampleKey; retain native method definition and document it.',
            target_split_policy='Use the accepted current meta.json split unchanged, including SDD.'))
        for condition,label in (('baseline','original_A'),('H','H'),('K_dual','K_dual')):
            source = original if condition=='baseline' else table
            selected = [r for r in source if r['Category']=='Mean' and r['Condition']==condition]
            if len(selected)!=1:raise ValueError('Missing/duplicate internal mean metrics.')
            source_file = (inputs['history'] if condition=='baseline' else history)/'metrics_combined.csv'
            own_rows.append(dict(Method=label,Dataset=dataset,Shot=shot,Seed=10,
                **{f:float(selected[0][f]) for f in FIELDS},ProtocolStatus='historical_checked',Source=str(source_file)))
        for method in EXTERNAL_METHODS:
            template.append(dict(Method=method,Dataset=dataset,Shot=shot,Seed=10,**{f:'' for f in FIELDS},
                                 ProtocolStatus='not_run',Source=''))
        for path in (inputs['meta'],history/'metrics_combined.csv',history/'references.csv',history/'sample_manifest.csv',
                     inputs['history']/'metrics_combined.csv',inputs['history']/'run_status.json'):
            inputs_hashes[str(path)] = digest(path)
        jobs.append(dict(Dataset=dataset,Shot=shot,Seed=10,Queries=len(queries),References=len(references),Directory=str(child)))
    write_csv(output/'own_metrics.csv',CSV_FIELDS,own_rows)
    means = complete_dataset_means(own_rows)
    write_csv(output/'own_six_dataset_mean.csv',list(means[0]),means)
    write_csv(output/'external_metrics_template.csv',CSV_FIELDS,template)
    write_csv(output/'jobs.csv',list(jobs[0]),jobs)
    evaluator = [ROOT/'models_test/branch_diagnostics_utils.py']
    for name in ('tools','dataset','metrics'):
        evaluator.extend(sorted((ROOT/name).rglob('*.py')))
    write_json(output/'protocol.json',dict(kind='external_evaluation_preparation',algorithm_experiment=False,
        datasets=list(DATASETS),shots=[1,2,4],seed=10,metrics=list(FIELDS),metric_scale='0-100',
        aggregation='Per category first, dataset category-macro mean; six-dataset arithmetic mean separately per shot.',
        runtime_false_positive_and_false_negative_metrics_included=False,
        subspacead_dictas_reproduction_started=False,external_models='Official DINO/PCA SubspaceAD and official DictAS, not CLIP-PCA or context_alignment substitutes.',
        input_sha256=inputs_hashes,evaluator_code_sha256={str(p):digest(p) for p in evaluator},
        imported_metrics_are_protocol_verified=False))
    if imported:
        write_csv(output/'external_metrics_unverified.csv',CSV_FIELDS,imported)
        means = complete_dataset_means(imported)
        if means: write_csv(output/'external_six_dataset_mean_unverified.csv',list(means[0]),means)
        gaps = []
        lookup = {(r['Dataset'],int(r['Shot'])):r for r in own_rows if r['Method']=='K_dual'}
        for row in imported:
            own = lookup[row['Dataset'],row['Shot']]
            gaps.append(dict(Method=row['Method'],Dataset=row['Dataset'],Shot=row['Shot'],
                **{f:float(own[f])-float(row[f]) for f in FIELDS},Comparison='K_dual minus external; protocol unverified'))
        write_csv(output/'gaps_unverified.csv',list(gaps[0]),gaps)
    (output/'README.md').write_text(
        '# 外部评估准备\n\n未运行SubspaceAD/DictAS，不代表已经复现。按每个job.json的参考图和查询清单对接官方入口。\n'
        '不得用额外目标图像拟合/调参；若方法必须依赖额外数据，应单独披露，不能标成相同协议。\n'
        '记录官方版本、编码器权重、参考增强、输入预处理、异常图插值与图像评分定义。\n'
        '最终通过现有Evaluator计算六指标。只导入CSV不能证明同协议；模板空白表示尚未运行，不是0分。\n'
        '本准备入口不修改SDD划分、不训练、不下载、不调用外部仓库。\n',encoding='utf-8')
    write_json(output/'run_status.json',dict(complete=True,preparation_only=True,jobs=18,external_methods_run=[]))
    print('外部评估准备完成；未启动外部复现：',output,flush=True)


if __name__=='__main__':main()
