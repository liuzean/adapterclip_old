r"""实验L：先从头训练12.5%融合监督，再测试第14轮与源域best。

python .\models_test\rma_fusion125_pipeline.py
六数据集各1/2/4-shot，定位12.5%为主、25%为辅助；每轮子进程退出清理。
--status只读位置；--resume_from仅复用完整训练并续跑未完成测试。
不启动/停止CRA任务，不修改其代码或结果。
"""
import argparse
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from fusion_ablation_utils import new_directory,project_path,read_json
from rma_control_io import write_json
from rma_fusion125_utils import checked_training,target_plan,training_plan,verify_plan
from rma_utils import cleanup,run_child

RESULT_ROOT=ROOT/'results'/Path(__file__).stem


def safe_read(path):
    try:return read_json(path) if path.is_file() else {}
    except (OSError,ValueError):return {}


def describe(session,state):
    if state.get('complete'):return '训练及测试均已完成。'
    stage=state.get('current_stage');receipt=safe_read(session/f'{stage}_progress.json')
    folder=receipt.get('session') or state.get('stage_results',{}).get(stage)
    if not folder:return f'当前阶段：{stage}；子任务尚未记录位置。'
    folder=Path(folder)
    if stage=='train':
        progress=safe_read(folder/'training_progress.json')
        return f"训练：epoch={progress.get('epoch','尚无记录')}，batch={progress.get('batch','尚无记录')}"
    current=safe_read(folder/'run_status.json').get('current')
    return f'测试：数据集/shot={current or "尚无记录"}'


def completed_child(receipt,stem):
    folder=project_path(read_json(receipt)['session'],ROOT)
    folder.relative_to((ROOT/'results'/stem).resolve())
    if read_json(folder/'run_status.json').get('complete') is not True:raise ValueError(stem+' did not finish.')
    return folder


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--status');p.add_argument('--resume_from')
    a=p.parse_args(argv)
    if a.status and a.resume_from:p.error('Use status or resume separately.')
    if a.status:
        session=project_path(a.status,ROOT);session.relative_to(RESULT_ROOT.resolve())
        print('只读最后保存进度，不启动模型，也不判断进程是否仍在运行。')
        print(describe(session,read_json(session/'run_status.json')));return
    if a.resume_from:
        session=project_path(a.resume_from,ROOT);session.relative_to(RESULT_ROOT.resolve())
        plan=read_json(session/'execution_plan.json');verify_plan(plan)
        state=read_json(session/'run_status.json')
        if state.get('complete'):print('实验L已经完成：',session);return
        train_receipt=session/'train_progress.json'
        folder=state['stage_results'].get('train') or safe_read(train_receipt).get('session')
        if not folder or safe_read(Path(folder)/'run_status.json').get('complete') is not True:
            raise ValueError('本序列训练未完整完成；没有实现epoch续训。请新启动序列从头训练，旧结果保留。')
        checked_training(folder);state['stage_results']['train']=str(folder)
        if 'train' not in state['completed_stages']:state['completed_stages'].append('train')
    else:
        reference=ROOT/'results/rma_score_coordination_inference/experiment_H_inference_seed10'
        plan=training_plan(ROOT/'results/train_rma_scratch_new/mvtec_seed10')
        target=target_plan(reference);plan['input_sha256'].update(target['input_sha256'])
        plan.update(reference_run=str(reference),jobs=target['jobs']);verify_plan(plan)
        session=new_directory(RESULT_ROOT,'L_seed10');write_json(session/'execution_plan.json',plan)
        state=dict(complete=False,current_stage='train',completed_stages=[],stage_results={})
        write_json(session/'run_status.json',state)
    try:
        if 'train' not in state['completed_stages']:
            state.update(current_stage='train');write_json(session/'run_status.json',state)
            print('[1/2] L：冻结CLIP、从头训练四适配器15轮，仅融合监督改12.5%',flush=True)
            run_child([sys.executable,'-u',str(ROOT/'models_test/train_rma_fusion125.py'),
                       '--_progress_file',str(session/'train_progress.json')]);cleanup()
            folder=completed_child(session/'train_progress.json','train_rma_fusion125');checked_training(folder)
            state['stage_results']['train']=str(folder);state['completed_stages'].append('train')
            write_json(session/'run_status.json',state)
        training=Path(state['stage_results']['train']);_,train_state,weights=checked_training(training)
        write_json(session/'selected_checkpoints.json',dict(primary=weights['epoch14'],secondary=weights['best']))
        verify_plan(plan);state.update(current_stage='test');write_json(session/'run_status.json',state)
        print(f"[2/2] L：第14轮主对照＋源域best第{train_state['best_epoch']}轮，六集18条件",flush=True)
        receipt=session/'test_progress.json';previous=safe_read(receipt).get('session') if a.resume_from else None
        command=[sys.executable,'-u',str(ROOT/'models_test/test_rma_fusion125.py')]
        command += ['--resume_from',previous] if previous else ['--training_run',str(training),'--_progress_file',str(receipt)]
        run_child(command);cleanup();testing=completed_child(receipt,'test_rma_fusion125')
        state['stage_results']['test']=str(testing);state['completed_stages'].append('test')
        state.update(complete=True,current_stage=None);state.pop('error',None)
        write_json(session/'run_status.json',state);print('实验L训练和测试完成：',session,flush=True)
    except BaseException as error:
        state.update(complete=False,error=str(error) or 'KeyboardInterrupt')
        message='运行停止，后续不启动。\n'+describe(session,state)+'\n总记录：'+str(session)
        write_json(session/'run_status.json',state);(session/'stop_summary.txt').write_text(message,encoding='utf-8')
        print(message,file=sys.stderr,flush=True);raise
    finally:cleanup()


if __name__=='__main__':main()
