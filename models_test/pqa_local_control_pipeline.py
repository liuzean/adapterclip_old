r"""实验M串行入口：无CRA训练 -> 冻结CRA训练 -> 四条件完整测试。

python .\models_test\pqa_local_control_pipeline.py
GPU任务串行运行；训练臂之间、测试每dataset/shot之间，等待子进程退出释放内存。
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from fusion_ablation_utils import new_directory, project_path, read_json
from pqa_local_control_utils import (DATASETS, DUAL_RUN, REFERENCE_RUN, checked_training,
                                     target_inputs, training_plan, verify_plan)
from rma_control_io import write_json
from rma_utils import cleanup, run_child

RESULT_ROOT = ROOT/'results'/Path(__file__).stem
STAGES = ('plain_train', 'cra_train', 'test')


def show_progress(session, state):
    stage = state.get('current')
    print('阶段：', stage, '；已完成：', state.get('completed', []))
    receipt = session/f'{stage}_receipt.json'
    if not receipt.is_file(): return
    child = Path(read_json(receipt)['session'])
    if stage in STAGES[:2]:
        path = child/'training_progress.json'
        progress = read_json(path) if path.is_file() else {}
        print(f'训练：epoch={progress.get("epoch")}，batch={progress.get("batch")}')
    elif stage == 'test':
        progress = read_json(child/'run_status.json')
        print('测试数据集/shot：',progress.get('current'))
    print('子任务目录：',child)


def verified_stage(session, stage):
    child = Path(read_json(session/f'{stage}_receipt.json')['session']).resolve()
    expected = ROOT/'results'/('test_pqa_local_control' if stage == 'test' else 'train_pqa_local_control')
    child.relative_to(expected.resolve())
    if stage == 'test':
        from test_pqa_local_control import verified_complete
        status = read_json(child/'run_status.json'); plan = read_json(child/'campaign_config.json'); verify_plan(plan)
        if status.get('complete') is not True or set(status['job_results']) != set(plan['jobs']):
            raise ValueError('Target stage incomplete.')
        for key, folder in status['job_results'].items():
            Path(folder).resolve().relative_to(child); verified_complete(folder,plan['jobs'][key]['queries'])
    else:
        checked_training(child, 'plain' if stage == 'plain_train' else 'cra')
    return child


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--status'); p.add_argument('--resume_from'); a = p.parse_args(argv)
    if a.status and a.resume_from: p.error('Use status or resume separately.')
    if a.status:
        session=project_path(a.status,ROOT);session.relative_to(RESULT_ROOT.resolve())
        print('只显示最后保存进度，不启动任务，也不判断进程是否仍在运行。')
        show_progress(session,read_json(session/'run_status.json'));return
    if a.resume_from:
        session=project_path(a.resume_from,ROOT);session.relative_to(RESULT_ROOT.resolve())
        plan,state=read_json(session/'campaign_config.json'),read_json(session/'run_status.json');verify_plan(plan)
        if set(state['completed'])-set(STAGES):raise ValueError('Unknown completed stage.')
    else:
        _,_,_,plan=training_plan()
        # Check every target ledger before spending either training budget.
        for dataset in DATASETS:
            for shot in (1,2,4):target_inputs(REFERENCE_RUN,DUAL_RUN,dataset,shot)
        session=new_directory(RESULT_ROOT,'M_seed10')
        write_json(session/'campaign_config.json',plan)
        state=dict(complete=False,current=None,completed=[]);write_json(session/'run_status.json',state)
    try:
        for stage in STAGES:
            state.update(complete=False,current=stage);state.pop('error',None)
            write_json(session/'run_status.json',state)
            if stage in state['completed']:
                verified_stage(session,stage);print('核对并复用完成阶段：',stage,flush=True);continue
            receipt=session/f'{stage}_receipt.json'
            previous=Path(read_json(receipt)['session']) if receipt.is_file() else None
            if previous and (previous/'run_status.json').is_file() and read_json(previous/'run_status.json').get('complete'):
                verified_stage(session,stage)
            else:
                entry='test_pqa_local_control.py' if stage=='test' else 'train_pqa_local_control.py'
                command=[sys.executable,'-u',str(ROOT/'models_test'/entry)]
                if previous:command+=['--resume_from',str(previous)]
                elif stage=='test':
                    command+=['--plain_run',str(verified_stage(session,'plain_train')),
                              '--cra_run',str(verified_stage(session,'cra_train'))]
                else:command+=['--arm','plain' if stage=='plain_train' else 'cra']
                command+=['--_progress_file',str(receipt)]
                print('启动：',stage,flush=True);run_child(command)
                verified_stage(session,stage);cleanup()
            state['completed'].append(stage);write_json(session/'run_status.json',state)
        verify_plan(plan);state.update(complete=True,current=None);write_json(session/'run_status.json',state)
        print('实验M全部完成：',session,flush=True)
    except BaseException as error:
        state.update(complete=False,error=str(error) or 'KeyboardInterrupt');write_json(session/'run_status.json',state)
        print('运行停止，后续任务不启动。',flush=True);show_progress(session,state);raise


if __name__=='__main__':main()
