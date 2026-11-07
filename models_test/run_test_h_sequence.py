r"""先完整运行test_h.py默认H，再运行test_h.py --mode E。

Root: python .\models_test\run_test_h_sequence.py
每阶段六数据集各1/2/4-shot；失败停止，--status只读进度，不恢复任务。
"""
import argparse
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from context_alignment import DATASETS
from fusion_ablation_utils import new_directory,project_path,read_json
from rma_score_coordination import write_json
from rma_utils import cleanup,digest,run_child
from run_rma_followup_sequence import progress


def require_complete(mode,report):
    """Never enter E after an incomplete/failed or wrong-mode H campaign."""
    child=report.get('child_status',{})
    if child.get('complete') is not True:
        raise ValueError('Child exited without completing the full mode.')
    session=Path(report['session']);config=read_json(session/'campaign_config.json');opts=config['options']
    expected={f'{dataset}/{shot}' for dataset in DATASETS for shot in (1,2,4)}
    if (opts['mode']!=mode or opts['datasets']!=list(DATASETS) or opts['shots']!=[1,2,4]
            or opts['verify_history'] is not True or set(child.get('completed_jobs',[]))!=expected
            or set(child.get('job_results',{}))!=expected):
        raise ValueError('Wrong mode/settings or incomplete18-condition coverage.')
    for folder in child['job_results'].values():
        folder=Path(folder);state=read_json(folder/'run_status.json')
        if not all(state.get(key) is True for key in ('complete','frozen_integrity_passed','history_parity_passed')):
            raise ValueError('A dataset/shot has not passed frozen/history verification.')
        if not state.get('output_sha256'):
            raise ValueError('Missing output integrity evidence.')
        for name,value in state['output_sha256'].items():
            if digest(folder/name)!=value:
                raise ValueError(f'Completed output changed: {folder/name}')


def describe(session,state,report):
    child=report.get('child_status',{})
    text=[f"当前模式：{state.get('current_mode')}",f"已完成模式：{state.get('completed_modes',[])}",f'总记录：{session}']
    if report.get('session'):text.append(f"当前test_h结果目录：{report['session']}")
    if child.get('current'):text.append(f"当前数据集/shot：{child['current']}")
    if child.get('current_output'):
        text.append(f"当前子任务目录：{child['current_output']}")
        status_path=Path(child['current_output'])/'run_status.json'
        if status_path.is_file():
            try:
                one=read_json(status_path);text.append(f"最近记录的查询数量：{one.get('processed_queries',0)}")
            except (OSError,ValueError):text.append('子任务状态暂不可读。')
    if child.get('completed_jobs') is not None:text.append(f"已完成条件：{child['completed_jobs']}")
    if report.get('note'):text.append(report['note'])
    if state.get('error'):text.append(f"停止原因：{state['error']}")
    return '\n'.join(text)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--status',help='Read progress only; no start/resume.')
    args=parser.parse_args(argv)
    if args.status:
        session=project_path(args.status,ROOT);state=read_json(session/'run_status.json')
        if state.get('complete'):
            print(f'H和E均已完成。总记录：{session}');return
        report=progress(session/f"{state['current_mode']}_progress.json",'test_h')
        print('以下仅为已保存进度，不启动或恢复推理。\n'+describe(session,state,report));return
    script=ROOT/'models_test/test_h.py'
    if not script.is_file():raise FileNotFoundError(script)
    session=new_directory(ROOT/'results'/Path(__file__).stem,'H_then_E_seed10')
    state=dict(complete=False,current_mode='H',completed_modes=[],stage_results={})
    write_json(session/'run_status.json',state)
    write_json(session/'execution_plan.json',dict(modes=['H','E'],datasets=list(DATASETS),shots=[1,2,4],seed=10,
        cleanup='Wait for every dataset/shot child and each full-mode process to exit before continuing.',automatic_resume=False))
    try:
        for mode in ('H','E'):
            state['current_mode']=mode;write_json(session/'run_status.json',state)
            receipt=session/f'{mode}_progress.json'
            command=[sys.executable,'-u',str(script)]
            if mode=='E':command.extend(['--mode','E'])
            command.extend(['--_progress_file',str(receipt)])
            write_json(session/f'{mode}_command.json',dict(command=command))
            print(f'[{1 if mode=="H" else 2}/2] 开始{mode}模式；总记录：{session}',flush=True)
            try:run_child(command)
            finally:cleanup()
            report=progress(receipt,'test_h');require_complete(mode,report)
            state['completed_modes'].append(mode);state['stage_results'][mode]=report['session']
            write_json(session/'run_status.json',state)
        state.update(complete=True,current_mode=None);write_json(session/'run_status.json',state)
        print(f'H和E均已完成，共36个数据集/shot条件。总记录：{session}',flush=True)
    except BaseException as error:
        state.update(complete=False,error=f'{type(error).__name__}: {error}')
        try:report=progress(session/f"{state['current_mode']}_progress.json",'test_h')
        except Exception as inspection_error:report=dict(note=f'无法读取子任务进度：{inspection_error}')
        state['stop_progress']=report;write_json(session/'run_status.json',state)
        summary='运行停止，后续模式未启动。\n'+describe(session,state,report)
        (session/'stop_summary.txt').write_text(summary+'\n',encoding='utf-8');print(summary,file=sys.stderr,flush=True)
        raise


if __name__=='__main__':main()
