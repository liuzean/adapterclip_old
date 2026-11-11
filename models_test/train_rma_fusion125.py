r"""实验L训练：从新D相同起点重训四适配器，仅融合监督RMA比例改12.5%。

python .\models_test\train_rma_fusion125.py
CLIP冻结；VA/TA/PQA/RMA从头共同训练15轮；不加载H旧适配器、不创建CRA。
各分支独立损失不变。源域选优仍用原新D25%完整输出，严格同口径；另保留epoch14。
只新增本文件及独立辅助模块，不修改/替换任何正在使用的共享代码。
"""
import argparse
import logging
import os
import random
import shutil
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from fusion_ablation_utils import new_directory,project_path,read_csv,read_json,write_csv
from rma_control_io import write_json,write_progress
from rma_fusion125_utils import joint_forward125,settings,training_plan,verify_plan
from rma_utils import cleanup,digest


def train(args,output,plan):
    """New-D sampling/update order is preserved; episodes must match before use."""
    import copy
    import torch
    from adaptcliplib import BinaryDiceLoss,FocalLoss
    from dataset import Dataset
    from fusion_ablation_utils import RecordedPromptDataset
    from rma_joint_utils import FullBatchSampler,adapter_states,require_joint_gradients
    from rma_scratch_common import initialize_fresh,protocol,source_validation,training_losses
    from rma_utils import atomic_checkpoint,check_frozen,curves,freeze_base,source_split
    from test import build_prompt_memory
    from tools import get_transform,setup_seed
    from tqdm import tqdm
    progress=dict(epoch=0,batch=0,phase='initializing')
    def note(phase,**details):
        progress.update(phase=phase,**details);write_progress(output/'training_progress.json',progress)
    try:
        setup_seed(10);config=protocol(args)
        model,adapters,rma=initialize_fresh(args,config)
        frozen=freeze_base([None,model]);modules=(*adapters,rma)
        parameters=[p for m in modules for p in m.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(parameters,lr=args.learning_rate,betas=tuple(args.betas),weight_decay=args.weight_decay)
        check_frozen(frozen,optimizer)
        transform,target_transform=get_transform(image_size=config.image_size)
        data=Dataset(root=str(args.train_data_path),transform=transform,target_transform=target_transform,
                     dataset_name='mvtec',k_shots=0,save_dir=str(output),mode='test',seed=10)
        meta=read_json(args.train_data_path/'meta.json')
        if any(int(r['anomaly'])!=0 for pool in meta['train'].values() for r in pool):raise ValueError('Non-normal source reference.')
        write_json(output/'training_reference_pool.json',meta['train'])
        split=source_split(args.train_data_path,10,args.validation_fraction)
        write_json(output/'source_split.json',split)
        val_data=copy.copy(data);val_data.data_all,val_data.length=split['validation_queries'],len(split['validation_queries'])
        val_loader=torch.utils.data.DataLoader(val_data,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers)
        data.data_all,data.length=split['train_queries'],len(split['train_queries'])
        supports=RecordedPromptDataset(str(args.train_data_path),split['validation_supports'],split['classes'],transform,target_transform)
        support_loader=torch.utils.data.DataLoader(supports,batch_size=args.batch_size,shuffle=False)
        memory=build_prompt_memory(model,support_loader,'cuda',data.obj_list,data.view_list,config.features_list,20)
        memory=({k:v.cpu() for k,v in memory[0].items()},{k:[v.cpu() for v in vs] for k,vs in memory[1].items()})
        sampler=FullBatchSampler(torch.utils.data.RandomSampler(data),args.batch_size)
        loader=torch.utils.data.DataLoader(data,batch_sampler=sampler,num_workers=args.num_workers)
        write_json(output/'training_queries.json',data.data_all)
        for name in ('training_reference_pool.json','source_split.json','training_queries.json'):
            if read_json(output/name)!=read_json(args.matched_new_d_run/name):raise ValueError('New-D source input mismatch: '+name)
        expected=read_csv(args.matched_new_d_run/'training_episodes.csv')
        if len(expected)!=15*len(data):raise ValueError('Historical episode budget differs.')
        metadata=dict(format='adaptclip-rma-scratch-v1',flow='new',initialization='CLIP pretrained; all four adapters fresh',
            training_scheme='L_scratch_new_fusion125',fusion125_control=True,train_local_fraction=.125,
            model_spec=rma.spec,base_protocol=vars(config),base_sha256=digest(args.baseline_checkpoint),
            pretrained_sha256=digest(args.pretrained_model),source_meta_sha256=digest(args.train_data_path/'meta.json'),
            train_options={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},seed=10,
            source_validation_used=True,selection_rule=plan['selection'],primary_same_epoch=14,
            matched_new_d_run=str(args.matched_new_d_run),frozen_modules=['CLIP'],
            trainable_modules=['VA','TA','PQA','RMA'],trainable_parameters=sum(p.numel() for p in parameters),
            train_query_count=len(data),validation_query_count=len(val_data))
        write_json(output/'model_manifest.json',metadata)
        focal,dice,rng=FocalLoss(),BinaryDiceLoss(),random.Random(10)
        history,steps,episodes,validation=[],[],[],[];best_score=float('-inf');best_epoch=None
        for epoch in range(1,16):
            note('training',epoch=epoch,batch=0)
            for module in modules:module.train()
            optimizer.zero_grad(set_to_none=True);epoch_rows=[]
            for step,items in enumerate(tqdm(loader,desc=f'L fusion125 {epoch}/15')):
                note('training',epoch=epoch,batch=step+1)
                batch=len(items['img_path']);shot=args.train_reference_shots[len(steps)%3];refs=[]
                for path,category in zip(items['img_path'],items['cls_name']):
                    chosen=rng.sample(split['training_support_pool'][category],shot);refs.extend(chosen)
                    row=dict(epoch=epoch,step=step+1,query=path,category=category,shot=shot,references=' | '.join(r['img_path'] for r in chosen))
                    if len(episodes)>=len(expected) or {k:str(v) for k,v in row.items()}!=expected[len(episodes)]:
                        raise ValueError('Query/reference episode differs from new D; stopped before forward.')
                    episodes.append(row)
                support_data=RecordedPromptDataset(str(args.train_data_path),refs,split['classes'],transform,target_transform)
                prompt_images=torch.stack([support_data[i]['img'] for i in range(len(support_data))]).cuda()
                with torch.no_grad():
                    features,patches=model.encode_image(items['img'].cuda(),config.features_list,DPAM_layer=20)
                    pf,pp=model.encode_image(prompt_images,config.features_list,DPAM_layer=20)
                    pf,pp=pf.reshape(batch,shot,-1),[p.reshape(batch,shot,*p.shape[1:]) for p in pp]
                globals_,pixels,raw=joint_forward125(model,adapters,rma,features,patches,pf,pp)
                masks=(items['img_mask'][:,0]>.5).float().cuda()
                loss,parts=training_losses('new',globals_,pixels,raw,items['anomaly'].cuda(),masks,focal,dice,args.fusion_weight)
                if not torch.isfinite(loss):raise ValueError('Nonfinite L training loss.')
                size=min(args.accumulation,len(loader)-(step//args.accumulation)*args.accumulation)
                (loss/size).backward()
                if (step+1)%args.accumulation==0 or step+1==len(loader):
                    require_joint_gradients(adapters,rma)
                    torch.nn.utils.clip_grad_norm_(parameters,args.gradient_clip,error_if_nonfinite=True)
                    optimizer.step();optimizer.zero_grad(set_to_none=True)
                row=dict(epoch=epoch,step=step+1,shot=shot,batch_size=batch,learning_rate=args.learning_rate,
                         **{k:float(v.detach()) for k,v in parts.items()})
                steps.append(row);epoch_rows.append(row)
                del loss,parts,globals_,pixels,raw,prompt_images,features,patches,pf,pp
            check_frozen(frozen,optimizer)
            score,values=source_validation(config,model,adapters,rma,val_loader,memory,output,epoch,progress=note)
            validation.extend(values)
            summary=dict(epoch=epoch,**{k:sum(r[k] for r in epoch_rows)/len(epoch_rows)
                         for k in epoch_rows[0] if k not in ('epoch','step','shot','batch_size')},validation_score=score)
            history.append(summary)
            improved=score>best_score
            if improved:best_score,best_epoch=score,epoch
            payload=dict(**metadata,**adapter_states(adapters),rma=rma.state_dict(),epoch=epoch,
                         validation_score=score,best_epoch=best_epoch,best_score=best_score)
            note('saving_checkpoint')
            atomic_checkpoint(output/f'epoch_{epoch}.pth',payload)
            if improved:
                # Exact selected-epoch copy: avoid treating different torch ZIP
                # filenames as evidence of different model tensors.
                temporary=output/'selected_copy.tmp';shutil.copyfile(output/f'epoch_{epoch}.pth',temporary)
                os.replace(temporary,output/'best.pth')
            atomic_checkpoint(output/'last.pth',dict(**payload,optimizer=optimizer.state_dict()))
            for name,values in (('training_history.csv',history),('training_steps.csv',steps),
                                ('training_episodes.csv',episodes),('validation_metrics.csv',validation)):
                write_csv(output/name,list(values[0]),values)
            curves(output,history)
            logging.getLogger('L').info('epoch=%s total=%.6f source=%.9f best_epoch=%s',epoch,summary['total_loss'],score,best_epoch)
            write_json(output/'run_status.json',dict(complete=False,epoch=epoch,best_epoch=best_epoch,best_score=best_score))
            cleanup()
        if len(episodes)!=len(expected):raise ValueError('Incomplete matched episodes.')
        verify_plan(plan);note('completed')
        files=[output/n for n in ('model_manifest.json','training_history.csv','training_steps.csv','training_episodes.csv',
                                  'validation_metrics.csv','source_split.json','training_queries.json','training_reference_pool.json','best.pth')]
        files += [output/f'epoch_{e}.pth' for e in range(1,16)]
        return dict(complete=True,epochs=15,best_epoch=best_epoch,best_score=best_score,
                    same_epoch_checkpoint=str(output/'epoch_14.pth'),selected_checkpoint=str(output/'best.pth'),
                    matched_episodes_passed=True,frozen_integrity_passed=True,four_adapter_gradient_checks_passed=True,
                    output_sha256={p.name:digest(p) for p in files})
    except BaseException as error:
        write_json(output/'run_status.json',dict(complete=False,error=str(error) or 'KeyboardInterrupt',**progress));raise
    finally:cleanup()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--matched_new_d_run',default='results/train_rma_scratch_new/mvtec_seed10')
    p.add_argument('--_progress_file',help=argparse.SUPPRESS)
    a=p.parse_args(argv);matched=project_path(a.matched_new_d_run,ROOT)
    args=settings(matched);plan=training_plan(matched);verify_plan(plan)
    output=new_directory(ROOT/'results'/Path(__file__).stem,'mvtec_seed10')
    write_json(output/'campaign_config.json',plan)
    write_json(output/'experiment_config.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    write_json(output/'run_status.json',dict(complete=False,epoch=0,batch=0))
    if a._progress_file:write_json(project_path(a._progress_file,ROOT),dict(session=str(output),phase='training'))
    logger=logging.getLogger('L');logger.setLevel(logging.INFO)
    handler=logging.FileHandler(output/'train.log',encoding='utf-8');logger.addHandler(handler)
    try:
        state=train(args,output,plan);write_json(output/'run_status.json',state)
        print(f"L训练完成：{output}\n同轮对照：epoch14；源域最优：epoch{state['best_epoch']}",flush=True)
    finally:handler.close();logger.removeHandler(handler)


if __name__=='__main__':main()
