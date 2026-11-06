# 三项RMA后续实验顺序运行

在项目根目录运行：

```powershell
python .\models_test\run_rma_followup_sequence.py
```

也可在models_test内运行。严格顺序执行：

1. `rma_fixed_fusion.py --evaluate_all_fractions`：复用0/25/37.5%，补12.5%的六数据集1/2/4-shot，汇总四比例。
2. `rma_local_diagnostics.py`：MVTec源域四图局部误差归因，1/2/4-shot。
3. `rma_reliability_beta_ablation.py`：六数据集1/2/4-shot，同次推理得到β4/16/64。

全部沿用各入口默认seed10、同一新D第14轮权重、参考图及协议，不训练。
第一阶段必须完成全部18轮，第二阶段全部3个shot，第三阶段18轮且全部3个β，才标为本阶段成功。
等待上一个阶段进程退出后再启动下一个，沿用子脚本的每dataset/shot独立进程和清理逻辑。
父调度器不导入torch；每阶段退出后调用清理。不会同时运行三项，不自动跳过或复用已完成的整项任务。

总调度记录位于：

`E:\projects\FSAD\AdaptCLIP_old\results\run_rma_followup_sequence\three_tasks_seed10\`

重复运行新建编号目录，不覆盖。三项实验的实际结果仍分别位于它们自己的results目录。
父记录包含执行命令、完成阶段、通过子任务回执得到的准确结果目录，不猜测“最新目录”。
失败或Ctrl+C停止后续阶段，终端与stop_summary.txt显示阶段、已完成任务和最近dataset/shot。
断电无法即时输出，重启后可只读检查保存的进度：

```powershell
python .\models_test\run_rma_followup_sequence.py --status .\results\run_rma_followup_sequence\three_tasks_seed10
```

`--status`只读，不启动、不续跑、不保证最后保存状态等于当前进程状态。
普通运行则从第一项重新执行；本入口没有自动断点续跑。

子脚本仅增加隐藏目录回执参数，不改它们的计算、默认shot、保存或筛选流程。
静态检查通过不代表三项模型实验已经执行；开发不自动启动推理或pytest。
