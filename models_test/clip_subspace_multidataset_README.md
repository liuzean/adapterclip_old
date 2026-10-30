# CLIP PCA：五个新数据集的 1/2/4-shot 实验

这是独立实验调度入口，直接调用已有 `clip_subspace.run_inference`。原 VisA 入口仍为 `clip_subspace.py`，其默认 1/2/4-shot 行为不变。

## 运行

在已有 Adaptclip 环境、项目根目录运行：

```powershell
cd E:\projects\FSAD\AdaptCLIP_old
python .\models_test\clip_subspace_multidataset.py
```

顺序为“数据集 → shot”：先完成 BTAD 的 1→2→4-shot，再完成 MPDD 的 1→2→4-shot，然后依次 SDD、DTD、part_glass，共 15 组，全部 seed=10。不重复 VisA。每组为完整测试集，shot 限制的是每类别正常参考图数量。

也支持从 `models_test` 目录运行 `python .\clip_subspace_multidataset.py`。所有相对路径以项目根目录为准。

默认输出每类正常/异常各一张固定对比图，不保存新 NPY。需要固定样本矩阵时：

```powershell
python .\models_test\clip_subspace_multidataset.py --save_maps
```

只读取输入、检查历史配置和列出计划，不启动模型的命令：

```powershell
python .\models_test\clip_subspace_multidataset.py --dry_run
```

## 输入及公平对照

- 完整精度 Baseline：`results/branch_diagnostics/multi_dataset_seed10/<数据集>/<shot>shot_seed10/`。
- 双输出历史控制：`results/dual_output/multi_dataset_seed10/<数据集>/<shot>shot_seed10/`。双输出只是对照，不参与 PCA 融合。
- 原始 AdaptCLIP 日志/参考名单：`results/<数据集>/<shot>/`。每组读取同一 shot 的历史结果，检查参考图与顺序一致、原始一位小数指标与完整精度 FINAL 在舍入范围内一致。
- 直接继承历史权重、特征层 6/12/18/24、518 图像尺寸、batch_size=8、sigma=4、全部适配器开关及参考编码批次，不重新随机抽参考。
- PCA 保持现有配置：EV=0.99；四层 patch 特征等权平均后 L2 归一化；正常参考 CPU float64 拟合；查询 float32 平方残差/4；原平滑及最大值评分。没有新增目标图、增强、训练、PQA 空间加权或分支融合。
- 每组仍输出 baseline、dual_output、clip_pca，保留全部六项指标。控制指标/逐图分数/可用历史矩阵不一致立即停止。

如果历史会话名不同，可以明确指定 `--source_root` 和 `--control_root`，路径应指向包含五个数据集目录的同一轮结果根目录。默认值对应已完成的 `multi_dataset_seed10`，不会自动选最新会话。

## 输出

```text
results/clip_subspace_multidataset/five_datasets_seed10/
├── campaign_config.json
├── input_audit.json
├── execution_plan.json
├── run_status.json
├── metrics_summary.csv
├── metrics_by_category.csv
├── metrics_vs_baseline.csv
├── BTAD/
│   ├── 1shot_seed10/
│   ├── 2shot_seed10/
│   └── 4shot_seed10/
├── MPDD/                 # 同样包含 1/2/4-shot 子目录
├── SDD/
├── DTD/
└── part_glass/
```

重复执行创建 `_run02` 等新会话，不覆盖历史结果，不自动续跑。先前 1-shot 调度器的 `five_datasets_1shot_seed10` 目录原样保留；本入口从头运行全部 15 组，不跳过以前做过的组。子目录包含六指标、逐图分数、历史一致性检查、PCA 拟合统计及状态文件、固定样本图、日志和配置。汇总表的 Shot 列区分 1/2/4-shot。只有显式 `--save_maps` 才保存固定样本原始/平滑矩阵，不保存全测试集矩阵。

每个数据集的每个 shot 使用独立进程。正常结束执行 GC/CUDA 缓存回收，父进程等待子进程退出后才运行下一组；这不保证 Windows 页文件立即缩小。失败停止后续组，未完成的表不能当成最终结果。

## 结果判断与验证状态

完成后分别与已有 VisA 1/2/4-shot 对应结果合并分析：同一 shot 的六指标按百分点差值等权求每个数据集净变化；至少三个数据集净提升且六数据集总净变化为正，才达到约定保留门槛。三种 shot 分别判断，不合并成 18 个数据集。该入口按固定计划完成 15 组，不因中途结果好坏跳过数据集或 shot。

本次仅做静态语法、依赖/调用关系、路径和异常流程检查，没有运行该入口、`--dry_run`、项目测试或模型推理。是否完成 15 组推理、实际资源占用和运行时控制一致性，等待用户运行确认。
