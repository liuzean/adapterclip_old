# 新D两种双输出独立验证

固定比例12.5%的双输出，在项目根目录执行：

```powershell
python .\models_test\rma_fixed_dual_output.py
```

β4双输出，单独执行：

```powershell
python .\models_test\rma_beta4_dual_output.py
```

也支持在models_test目录直接执行对应文件。两个入口独立运行，不会自动连续启动另一项。
每个默认Visa→BTAD→MPDD→SDD→DTD→part_glass，每数据集1→2→4-shot、seed10，共18轮。
每轮一个子进程，等待退出并清理后继续，失败/中断停止并显示dataset/shot及已完成数量。

## 严格的输出定义

两项实验都使用同一新D源域选出的第14轮best.pth，CLIP和全部适配器冻结，不训练、不保存新权重。

| 输出条件 | 最终图像分数 | 最终像素图 |
|---|---|---|
| baseline | 原A原最终分数 | 原A原最终图 |
| G0_L1 | 新D G0_L1原完整最终分数 | RMA固定25%＋ALIGN＋平滑 |
| G1_L1 | 完整新D原最终分数 | 与G0_L1相同的图 |
| local_candidate | 12.5%或β4自身的单输出最终分数，仅作对照 | 本候选融合＋ALIGN＋原sigma4平滑 |
| dual_output | **完整引用G0_L1最终分数** | **完整引用local_candidate最终图** |

G0_L1的完整最终图像分数，本身已包含原25%图最大值参与调和融合的结果。
dual_output不再调用max/融合/平滑，不把三分支native全局项与12.5%或β4图最大值重新结合。
单输出候选的自身图像分数仍计算并核对历史，以确认算法路径一致，但该分数不用于dual_output。
双输出张量直接引用，缓存中最多保留原A/G0_L1/候选三套像素数组，不复制dual或完整新D的重复图。

固定12.5%：每个位置α=.125，不计算分散度；β4：α(x)=.25×exp(-4d(x))，沿用实验十七的Top8分散度定义。
两项都不按类别/数据集手动开关，不改变参考图、模型权重、ALIGN、平滑、其他分支或评价协议。

## 历史和离线组合核对

固定比例默认历史：`results/rma_fixed_fusion/all_fractions_seed10`；β4默认历史：
`results/rma_reliability_beta_ablation/six_datasets_seed10`。

默认离线组合表：
`results/run_rma_followup_sequence/three_tasks_seed10/analysis_20261106/derived_dual_metrics_by_category.csv`。
其中固定比例取dual_alpha12.5，β4取dual_beta4，全部类别及Mean、三个shot都核对。

1. 启动前审计原A、G0_L1、完整新D、局部候选的权重、固定参考顺序/重复次数、查询身份及协议。
2. 确认离线组合每项I指标取历史G0_L1，每项P指标取历史局部候选。
3. 本轮同次推理重新得到G0_L1和候选；逐图核对历史最终分数。
4. 每个样本检查dual图像分数精确等于当前G0_L1，dual像素张量精确等于当前候选。
5. 所有类别及Mean的双输出六指标核对离线表；候选六指标核对历史；原A/G0_L1/完整新D核对原历史。
6. 指标容差沿用.001百分点，历史分数容差1e-6绝对/1e-5相对；双输出与本次供体指标要求完全相同。
7. 权重、实现及历史/离线文件有变化则拒绝继续，失败不标完成。

历史结果未保存全测试集像素矩阵，因此不会声称已把本轮所有像素与历史逐值重比。
“逐像素相同”指本次dual与本次候选的恒等引用；与历史核对的是所有类别指标和可追溯输入/公式。
本次输出图像、像素完全相同的指标输入只评价一次，恒等检查通过后复用指标，不增加重复缓存。

历史目录或分析文件在其他编号目录时，可明确指定：

```powershell
python .\models_test\rma_fixed_dual_output.py --local_history .\results\rma_fixed_fusion\all_fractions_seed10 --offline_metrics .\results\run_rma_followup_sequence\three_tasks_seed10\analysis_20261106\derived_dual_metrics_by_category.csv
```

β4入口支持相同参数，但local_history应指向β历史。脚本不会自行猜最新目录。

## 输出和结论边界

固定比例：`E:\projects\FSAD\AdaptCLIP_old\results\rma_fixed_dual_output\six_datasets_seed10\`。
β4：`E:\projects\FSAD\AdaptCLIP_old\results\rma_beta4_dual_output\six_datasets_seed10\`。
重复自动编号，不覆盖旧结果。

- metrics_summary.csv、metrics_by_category.csv：五条件完整六指标。
- deltas_vs_G0_L1.csv、deltas_vs_original_A.csv、deltas_vs_new_D.csv：dual相对三个对照的全部差值。
- retention_by_shot.csv、retention_overall.json：主对照G0_L1；每shot至少3/6数据集净正且总净和为正，至少两shot通过才整体可保留。
- 各dataset/shot：逐图分数、参考/查询清单、dual_historical_score_checks.csv、offline_dual_metric_checks.csv、
  原端点回归检查和run_status.json；关键一致性不通过会停止。
- evaluation_plan/locked_config.json：预先确定的双输出验证计划，source_selected=false，不重新选参。

默认只保存固定少量样本可视化；加 `--save_maps` 保存其原始/平滑四类对照和双输出图矩阵。
实际包含五条件，其中若图相同仍保存名称对应矩阵，便于人工核对；不保存全测试集矩阵。
配对选择来自此前目标集探索，应如实披露，不能表述为仅由源域选出的泛化方案。
离线推导达标不等于此处独立实现已通过；运行后以实际回归结果为准。

代码复用rma_reliability.target_worker，新增可选输出路由、缓冲器和最终核对接口，旧入口默认行为保留。
未修改原train.py/test.py/adaptcliplib。开发仅静态检查，不自动推理或运行pytest；实际效果/一致性尚未运行确认。
