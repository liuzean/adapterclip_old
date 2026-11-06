# 固定RMA局部融合比例源域对照

在项目根目录运行：

```powershell
python .\models_test\rma_fixed_fusion.py
```

也可在models_test目录运行 `python .\rma_fixed_fusion.py`。

固定seed=10、新D同一份源域选出的第14轮best.pth，不训练或保存新权重。
CLIP和四个适配器全部冻结。目标直接对照是G0_L1，同时报告原A和完整新D。

## 目的和执行顺序

上一轮可靠性规则在MVTec源域选择β=0，目标结果与G0_L1相同。它同时改变了RMA的平均贡献
和各位置的贡献分配，无法分清是降权本身不合适，还是分散度不能代表可靠性。

本轮仅比较四个固定比例：**0%、12.5%、25%、37.5%**。没有空间门控、额外检索、
数据集专用开关或其他新模块。0%使用新D自身三适配器局部图，并不恢复原A权重；
25%精确复用G0_L1计算张量。各位置、类别、数据集和shot都使用同一个选定比例。

1. 先运行MVTec源域验证1/2/4-shot，每shot一个子进程，同次推理计算四个比例。
   复用原345张验证查询、60张正常参考、batch2和参考顺序/shot前缀，不加入1380张训练查询。
2. 按类别宏平均后的六项指标与三个shot等权均值选择统一比例；1e-9百分点内视作数值持平，
   优先保留现有25%。保存 `calibration/locked_config.json`，目标标签不参与选择。
3. **若仍选25%，程序在源域阶段正常结束，不重复目标18轮。**
   根状态写 `target_evaluation_complete=false` 和跳过原因，不宣称新增目标验证通过。
4. 若选出其他比例，则锁定后运行Visa→BTAD→MPDD→SDD→DTD→part_glass，
   每数据集1→2→4-shot。每条件独立子进程，等待退出/清理再继续；失败停止并显示阶段、dataset、shot。

源域复用了此前选epoch14的验证集，不是独立无偏效果估计。只有完整目标评价才能判断泛化收益。

只做三轮源域校准：

```powershell
python .\models_test\rma_fixed_fusion.py --calibrate_only
```

复用本实验已完成且版本一致的校准（会新建目录，重新完整目标评价；选25%则仍停止）：

```powershell
python .\models_test\rma_fixed_fusion.py --calibration .\results\rma_fixed_fusion\six_datasets_seed10\calibration
```

不能传上一轮rma_reliability的校准目录，两者规则和候选集不同。没有自动从中断shot续跑。

## 计算和结果

`local_mean = (1-alpha)*新D三分支平均图 + alpha*RMA局部概率图`

其后保持原ALIGN调和融合、sigma=4平滑。图像原生全局项始终是新D三分支均值G0，
再与自己的平滑图最大值计算最终图像分数；不加入RMA全局项。
完整新D对照仍使用四分支原生全局项和原固定25%的局部图。

输出到E盘项目 `results/rma_fixed_fusion/six_datasets_seed10/`，重复自动编号、不覆盖旧结果：

- `calibration/source_selection.csv`：四个比例的源域综合成绩。
- `calibration/<shot>shot_seed10/metrics_combined.csv`：四比例及完整新D的逐类别/均值六指标。
- `calibration/locked_config.json`：统一比例、权重/代码/源域审计和选择准则。
- 若进入目标评价：`metrics_summary.csv`、`metrics_by_category.csv`、三个 `deltas_vs_*.csv`、
  `retention_by_shot.csv`、`retention_overall.json`，以及各dataset/shot逐图分数和身份/回归核对。
- `run_status.json`：正常结束、源域停止或失败/中断的具体阶段和已完成条件。

目标条件是baseline、G0_L1、G1_L1、**G0_L1_fixed**。主判定固定相对G0_L1，
每shot要求至少3/6数据集六指标净和严格为正且六数据集总净和为正；至少两个shot通过才整体可保留。
所有六指标和下降项目均保存。若没有运行目标评价，就不生成完整目标保留结论。

复用 `rma_reliability.py` 和 `rma_reliability_calibration.py` 的冻结推理/身份审计/指标计算，
通过显式可选实验接口接入；旧入口默认可靠性公式、源域选β及18轮运行逻辑不变。
`rma_reliability_utils.summarize` 仅增加候选名称参数，旧默认不变。
共享文件版本变化后，旧校准的代码哈希保护会拒绝复用，历史文件不会修改；新实验需新校准。

默认只保存历史选定少量样本对比图；加 `--save_maps` 才保存选定矩阵及固定比例矩阵。
本轮不计算分散度，逐图 `DispersionMean` 为N/A；`RMAFractionMean/Min/Max` 均为选定常数。
固定比例矩阵沿用共享保存器的 `reliability_maps` 子目录，仅有rma_fraction，没有dispersion字段。

开发仅做静态检查，不自动校准、推理或运行pytest。模型效果尚未验证。

## 补齐全部比例的目标消融

```powershell
python .\models_test\rma_fixed_fusion.py --evaluate_all_fractions
```

这是用户要求新增的探索性目标比例消融，与默认“源域选一个比例”的流程分开。
固定原权重/seed/参考及全部六指标，不再由MVTec分数淘汰候选：

- 0%直接复用实验十六G0_L0（新D自身三适配器，不是原A）。
- 25%复用实验十六G0_L1，完整新D的G1_L1也保持原25%的局部图作额外对照。
- 37.5%复用已完成 `results/rma_fixed_fusion/six_datasets_seed10`。
- 仅新增12.5%的六数据集各1/2/4-shot，共18轮。每轮独立进程清理，失败停止并报告dataset/shot。

代码会核对历史权重、协议、参考/查询清单、类别覆盖和共同对照的指标，记录来源与复用指标文件哈希。
历史缺失或不一致就停止，不静默拼接。其他37.5%历史可通过 `--fixed_history` 指定。

输出新建 `results/rma_fixed_fusion/all_fractions_seed10/`（重复编号）。
`all_fraction_metrics.csv`、`all_fraction_metrics_by_category.csv` 为四比例全六指标；
`all_fraction_deltas_vs_25.csv`、`all_fraction_retention_by_shot.csv`、`all_fraction_retention_overall.json`
以25%作统一直接对照，仍要求至少两个shot通过；25%自身不会标为增量收益。
12.5%还单独输出相对原A、G0_L1、完整新D的三份差值表。

`evaluation_plan/locked_config.json` 只是预先声明的12.5%目标测试计划，明确标注
`parameter_origin=predeclared_target_ablation`、`source_selected=false`，不是源域最优校准结果。
复用了共享执行器的calibration字段名，但没有伪造源域选出12.5%的结论。
根据四比例目标结果选参数属于探索性选择，应与源域选参的泛化结果分开报告；
不能把事后选出的比例再称为“仅由源域确定”。不按数据集名称分别挑比例，不省略下降结果。
