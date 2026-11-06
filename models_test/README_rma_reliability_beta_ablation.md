# 实验十七 β4 β16 β64 目标数据集消融

项目根目录运行：

```powershell
python .\models_test\rma_reliability_beta_ablation.py
```

也可在models_test目录运行 `python .\rma_reliability_beta_ablation.py`。

默认Visa→BTAD→MPDD→SDD→DTD→part_glass，每数据集1→2→4-shot，seed10。
同一新D第14轮权重、原参考图顺序和重复次数、原查询、六项指标及评价协议，不训练。
本轮明确为预先列出β4/16/64的目标消融，不根据MVTec综合分数淘汰候选，不重新跑源域校准。
不能把目标比较后选择的β表述为仅由源域选出的泛化结果；所有β的下降项目都保留。

## 同次推理中的条件

| 输出 | 定义 |
|---|---|
| baseline | 原A完整最终图和最终图像分数 |
| G0_L1 | β0；RMA局部25%，新D三分支native全局项G0，原ALIGN和平滑 |
| G1_L1 | 完整新D原结果，RMA局部仍25%，native全局项含RMA |
| beta_4 / beta_16 / beta_64 | 各自α(x)=.25×exp(-β×d(x))，只改局部融合，native全局项保持G0 |

沿用实验十七Top8正常参考patch分散度d的计算、四层等权及插值，不换可靠性定义。
各条件都用自己的平滑图最大值计算最终图像分数，不能认为只影响像素指标。
β0无需独立重跑：本轮共享计算G0_L1作直接对照，并逐图分数/逐类别六指标核对历史结果。
原A和完整新D也核对历史，任何不一致就停止。不会按数据集名称选开关或改β。

每dataset/shot独立子进程，共18轮；每轮查询编码、RMA和分散度只计算一次，三个β共用特征。
等价于54个新增候选条件。每类别最多缓存五套唯一像素图（原A、G0_L1、三个β）；
完整新D与G0_L1共享像素指标。相比单β，类别内CPU图缓存增加，但不累计缓存所有类别/数据集。
每类别评价后清理空闲缓存，进程退出后继续下一shot。失败/中断停止，并显示dataset、shot和已完成数量。

## 输出

`E:\projects\FSAD\AdaptCLIP_old\results\rma_reliability_beta_ablation\six_datasets_seed10\`

重复运行自动编号，不覆盖旧结果。

- metrics_summary.csv、metrics_by_category.csv：六条件的完整六指标，G0_L1标Beta0。
- all_beta_deltas.csv及三个deltas_vs_*.csv：各β分别减G0_L1、原A、完整新D。
- all_beta_retention_by_shot.csv、all_beta_retention_overall.json：每β、每对照独立判断。
- beta_4/、beta_16/、beta_64/：分别保存该β的差值和保留判断。
- 各dataset/shot：image_scores.csv包含各条件分数、分散度及每β实际RMA比例的均值/最小/最大值；
  同时保留固定输入身份、回归核对、六指标和运行状态。
- evaluation_plan/locked_config.json：预先声明的目标测试计划，source_selected=false。
  沿用共享执行器的calibration字段名，但这不是源域选出β4的记录；beta4只是批量执行的首个候选，三个β全部报告。

主要对照固定为G0_L1。每个shot至少3/6数据集六指标百分点净和严格为正且总净和为正，
同一个β至少两个shot通过才整体可保留；不能把β4的通过shot与β16的通过shot拼起来。
原A和完整新D的对照另外完整报告，不事后换主对照。

默认固定少量历史样本可视化；若需其异常图、分散度及三个β各自比例矩阵：

```powershell
python .\models_test\rma_reliability_beta_ablation.py --save_maps
```

只保存选定样本矩阵，不保存全测试集矩阵。旧单β入口和固定比例入口的默认计算保持原样；
共享rma_reliability.py只增加显式多候选接口，未改原融合公式。共享文件版本变动会触发旧校准复用的哈希保护，
原结果文件不会修改；本入口使用新的预声明目标计划，不复用旧源域最优β作为筛选。
不改原train.py/test.py/adaptcliplib。开发只做静态检查，未自动运行模型或pytest，尚无新增效果结论。
