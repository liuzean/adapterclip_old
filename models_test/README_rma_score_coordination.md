# 实验H 图像评分贡献协调

完成状态更新 2026-11-07：本入口源域选lambda及目标离线评价已完成；随后rma_score_coordination_inference完成18条件真实推理，精简test_h的H/E又完成各18条件回归。以下仍描述本入口自身的离线流程，不把后续真实推理计为新算法收益。当前主线、全部代价与选参来源见README_H_mainline.md。

公式：`SH = (1-lambda) * SE + lambda * G0`。

- SE：实验E已有完整最终图像分数，即G0_L1完整评分，包含25%局部图最大值参与评分的影响。
- G0：新D的VA/TA/PQA原生全局概率均值，不是原A分数，也不包含RMA全局项。
- 像素定位：完整沿用实验E的12.5%局部融合＋ALIGN＋平滑最终图，指标直接复用E的完整像素评价。
- 权重：新D同一第14轮best.pth，CLIP及所有适配器冻结，不重新训练。

## 默认运行

从项目根目录：

```powershell
python .\models_test\rma_score_coordination.py
```

也支持进入models_test后运行`python .\rma_score_coordination.py`。路径均基于项目根目录解析。

一次启动依次完成：

1. 核对权重、MVTec源域验证划分及正常参考、实验E与历史G0逐图身份/标签/配置。
2. MVTec已有345张验证查询，依次1/2/4-shot，每个shot一个GPU子进程，复用原60张正常参考的首S张，保持原batch=2。每次查询编码一次，同次推理计算SE与G0；不存全测试像素缓存或矩阵。
3. 每个shot在同一批预测上计算固定lambda网格`0,0.25,0.5,0.75,1`的图像三指标。指标沿用原项目Evaluator的PR曲线梯形面积AP。
4. 源域类别宏平均，再对图像三指标和三个shot等权平均；选分数最高的一个lambda。1e-9内视为数值并列，优先较小lambda。lambda=0精确恢复E，应如实记录为没有新增改变。
5. 写入`locked_config.json`后，按Visa→BTAD→MPDD→SDD→DTD→part_glass、各1/2/4-shot，依次完成18个CPU离线评价子进程。每条件退出并清理后再继续。

目标集已有完整的SE与G0逐图分数，所以不再运行目标图像编码或RMA、不重新计算像素指标，不生成新的图或矩阵。H是离线候选验证；若后续保留，仍应另做真实推理入口的一致性验证。

目标标签只用于评价锁定后的一个lambda，不按数据集挑系数、不运行目标集系数网格、不将目标结果传给源域选择函数。源域验证集此前参与epoch14权重选择，属于已有验证集复用，不称为独立盲测。H方向也来自已观察的目标域诊断，应如实披露。

## 保存结果

```text
E:\projects\FSAD\AdaptCLIP_old\results\rma_score_coordination\experiment_H_seed10\
```

重复运行自动分配`_run02`等目录，不覆盖原结果。

- `source/<shot>shot_seed10/`：源域SE/G0逐图分数、历史端点检查、有效参考、源域候选图像指标。
- `source_selection.csv`：五个系数的源域选择分数。
- `locked_config.json`：唯一选定系数、选择规则、源域记录指纹，明确目标标签没有参与选择。
- `targets/<dataset>/<shot>shot_seed10/`：锁定候选H与A/E/G0_L1/完整新D的完整六指标、离线H分数、E图像指标重算校验、查询和参考清单。
- `metrics_summary.csv`：所有数据集、shot、类别及Mean的六指标；像素指标来自已有E或各对照的历史记录。
- `dataset_deltas.csv`：H相对E、原A、G0_L1、完整新D的六指标差值及净和。
- `retention_by_shot.csv`、`retention_overall.json`：以E为直接对照的保留判断。每shot至少3/6数据集六指标净正且总净和正，整体至少两个shot通过。完整六集及三个shot未跑齐时不作整体判定。
- `campaign_config.json`：实际选项、权重/代码/输入指纹和验证范围，不记录Git提交哈希。
- `run_status.json`：阶段、当前条件、完成列表、每个条件实际结果目录。

H像素三指标始终与E完全相同。lambda=0时全部六指标也精确沿用E，不把CPU/GPU数值舍入差写成提升。每个目标条件用原Evaluator在CPU重算E的图像三指标，以0.001个百分点容差核对历史；源域SE逐图以1e-6容差核对历史G0_L1端点。

## 进度与续跑

只读查看，不启动实验：

```powershell
python .\models_test\rma_score_coordination.py --status .\results\rma_score_coordination\experiment_H_seed10
```

中断后恢复：

```powershell
python .\models_test\rma_score_coordination.py --resume_from .\results\rma_score_coordination\experiment_H_seed10
```

请替换成实际目录。续跑恢复原选项，只复用已完成且代码、权重和输出指纹一致的条件；未完成条件在新的尝试目录重跑，旧文件保留。失败会输出当前阶段、数据集/shot、已完成条件及目录，后续停止。元数据原子替换，降低突然中断损坏进度文件的风险。

`--datasets`和`--shots`可显式限制目标离线评价范围；源域始终使用全部1/2/4-shot，避免系数随目标筛选改变。默认维持seed=10，不提供目标域调参入口。

## 文件与开发验证

`rma_score_coordination.py`负责流程和推理/离线评价，`rma_score_coordination_utils.py`负责评分组合、原评价器适配和源域选择；复用现有权重加载、数据集、参考记忆、分支推理和评分公式，不修改稳定代码或既有实验实现。

契约测试在`tests/test_rma_score_coordination.py`，覆盖精确端点、宏平均、AP口径、源域并列选择、异常输入和只读进度。开发阶段按项目约定只做静态检查与已保存文件审查，不自动运行pytest或实验；实际未验证项以交付说明为准。
