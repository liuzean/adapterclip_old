# 实验K双输出真实推理一致性验证

这是K的小型补充验证，不是新训练实验。固定H使用的新D第14轮权重、已经完成训练的CRA第9轮权重、seed10及历史正常参考；不更新任何参数，不重新选权重或比例。

在项目根目录运行：

```powershell
python .\models_test\cra_dual_output.py
```

也支持从models_test目录直接运行`python .\cra_dual_output.py`。

两条输出分别为：

- 图像检测：K_CRA已经验证的完整最终分数，保留`0.75SE_K+0.25G0_K`，其中SE_K仍使用其原25%局部图参与评分。
- 像素定位：关闭CRA的H完整最终定位图，即12.5%局部混合→ALIGN→sigma4平滑。

直接引用这两条本次推理的最终张量；不根据H定位图的最大值重新计算K图像分数，不把K的12.5%定位图当成H图，不使用目标标签选择输出或改规则。规则来自已有目标结果探索，因此不能描述成新的独立盲测。

默认Visa→BTAD→MPDD→SDD→DTD→part_glass，每个数据集1→2→4-shot，共18条件。每批仅一次CLIP编码和一次RMA前向，再计算CRA关闭/开启的H/K分支。只存两组独立的单类别像素缓冲；双输出像素指标复用相同H张量的评价值，图像指标复用相同K分数的评价值，避免存第三份大矩阵。

## 固定输入及核对

- H底座：`results/train_rma_scratch_new/mvtec_seed10/best.pth`，固定已验证SHA256和epoch14。
- CRA：`results/train_cra/mvtec_seed10/best.pth`，固定本次第9轮SHA256 `7a0ea7beb98afc610ffa5390309d5ce3b1edfe0df37ab0cdd9e51ce750c8e150`。
- 历史H输入：`results/rma_score_coordination_inference/experiment_H_inference_seed10`。
- 本次K历史：`results/test_cra/K_seed10`。
- 离线组合表：`results/test_cra/K_seed10/analysis_20261110/offline_dual_metrics.csv`。

启动前核对已有H/K完成状态、参数/代码/文件指纹、查询和参考身份以及离线表是否来自指定供体。每轮实际推理后核对：

1. H/K逐图分数与已完成K实验一致，绝对容差1e-6。
2. H/K各类别及Mean的完整六指标与历史一致，容差0.001个百分点。
3. 本次双输出图像分数逐图等于K，定位图逐像素等于H；对应供体指标必须精确相同。
4. 双输出数据集Mean六指标与离线组合一致，容差0.001个百分点；各类别同时按历史供体核对。
5. CLIP、旧适配器、RMA及CRA参数/运行缓冲均冻结，共享特征未被原地修改。

历史没有保存全部像素数组，不能声称逐像素比过全部历史图。这里逐像素检查的是本次实际计算的H与双输出。

## 输出与中断

首次输出：`results/cra_dual_output/K_dual_seed10/`，重复运行自动编号。不会覆盖H、K或权重。主要文件：

- `metrics_summary.csv`、`metrics_by_category.csv`：H、K、双输出及历史原A完整六指标。
- `deltas_vs_H.csv`、`deltas_vs_K_CRA.csv`、`deltas_vs_baseline.csv`：数据集Mean差值。
- `retention_by_shot.csv`、`verification_summary.json`：三个shot分开判定，至少两个shot通过才考虑保留。
- 每轮`image_scores.csv`、`score_checks.csv`、`history_metric_checks.csv`、`donor_metric_checks.csv`、`offline_checks.csv`、配置、参考/查询清单及完成指纹。

默认不生成新图片或npy矩阵文件。每dataset/shot独立子进程，等待退出并清理后继续；类别评价完成也清理可释放缓存。失败记录当前数据集/shot并停止后续条件。

```powershell
python .\models_test\cra_dual_output.py --status .\results\cra_dual_output\K_dual_seed10
python .\models_test\cra_dual_output.py --resume_from .\results\cra_dual_output\K_dual_seed10
```

status只读，不启动模型。resume恢复原设置，跳过已经完整核对且输出未改动的条件；未完成条件在新尝试目录从头推理。使用终端实际输出的会话路径。

## 开发验证范围

该入口已做静态语法/接口和18条件既有输入、离线供体数据核对；没有自动运行新模型推理或pytest。`tests/test_cra_dual_output.py`是独立CPU回归契约。现有K真实运行不能替代本入口的一致性验证，需以实际运行结果为准。

实现只新增cra_dual_output.py、cra_dual_output_utils.py及说明/测试，不修改已经完成K实验的模型、训练、测试代码及指纹，原始test.py保留。
