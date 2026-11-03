# 旧B权重的双输出验证

本实验只做测试，不重新训练。默认固定使用旧实验九的源域最好权重：
`results/train_rma/pipeline_20261030_210844_733256/best.pth`（epoch15）。
从对应历史B测试会话读取并校验权重SHA256，不按数据集或shot切换权重。

同一次查询编码和RMA推理记录三个条件：

| Condition | 图像分数 | 像素异常图 |
|---|---|---|
| baseline | 原A最终分数 | 原A最终图 |
| rma | 旧B最终分数 | 旧B最终图 |
| dual_output | 原A最终分数 | 旧B最终图 |

旧B最终图包含VA/TA/PQA/RMA四分支融合、ALIGN融合及原有高斯平滑。
不是RMA单支路图。双输出不重新计算图像融合，不用B图最大值替换A原始分数的一部分。

## 运行

项目根目录执行：

```powershell
python .\models_test\rma_dual_output.py
```

也支持在models_test目录执行 `python .\rma_dual_output.py`，所有路径基于项目根目录。
默认依次Visa、BTAD、MPDD、SDD、DTD、part_glass，每个数据集依次1/2/4-shot，seed=10，共18轮。
每轮独立子进程，等待退出释放进程内存后启动下一轮；失败或中断停止，并在run_status.json保留当前数据集、shot和已完成列表。
不承诺系统文件缓存或分页文件立即缩小。

默认保存历史固定少量样本的可视化，不额外保存npy矩阵。
若需要保存这些选定样本的原始及平滑后矩阵：

```powershell
python .\models_test\rma_dual_output.py --save_maps
```

显式小范围执行示例（仅在用户需要时使用）：`--datasets Visa --shots 1`。
`--reference_run`可显式选择另一份完整B测试会话，但那是另一项候选；默认不使用本次参数对齐的新B。

## 结果

新会话保存在 `results/rma_dual_output/old_B_six_datasets_seed10/`，重复运行创建编号目录，保留历史结果。

- metrics_summary.csv：三条件、六数据集、各shot、六指标宏平均。
- metrics_by_category.csv：三条件的各产品类别六指标。
- deltas_vs_original.csv：B及双输出分别减原A的六项差值和净变化。
- retention_by_shot.csv：分别记录上升数据集数、总上升、总下降和保留判断；不合并shot。
- campaign_config.json、input_audit.json、run_status.json：实际权重、协议、运行范围和完成状态。

每个数据集/shot子目录还有image_scores.csv（三条件逐图分数）、metrics_combined.csv、参考及查询身份记录、日志和可视化。
保留原A的逐图分数、六指标和固定样本矩阵核对；old_B_metric_checks.csv和old_B_score_checks.csv核对历史B的完整六指标及逐图分数。
历史B未保存全测试集异常图，因此无法逐像素核对全量历史B；本次双输出图与本次B图在每张图上检查完全相同。

双输出图像输入等于A、像素输入等于B，因此复用本次A的图像评价、本次B的像素评价生成双输出六指标。
不是直接读取历史指标充当新预测；所有测试图仍实际推理，历史指标只用于一致性核对。
这样无需第三份全测试类别像素缓存，也无需重复执行相同评价。

## 修改范围与验证

新入口与辅助模块在models_test，新回归测试在tests/test_rma_dual_output.py。
已有test_rma.py仅增加默认不启用的输出路由和缓冲区工厂参数，可视化按实际条件列表展示；默认两条件计算和训练流程不变。
原train.py、test.py、模型核心和权重不修改。回退时移除test_rma.inference的两个可选参数及对应调用，恢复原固定条件的可视化列表即可。
开发阶段只做AST、内存编译和Ruff静态检查；未自动启动训练、推理或pytest。
本实验运行前没有新的实测提升结论；之前双输出收益是已有指标的离线组合推导。

## 实测结果：2026-11-03核对

运行目录为results/rma_dual_output/old_B_six_datasets_seed10，六数据集1/2/4-shot、seed10的18组全部完成。
使用同一份旧B源域best权重，未重新训练。原A和旧B完整六指标与历史记录差值均为0。

| shot | 双输出上升数据集 | 总上升 | 总下降 | 总净和 | 旧B完整输出净和 | 双输出符合保留条件 |
|---|---|---|---|---|---|---|
| 1 | 4/6 | 6.191975 | 3.340760 | +2.851214 | -0.581606 | 是 |
| 2 | 4/6 | 6.495811 | 4.597160 | +1.898651 | -2.688094 | 是 |
| 4 | 4/6 | 5.107849 | 4.062217 | +1.045633 | -6.012843 | 是 |

各数据集净和为六项百分点差值之和，表中再跨六数据集相加，不是平均AUROC。
双输出图像三指标与原A相同；像素输出在BTAD、MPDD、DTD、part_glass净提升，VisA、SDD净下降，三个shot一致。
六数据集等权平均的像素指标变化（P-AUROC / P-AP / P-F1max，百分点）：

- 1-shot：+0.105574 / +0.143111 / +0.226517。
- 2-shot：+0.053020 / +0.054057 / +0.209365。
- 4-shot：+0.051153 / -0.006491 / +0.129610。

收益幅度小且随shot减弱。尤其SDD的像素AP/F1退化，不能声称所有数据集/指标全面提升。
这确认此前离线组合的实现与同条件复现正确，可保留为实验候选；尚未验证其他随机种子。

实际离线核对：1476项历史A/B指标、15408条逐图双输出分数、420对保存的原始/平滑后B与双输出矩阵全部通过；
独立重算315项类别图像AUROC，最大误差约0.000008583个百分点。
全测试集像素指标未从全量矩阵独立重算；已检查运行时相等回执及选定样本矩阵，不以个案替代全测试集指标。
详细分析、六项完整值和核对证据保存于results/rma_dual_output/old_B_six_datasets_seed10/analysis_20261103/。
