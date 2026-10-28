# 双输出：Baseline 图像评分 + no_TA 像素定位

本轮只验证一个固定的输出分流，不训练、不移植模块、不调整权重：

| 条件 | 用于像素指标的异常图 | 用于图像指标的分数 |
|---|---|---|
| baseline | 原平滑 Baseline 图 | 原 Baseline 最终分数 |
| no_TA | 原平滑 no_TA 图 | 原全局均值与 no_TA 图最大值调和融合 |
| dual_output | 原平滑 no_TA 图 | **原 Baseline 最终分数** |

设 H 为原调和融合、G 为原 Gaussian 平滑：Baseline 图是 `G(H(mean(VA,TA,PQA), ALIGN))`，no_TA 图是 `G(H(mean(VA,PQA), ALIGN))`。双输出的图像评分始终是 `H(mean(VA_global,TA_global,PQA_global), max(Baseline图))`，**不能将这里的最大值换成 no_TA 图最大值**。

TA 仍参与 Baseline 图和原生全局评分，所有 Adapter 保留。PQA 四层均值、ALIGN、匹配、残差、平滑、输入尺度及六项原始评价指标均沿用已有实现。

## 从项目根目录运行

先手动验证 1-shot：

```powershell
cd E:\projects\FSAD\AdaptCLIP_old
python .\models_test\dual_output.py --k_shots 1
```

依次运行完整的 1/2/4-shot：

```powershell
python .\models_test\dual_output.py
```

默认 VisA、seed=10；每个 shot 独立子进程，上一组退出后再开始下一组。类别结束释放类别缓冲，每批只前向一次模型及各 Adapter。类别缓冲只保存两套唯一像素图；精确像素指标计算仍会有内存峰值，Windows 页面文件不保证立即缩小。

关闭新分流，候选的两个输出都恢复 Baseline（仍保留 no_TA 对照）：

```powershell
python .\models_test\dual_output.py --k_shots 1 --disable_dual_output
```

可选保存矩阵或减少输出：

```powershell
python .\models_test\dual_output.py --k_shots 1 --save_maps
python .\models_test\dual_output.py --k_shots 1 --no-save_visualizations
```

也支持在 `models_test/` 内运行，相对参数路径均以项目根目录解析。

## 配置与依赖

默认 `--source_root results/branch_diagnostics`：读取原 1/2/4-shot 配置及参考图名单，沿用实际权重、518 输入、batch size=8、sigma=4、层 6/12/18/24、三个 Adapter 和 PQContext 开关；不重新随机抽参考图。

默认 `--ablation_root results/branch_ablation/visa_seed10`：读取上一轮完整结果，核对相同配置、参考图、测试顺序及标签，然后逐图、逐类比较。本入口需要这两套已完成结果。若上一轮保存在编号目录，明确指定：

```powershell
python .\models_test\dual_output.py --ablation_root results/branch_ablation/visa_seed10_run02
```

直接调用已有 `branch_ablation.infer_batch`，用 `nullcontext()` 关闭内部观察；其 no_VA 小规模融合仍会计算，但立即丢弃，不缓存或评价。没有再次前向，也没有 PQA hook、四层诊断统计或额外层图。原初始化、参考编码和汇总复用 `fusion_ablation`；原平滑、绘图、Evaluator 不变。没有复制模型加载、数据集或整份原 `test.py`，没有新增第三方依赖。

所有代码依赖现有三个实验的辅助模块；上传时保留这些已有依赖。

## 输出结果

默认新建 `E:\projects\FSAD\AdaptCLIP_old\results\dual_output\visa_seed10\`。已有同名结果则创建 `visa_seed10_run02/` 等新目录，不覆盖旧结果。

| 文件 | 用途 |
|---|---|
| 父目录 metrics_summary.csv | 三个 shot、三种条件的宏平均六指标，全部完成时 9 行 |
| 父目录 metrics_by_category.csv | 12 类完整指标，全部完成时 108 行 |
| 父目录 metrics_vs_baseline.csv | 相对当次 Baseline 的六指标变化，全部完成时 117 行（含 Mean） |
| 父目录 input_audit.json / run_status.json | 输入摘要、固定输出策略及整体完成状态 |
| 每 shot 的 metrics_combined.csv | 三条件、12 类及 Mean，39 行 |
| baseline_reference_comparison.csv | 与原 branch_diagnostics 比较 Baseline 六指标，78 项 |
| ablation_reference_comparison.csv | 与上一轮比较 Baseline/no_TA 六指标，156 项 |
| image_scores.csv | 全 2162 图的三个原生全局分数、两图最大值、三条件最终分数和输出来源 |
| image_score_parity.csv | 全 2162 图、每图八个原始分数组件与上一轮比较；记录最大误差及失败字段 |
| routing_checks.csv | 每批候选图与所选原图、候选分数与 Baseline 分数的精确相等检查 |
| baseline_map_checks.csv | 所选图片 Baseline raw/evaluated 与最早保存的 FINAL 矩阵比较；默认 48 项 |
| visualization_samples.csv | 每类前 1 张正常、前 1 张异常，共 24 张固定图片；不按预测选样 |
| experiment_config.json / reference_samples.txt / source_sample_manifest.csv | 实际参数、原参考图、原全量测试清单 |
| run.log / run_status.json | 日志和当前 shot 状态 |

默认每 shot 24 张四面板图：输入、GT、Baseline、no_TA，统一 [0,1] 色标。候选使用其中的 no_TA 图，关闭分流时使用 Baseline 图，不重复绘制一张完全相同的图。`--vis_per_type 3` 可恢复每类正常/异常各 3 张，共 72 张。

`--save_maps` 仅保存所选样本的 baseline/no_TA 两套 raw/evaluated `.npy`，不重复保存候选别名。默认 24 张、518×518 float32 时约 98 MiB/shot，三组约 295 MiB。不保存全测试集特征或矩阵。关闭可视化时仍核对固定图片的原矩阵。

指标单位为百分点、类别等权；AP 沿用原仓库 AUPR 梯形 PR 面积。候选指标直接复用同一次运行中对应输出的实际评价结果：三个 I 指标来自 Baseline，三个 P 指标来自 no_TA；关闭时六项都来自 Baseline。这不是读取历史指标冒充新推理。本次也会重新计算两条件指标并与历史结果核对。

## 如何验收

1. 查看父子 `run_status.json` 是否 `completed` 且 `complete=true`。失败、中断或尚未完成时，已有汇总可能只含部分 shot，不可视为完整结果。
2. 检查两份 `*_reference_comparison.csv` 全部 Passed；默认六指标容差 0.001 个百分点。检查逐图分数 Passed，默认 atol=1e-6、rtol=1e-5。
3. 检查路由两项精确相等为 True；原矩阵比较应 passed，缺失时标 unavailable，不能声称完成矩阵核对。默认矩阵容差 atol=1e-6、rtol=1e-5。
4. 确认 dual_output 三个 I 指标与 Baseline 相同、三个 P 指标与 no_TA 相同；关闭后六项恢复 Baseline。任一已启用的历史比较失败会停止后续 shot，并保留日志及差异。
5. 后续评价六项总体收益及损失。既有结果提示 P-AP/P-F1max 收益伴随 P-AUROC 损失；这次分流不解决后者。VisA 已用于探索，本次复现不能替代独立数据或源域验证，不应宣称六项全面提升。

## 开发检查

Codex 仅做静态检查，不执行本入口、`--help`、测试、模型加载或推理。`tests/test_dual_output.py` 是输出分流、Baseline 恢复、两套唯一缓冲、原指标结果归属及失败停止的回归测试，已编写但未运行；它不能回退代码，也不用于判断检测性能。

用户需要单独运行这些逻辑测试时，在项目根目录执行：

```powershell
python -m unittest discover -s tests -p test_dual_output.py
```

完整静态检查记录位于 `results/dual_output/development_review_20261027.md`。原正式文件和已有实验脚本均不修改，也没有执行任何 Git 写操作。
