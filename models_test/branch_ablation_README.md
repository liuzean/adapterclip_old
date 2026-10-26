# 单独移除 VA/TA 的融合消融与 12 类 PQA 诊断

本实验比较 VisA、seed=10、1/2/4-shot 的 baseline、no_VA、no_TA，并在同一次完整推理中诊断 12 类固定少量图片的四层 PQA 响应。没有训练或移植新模块。

## 从项目根目录运行

在原来的 Adaptclip 环境中：

```powershell
cd E:\projects\FSAD\AdaptCLIP_old
python .\models_test\branch_ablation.py
```

默认依次执行 1、2、4-shot。每个 shot 使用独立子进程；进程退出、CPU/GPU 资源释放后再开始下一组。类别结束也释放缓冲。默认 `num_workers=0`，单 shot 测试集只遍历一次，三种条件共享输出，VA/TA/PQA 均只前向一次/批。

先手动验证一个 shot：

```powershell
python .\models_test\branch_ablation.py --k_shots 1
```

可选参数：

```powershell
# 正常/异常各取每类第一张，固定诊断样本减少为 24 张；完整指标仍用全测试集
python .\models_test\branch_ablation.py --k_shots 1 --diagnostic_per_type 1
# 保存所选诊断图的原始和评价矩阵（默认只保存统计和可视化）
python .\models_test\branch_ablation.py --k_shots 1 --save_maps
# 保留统计，关闭可视化
python .\models_test\branch_ablation.py --k_shots 1 --no-save_visualizations
```

脚本同样兼容从 `models_test/` 内直接运行。所有相对参数路径基于项目根目录解析。

## 配置与参考图

`--source_root` 默认为 `results/branch_diagnostics`，读取各 `visa_{shot}shot_seed10/experiment_config.json`，沿用权重、数据路径、层 6/12/18/24、518 输入、batch size=8、sigma=4、Adapter 与 PQContext 开关。无需重新传模型参数。

直接使用各源目录的 `reference_samples.txt`；不重新抽样。复用原始数据变换、权重加载和 `test.build_prompt_memory`，保持参考图片顺序、重复次数及原编码 batch 组成。

新入口依赖已提交的 `fusion_ablation.py`（仅复用初始化和汇总）、`fusion_ablation_utils.py`、`branch_diagnostics_utils.py` 及原项目模块。旧实验入口不会启动，也没有改写其参数或全局变量。上传代码时需保留这些已有依赖。

## 三种受控条件

以下 mean/H 为原 `adaptcliplib.fusion_fun` 的算术/调和融合：

| 条件 | 平滑前像素图 |
|---|---|
| baseline | H(mean(VA, TA, PQA), ALIGN) |
| no_VA | H(mean(TA, PQA), ALIGN) |
| no_TA | H(mean(VA, PQA), ALIGN) |

移除一个分支后，其余两支的均值分母为 2。所有条件均保留 ALIGN。先融合，再按原 Gaussian 平滑；不做逐图归一化或分数裁剪。

三条件的图像原生全局部分均固定为原 mean(VA_global, TA_global, PQA_global)，随后与各自平滑图最大值调和融合。因此这里的 no_VA/no_TA 指像素融合中的移除，不代表完整移除 Adapter 或跳过它的计算。

六项指标全部保留，使用原 Evaluator，AP 仍为仓库原 AUPR 梯形 PR 积分。类别 Mean 等权，单位为百分数，不可用选中样本统计替代完整指标。

## 固定诊断样本

默认从原始 `sample_manifest.csv` 按记录顺序取每类前 3 张正常、前 3 张异常图片，12 类共 72 张/shot。选择不读取预测或指标。跨 shot 核对类别、类型、路径、Sample Key 和顺序完全一致；数量不足或名单变化会报错，不自动换图。

三组参考图仍分别为原 1/2/4-shot 名单。测试全量样本、标签和顺序与原记录逐项核对。

## 四层与内部观察

直接取原 `PQAdapter.forward` 返回的四个 `local_scores[i][:,1]`，得到 `[B,518,518]` 的 PQA_L6/L12/L18/L24。原 `[B,4,518,518]` 均值得到 PQA_Fused。每张选中图片核对四层均值。

另外，仅对包含选中样本的批次临时注册只读 hook：

| 阶段 | 内容/典型单样本形状 |
|---|---|
| pq_context_before_bn | 原 query + abs(query-aligned_prompt)，sharebn 输入 `[768,37,37]` |
| pq_context_after_bn | 原 sharebn 输出 `[768,37,37]` |
| local_logit_difference | 原 local head 的异常 logit 减正常 logit `[148,148]` |
| local_probability_before_resize | 由原 head logits 计算的诊断 softmax 异常通道 `[148,148]` |

PQA 的四层 local head 在原实现中共享同一个模块实例。观察器按唯一实例注册一次，再用各层独立 BN 的真实调用顺序归属层号，并检查顺序和覆盖；不向四个别名重复挂 hook。正常退出和异常退出均移除已注册 hook。hook 返回 `None`，不修改输入、输出、权重、BN 状态或随机数。

诊断 softmax 只对选中样本的原 logits 额外计算统计，不再次调用 head，也不反馈给模型。统计保留实际 dtype，使用 double 精度只做诊断归约；不改变推理精度。

还会统计原 forward 已返回、已插值的四层 ALIGN 距离图。**没有重新计算 Top-1，也没有将 PQContext 的 BN 输入冒称为独立残差。** 本轮不保存 Top-1 坐标或精确的独立残差；如果后续需要，应另行设计等价提取。

每层异常图保存 raw/evaluated 两套数值统计：原 min/max/mean/std/P50/P95、缺陷/背景统计、正常图 Top1% 均值，新增零值比例、低于 1e-12 的像素比例和峰值是否位于缺陷。1e-12 仅用于诊断描述，不参与检测算法或阈值调参。

## 输出目录

```text
results/branch_ablation/visa_seed10/
  input_audit.json
  run_status.json
  metrics_summary.csv
  metrics_by_category.csv
  metrics_vs_baseline.csv
  visa_1shot_seed10/
  visa_2shot_seed10/
  visa_4shot_seed10/
```

重复运行自动创建 `visa_seed10_run02/` 等新目录。父目录汇总仅包含已成功完成的 shot；应同时查看父子 `run_status.json`，失败或中断后的表格均可能不完整。

每个 shot 的主要文件，默认 72 张诊断图片时预计行数：

| 文件 | 含义 |
|---|---|
| pixel_metrics.csv / image_metrics.csv / metrics_combined.csv | 三条件的 12 类及 Mean 六指标 |
| baseline_reference_comparison.csv | 12 类及 Mean 共 78 项原 baseline 核对 |
| baseline_map_checks.csv | PQA/ALIGN/FINAL 的 raw/evaluated 对照，默认共 432 项；缺失原矩阵标 unavailable |
| pqa_layer_statistics.csv | 每图四层的 raw/evaluated 统计，共 576 行 |
| pqa_layer_summary.csv | 192 个类别/类型/层/阶段分组，各组默认 3 张图；含全零样本数等 |
| pqa_internal_statistics.csv | BN 前后、logit 差、softmax 四阶段，共 1,152 行 |
| alignment_layer_statistics.csv | 四层原始距离图及其平滑图统计，共 576 行 |
| diagnostic_statistics.csv | PQA_Fused、ALIGN、三条件的图统计，共 720 行 |
| image_scores.csv | 全部 2,162 图的原生全局分数、固定全局均值及三条件 map max/最终分数 |
| diagnostic_samples.csv / diagnostic_config.json / diagnostic_coverage.json | 固定名单、取样规则、hook 语义和覆盖检查 |
| experiment_config.json / reference_samples.txt | 实际参数和原参考图名单 |
| source_sample_manifest.csv | 原实验清单；其中旧输出路径相对于 source_result_dir |
| run.log / run_status.json | 运行日志及完成状态 |

默认生成：

- `visualizations/pqa_layers/<类别>/<类型>/`：原输入、GT、四层、PQA_Fused、ALIGN、baseline 九面板。
- `visualizations/branch_conditions/<类别>/<类型>/`：原输入、GT、baseline/no_VA/no_TA 五面板，按两行排列。

显示统一固定 [0,1] 色标；原数组与指标不归一化。`--save_maps` 仅保存所选样本的四层 PQA、PQA_Fused、ALIGN 和三条件图，各含 raw/evaluated；逐层 ALIGN 仅保存统计。默认三 shot、72 图、518×518 float32 的矩阵额外占用约 3.9 GiB，因此默认关闭矩阵保存。

## 一致性与内存

每类结束便核对 baseline，最后核对完整宏平均，默认指标容差 0.001 个百分点；选中旧矩阵默认 atol=1e-6、rtol=1e-5。核对失败会保存差异并停止后续 shot。

模型使用 eval/no_grad，原 Adapter、参考匹配、残差、层均值和评价器均保持原实现。类缓冲只保留三种平滑图和一份 uint8 GT；诊断 hooks 仅保留标量，不保留全数据集特征或四层图。精确像素指标仍有内存峰值，子进程释放不保证 Windows 页面文件立即缩小。

## 开发检查与手动验证

本次只做源文件阅读、AST/编译静态检查、Ruff 符号检查、签名/维度/路径审查；没有运行本入口或测试程序，没有加载权重或进行推理。

`tests/test_branch_ablation.py` 针对样本固定性、共享 head 单次归属、hook 异常清理、输出不变、正确融合分母和失败停止调度编写了回归检查，尚未执行。它不承担模型性能评估或代码回退功能。

运行后先检查 baseline 差异与 diagnostic_coverage，再分析六项指标和按类/按层的零值比例。当前不能声称某层普遍失效或新条件提升性能。
