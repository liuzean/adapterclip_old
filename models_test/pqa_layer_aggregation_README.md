# PQA 层间聚合实验

只改变原 PQA 四层异常概率图的聚合方式，比较聚合本身与双输出的独立作用。它是待验证的轻量候选，不是已经证明有效的方法，也没有移植未审查的论文代码。

## 运行

在原 Adaptclip 环境、项目根目录：

```powershell
cd E:\projects\FSAD\AdaptCLIP_old
# 先运行完整 1-shot 验证
python .\models_test\pqa_layer_aggregation.py --k_shots 1
# 依次完整运行 1/2/4-shot
python .\models_test\pqa_layer_aggregation.py
```

默认 VisA、seed=10，复用 `results/branch_diagnostics/visa_{shot}shot_seed10/` 的模型配置、权重、数据、518 输入、batch size=8、sigma=4、层 6/12/18/24、三个 Adapter/PQContext 开关及原参考图顺序，不重新抽样。

默认 `--control_root results/dual_output/visa_seed10`，读取已完成双输出会话中的 Baseline/no_TA 记录进行核对；若它在编号目录，必须明确指定，不自动挑选成绩最好的一次。

支持在 `models_test/` 内启动；所有相对参数路径均基于项目根目录。每个 shot 独立子进程，上一个退出后再开始下一个；单 shot 每批只有一次主干、VA、TA、PQA 前向，四种条件共享输出。

## 四种条件

设 `P0` 为原四层均值、`P1` 为候选加权 PQA 图，`G` 为原 Gaussian，`H` 为原调和融合，`g` 为固定原生全局均值 `mean(VA_global,TA_global,PQA_global)`：

- `M0 = G(H(mean(VA,TA,P0),ALIGN))`
- `N0 = G(H(mean(VA,P0),ALIGN))`
- `M1 = G(H(mean(VA,TA,P1),ALIGN))`
- `N1 = G(H(mean(VA,P1),ALIGN))`

| 条件 | 像素图 | 图像分数 |
|---|---|---|
| baseline | M0 | H(g,max(M0)) |
| dual_output | N0 | H(g,max(M0)) |
| pqa_weighted | M1 | H(g,max(M1)) |
| pqa_weighted_dual | N1 | H(g,max(M1)) |

新家族的两个图像分数都来自 M1。它们相互相同，但**不保证等于原 Baseline**。这样才能检查修改 PQA 对分类的真实影响，不强制复制原分数掩盖退化。TA 的原生全局贡献、PQAdapter.forward、匹配、残差、BN/head、PQA 原生全局 logits、ALIGN、平滑和原六指标定义不变。

## 候选规则

每张图独立计算，不读取类别、标签、GT、历史指标，也不使用其他 batch 样本或可训练状态：

1. 原四层概率图缩略到固定 32×32，仅用于算权重。每层单独转 float64 池化，避免创建整份四层 double 图。
2. 对每层空间图中心化、按空间最大绝对差缩放并作 L2 归一化；等价于用于 Pearson 相关的空间方向。常量层无空间方向，不参与相关投票。
3. 每层的 agreement 是与其他有效层的正相关均值（自身不投票、负相关按 0 计）。归一化为 q；如果没有有效正一致性（总量 <= float64 epsilon），退回均匀权重。
4. `w = (1-alpha)/4 + alpha*q`，默认 `alpha=0.5`，每层至少保留 0.125 的权重；不永久删除任何层。
5. 将权重用于**原始全分辨率概率图**加权。缩略、中心化、归一化结果不代替最终预测，不做逐图预测归一化或裁剪。无一致性样本直接返回原始均值。

alpha=0.5 和 grid=32 是预先固定的保守候选设置，没有根据 VisA 标签拟合或搜索。`--aggregation_strength` 允许显式修改用于将来独立验证，但入口不搜索、不挑选最优结果。先固定规则再评价，不在当前测试集上反复挑 alpha。

它只描述“层间空间一致性”，不代表真实正确概率。共同背景误报也可能一致；某个正确的独有缺陷响应可能被降权。已经全零的异常概率不能靠加权恢复。本轮没有修复 logits 饱和，也不保证追回 P-AUROC 或提高任何指标。

## 恢复原计算

```powershell
python .\models_test\pqa_layer_aggregation.py --k_shots 1 --disable_aggregation
# 等价的有效强度设置
python .\models_test\pqa_layer_aggregation.py --k_shots 1 --aggregation_strength 0
```

关闭后 `pqa_weighted` 应等于 baseline，`pqa_weighted_dual` 应等于现有 dual_output；**关闭聚合不等于同时关闭双输出**。推理代码会检查两组平滑图与分数逐元素精确恢复。恢复整个原算法应看 baseline 条件，或关闭后看 pqa_weighted 条件。

## 输出

默认会话目录 `E:\projects\FSAD\AdaptCLIP_old\results\pqa_layer_aggregation\visa_seed10\`，重复运行编号 `_run02` 等，不覆盖旧结果。父子 run_status 必须 completed/complete=true，失败中断表格均可能不完整。

| 文件 | 内容 |
|---|---|
| 父 metrics_summary.csv | 三个 shot×四条件，完整时 12 行宏平均六指标 |
| 父 metrics_by_category.csv | 完整时 144 行类别结果 |
| 父 metrics_vs_baseline.csv | 完整时 156 行差值，含 Mean |
| 每 shot metrics_combined.csv | 四条件×12 类及 Mean，共 52 行 |
| baseline_reference_comparison.csv | 原始 Baseline 六指标 78 项核对 |
| control_reference_comparison.csv | Baseline/现双输出历史六指标共 156 项核对 |
| image_scores.csv / image_score_parity.csv | 全 2162 图的全局分量、四像素图最大值/分数、各条件输出来源；原八个分量逐图历史核对 |
| layer_weights.csv | 全 2162 图的 L6/L12/L18/L24 权重、agreement、fallback、有效 alpha |
| routing_checks.csv | 每批输出来源与是否执行关闭恢复检查；关闭时失败会立即报错 |
| baseline_map_checks.csv | 24 张固定图片的原 FINAL raw/evaluated 矩阵核对，默认 48 项；缺失标 unavailable |
| experiment_config.json / input_audit.json | 实际参数、输出映射、依赖版本和输入文件内容摘要，不记录 Git Commit Hash |
| reference_samples.txt / source_sample_manifest.csv / visualization_samples.csv | 原参考图、原测试清单、固定展示名单 |
| run.log / run_status.json | 日志及完成状态 |

默认每类清单中首张正常/异常，共 24 张六面板图：原图、GT、四条件定位图，统一 [0,1]。`--vis_per_type 3` 可选 72 张；选样不读取预测或指标。

默认不保存 `.npy`。`--save_maps` 仅保存上述 24 张的四套 raw/evaluated 矩阵，约 197 MiB/shot、三组约 590 MiB；不保存全量特征或矩阵。`--no-save_visualizations` 可关闭图表，仍做固定原矩阵核对。

类别缓冲需要四套唯一像素图，比 dual_output 的两套更多内存。每类结束释放缓冲，各 shot 退出后释放进程资源，但精确像素评价仍有峰值，不能保证 Windows 页面文件立即缩小。

## 结果判断

先核对两种原有对照和逐图分数，默认指标容差 0.001 个百分点、数组 atol=1e-6/rtol=1e-5。源输入或控制条件变化、核对失败会中止后续 shot。

再分别比较 pqa_weighted 对 baseline、pqa_weighted_dual 对 dual_output，分清聚合和双输出收益；六项全部保留。AP 沿用原仓库梯形 PR 面积，类别等权。不得按类挑选条件，不能在实验前声称加权更好。最好用源域/独立验证划分确定设计后再报告目标测试结果；本次仍是探索候选。

## 开发验收

Codex 仅做静态检查，不执行入口、`--help`、模型、模拟输入或测试。`tests/test_pqa_layer_aggregation.py` 提供 7 项回归测试，已编写未运行；它检查程序逻辑，不承担性能评价或代码回退。用户可自行运行：

```powershell
python -m unittest discover -s tests -p test_pqa_layer_aggregation.py
```

静态检查记录：`results/pqa_layer_aggregation/development_review_20261028.md`。本轮只新增独立文件，现有正式代码、三个旧实验及 dual_output 文件保持不变。历史双输出提交建议：`Add dual-output routing with baseline image scores and no-TA pixel maps`；不得把两项像素收益写成六指标全面提升。
