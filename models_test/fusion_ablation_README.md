# VisA 融合消融与 PQA 四层诊断

本入口落实本轮确认的安排：VisA、seed=10，分别评估 1/2/4-shot，保留六项指标；同一次推理顺带提取指定四张图片的 PQA 四层图。之后依据实际结果决定是否移植针对性模块。当前没有移植或训练新模块。

## 运行

在原来的 Adaptclip 环境中执行：

```powershell
cd E:\projects\FSAD\AdaptCLIP_old\models_test
python .\fusion_ablation.py
```

默认依次运行三个 shot，每个 shot 独立子进程。一个 shot 失败或被中断时停止后续条件。每个 shot 完整遍历 VisA 一次，同时得到三种融合结果，不会为三种条件分别重复模型推理。

建议先单独运行 1-shot，确认 baseline 核对通过，再执行其他 shot：

```powershell
python .\fusion_ablation.py --k_shots 1
python .\fusion_ablation.py --k_shots 2
python .\fusion_ablation.py --k_shots 4
```

这些单独命令各生成独立结果目录；默认无 `--k_shots` 的一次运行才能自动汇总三组为同一张九行表。

可选参数：

```powershell
# 附加保存指定四张图的原始/平滑后矩阵
python .\fusion_ablation.py --k_shots 1 --save_maps
# 不生成可视化，但仍保存四层统计和全测试集指标
python .\fusion_ablation.py --k_shots 1 --no-save_visualizations
# 使用已有源实验的数据读取进程数；默认 0 更节省主机内存
python .\fusion_ablation.py --k_shots 1 --num_workers 4
```

`--source_root` 默认指向项目根目录的 `results/branch_diagnostics`。相对路径均基于项目根目录；不依赖 PowerShell 当前目录。其他参数从各 shot 的 `experiment_config.json` 读取，包括权重、数据路径、518 尺寸、batch size=8、sigma=4、四层特征、全部 Adapter 开关和 PQContext。无需再传 `--visual_learner`、`--textual_learner`、`--pq_context` 等旧入口参数。

## 三种条件的确切含义

记 `mean`、`harmonic_mean` 为原始 `adaptcliplib.fusion_fun`，其内部细节保持原样：

| Condition | 平滑前像素图 |
|---|---|
| baseline | harmonic_mean(mean(VA, TA, PQA), ALIGN) |
| no_align | mean(VA, TA, PQA) |
| pqa_align | harmonic_mean(PQA, ALIGN) |

每种条件都先融合原始图，再按原始 SciPy Gaussian 设置平滑。评价、统计和矩阵保存均不做逐图归一化。

**三种条件的图像级全局部分固定为 mean(VA_global, TA_global, PQA_global)**；只将相应平滑后像素图的最大值代入原始最终调和融合。`pqa_align` 仍保留 VA/TA 的原生全局分数，因此它是像素融合消融，不代表一个完全移除 VA/TA 的完整模型。`no_align` 只移除最后的 ALIGN 融合；PQA 内部的 Top-1 对应、残差和 PQContext 仍原样运行。

本轮只实现约定的三个条件；逐一移除 VA/TA/PQA、改权重、改尺度、换模块均留待结果分析后决定。

## 公平性与 baseline 核对

- 各 shot 直接读取原始 `reference_samples.txt`，不调用 `PromptDataset` 的随机抽样构造函数；复用原始 `__getitem__` 和 `test.build_prompt_memory`。保持原参考图顺序、重复次数、编码 batch size 和最后一个 batch 的大小。
- 读取 VisA `meta.json`，核对参考图是对应类别的正常训练样本，并核对完整测试列表、顺序和标签。跨 shot 核对模型配置、评价定义、源版本信息、测试顺序和参考图包含关系。
- 每批只提取一次特征，VA/TA/PQA 各执行一次，三种融合共享同一批输出。Adapter 均使用严格加载的原参数和 `eval()`。
- 六项指标调用原 `tools.Evaluator`，不添加 P-AUPRO。仓库中名为 AP 的指标是原 AUPR 梯形积分，继续保持原定义。所有指标为百分数，Mean 为类别等权宏平均，未定义指标为 `N/A`。
- baseline 的 12 类和 Mean 共 78 个指标数值与已有完整精度 CSV 对照，默认绝对容差 `0.001` 个百分点。源记录存在矩阵时，还核对四个指定样本的 PQA/ALIGN/FINAL 原始图及平滑图，默认 `atol=1e-6, rtol=1e-5`。
- 每完成一类便核对该类 baseline，最后再核对完整宏平均；不一致时写出检查结果并停止，不能直接把差异当作新方法收益。缺失旧矩阵标为 `unavailable`，不会伪称一致。此核对针对已有诊断结果，不代表本次额外运行了原始 `test.py`。
- 路径一致无法证明历史权重内容从未改变。旧配置未记录权重内容摘要，若旧权重已被替换，运行时对照可能失败。

## PQA 四层来源和形状

原始 `PQAdapter.forward()` 已返回 `global_logits, local_scores, align_scores`。直接取四个 `local_scores[i][:, 1]`，按 `features_list=[6,12,18,24]` 命名，不创建新 Adapter。

以当前 ViT-L/14、518 输入为例：

1. 每层 query `[B,1370,768]`，prompt `[B,S,1370,768]`；去掉 CLS 后有 37×37=1369 个 patch。
2. 原代码归一化特征，在全部参考 patch 中做 Top-1 最近邻，得到 ALIGN 距离和对应参考特征。
3. PQContext 开启时计算 query + abs(query - aligned_prompt)，形状 `[B,768,37,37]`，进入原 `sharebn[i]` 和 local adapter。
4. 原 local adapter 输出两通道 logits，经 softmax、双线性插值后为 `[B,2,518,518]`。异常通道为 `[B,518,518]`。
5. 四层异常通道按原代码拼接为 `[B,4,518,518]`，均值得到 PQA；原始四层 ALIGN 按原公式调和融合。代码核对提取后的四层均值，并进一步与已保存原 PQA 图比对。

原实现的 local/global adapter 在各层共享实例，不能将它们描述为四个独立训练的 head。所有最近邻、残差及 Adapter 计算均仍由原始 forward 完成。

只定位 `sample_manifest.csv` 中 `macaroni1` 的异常 000/001/002 和正常 0790，要求类别、类型、文件名共同匹配且唯一，不手工拼接图片路径，也不自动换图。

## 输出

默认第一次运行：

```text
results/fusion_ablation/visa_seed10/
  input_audit.json
  run_status.json
  metrics_summary.csv                 # 三个 shot × 三种条件，六指标宏平均
  metrics_by_category.csv             # 三个 shot × 三种条件 × 12 类
  metrics_vs_baseline.csv             # 每类及 Mean，相对本轮 baseline 的百分点差
  visa_1shot_seed10/
  visa_2shot_seed10/
  visa_4shot_seed10/
```

再次运行建立 `visa_seed10_run02/` 等新目录；已有结果不覆盖。单 shot 运行只包含该 shot。部分成功时汇总表可能只有已完成 shot，应同时检查父子目录的 `run_status.json`。

每个 shot 内的主要文件：

| 文件 | 用途 |
|---|---|
| pixel_metrics.csv / image_metrics.csv | 三种条件的各类别及 Mean 指标 |
| metrics_combined.csv | 合并六指标，成功核对 baseline 后写出 |
| baseline_reference_comparison.csv | 当前 baseline 对已有 FINAL 完整精度指标的差异 |
| baseline_map_checks.csv | 四张诊断样本与已有矩阵的逐图对照 |
| pqa_layer_statistics.csv | 四图 × 四层 × raw/evaluated，共 32 行/shot |
| diagnostic_statistics.csv | 同四图的 PQA_Fused、ALIGN、三种融合图统计，共 40 行/shot |
| image_scores.csv | 全部测试图的三个原生全局分数、固定全局均值、三种 map max/最终图像分数 |
| experiment_config.json / diagnostic_config.json | 实际配置、版本、容差及诊断样本 |
| reference_samples.txt | 本轮实际使用的原参考图名单 |
| source_sample_manifest.csv | **原实验**的样本清单；其中输出路径相对于 source_result_dir |
| diagnostic_samples.csv | 本轮四张诊断图片的定位信息 |
| run.log / run_status.json | 运行记录与完整性状态 |

`visualizations/pqa_layers/` 下每图九面板：模型实际输入（反归一化）、GT、PQA_L6/L12/L18/L24、PQA_Fused、ALIGN、FINAL(baseline)。

`visualizations/fusion_conditions/` 下每图五面板：实际输入、GT、baseline/no_align/pqa_align。显示平滑后的图，所有 shot 固定同一 `[0,1]` 色标，面板另标注原始和平滑后数值范围。超过显示范围时仅颜色饱和，实际数据不裁剪。图中 Original 是模型变换后的输入，保持与掩码/热图空间对齐。

`--save_maps` 额外保存 `raw_maps/<类别>/<类型>/<sample_key>/{raw,evaluated}/`，每阶段包含四层、PQA_Fused、ALIGN、三种融合共九个 `.npy`；不保存全数据集矩阵。

层统计区分 raw/evaluated，包含 min/max/mean/std/P50/P95、缺陷/背景统计及差值；正常图另有最高 1% 像素均值。四张图的统计只用于定位现象，六项性能指标来自完整测试集。

## 内存与磁盘

按类别仅保留三种平滑图和一份 uint8 GT；类结束即评价并释放，shot 结束退出子进程。默认 DataLoader `num_workers=0`。这会减少持续保留的主机内存，但精确像素指标排序仍有峰值内存需求，不能保证 Windows 页面文件不增长或 C 盘立即恢复。

模型/数据/结果路径按源配置和本脚本规则解析；本入口没有迁移 Windows 页面文件、系统 TEMP 或第三方缓存。无需为本轮实验调整系统设置。

## tests 与实际验证

`tests/test_fusion_ablation.py` 是程序正确性回归检查：固定全局分数策略、原始 baseline 公式、非有限值拒绝、baseline 比较、结果不覆盖以及失败停止调度。它使用小张量或模拟进程，不加载模型权重。它既不研究模型性能，也不承担“退回代码”的功能；代码回退由用户决定，Git 管理。

本次仅做 AST/编译静态检查、Ruff 未定义/未使用符号检查及源码调用审查，**没有运行此入口或 tests，也没有加载权重或推理**。真实权重加载、运行时张量、数值复现、显存峰值、CSV/图片实际生成均需用户手动运行确认。静态检查记录与本轮只读输入核对见 `results/fusion_ablation/development_review_20261026.md`。
