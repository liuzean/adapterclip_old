# CLIP 正常子空间残差实验

状态：独立实验实现；仅静态检查，尚无运行结果或性能提升结论。

来源：[SubspaceAD](https://github.com/CLendering/SubspaceAD) 的正常子空间重构残差思想。这里复用 AdaptCLIP 的 CLIP 特征，不是官方 DINO+PCA 复现。本地 SubspaceAD 只用于源码审查，本脚本不导入它，也不下载权重。

## 运行

在已有 Adaptclip 环境、项目根目录运行三组：

```powershell
cd E:\projects\FSAD\AdaptCLIP_old
python .\models_test\clip_subspace.py --save_maps
```

按 1/2/4-shot 顺序运行，每个 shot 一个子进程。全部是 VisA、seed=10；模型路径、数据路径、图像尺寸 518、特征层 6/12/18/24、batch_size=8、sigma=4、适配器开关等严格继承已完成的 branch_diagnostics 配置。

首次建议先跑 1-shot 检查适配及运行资源，再决定是否顺序跑三组（重复运行将创建新目录，不续跑或覆盖）：

```powershell
python .\models_test\clip_subspace.py --k_shots 1 --save_maps
```

关闭 PCA、验证恢复原 baseline 的单组命令：

```powershell
python .\models_test\clip_subspace.py --k_shots 1 --disable_pca --no-save_visualizations
```

关闭时不拟合 PCA，候选图像分数及像素图直接共享 baseline；dual_output 对照仍保留。关闭模式也尚未运行。

支持从 models_test 内执行 `python .\clip_subspace.py`。默认源目录 `results/branch_diagnostics`，历史对照 `results/dual_output/visa_seed10`；若使用不同结果会话，可明确传 `--control_root`。默认 `--pca_ev 0.99`、`--num_workers 0`、`--vis_per_type 1`。不要根据 VisA 测试指标搜索 PCA 参数。

## 三个主条件

| 条件 | 像素输出 | 图像输出 |
|---|---|---|
| baseline | 原完整融合图 | 原完整评分 |
| dual_output | 原 no_TA 图 | 原 baseline 评分 |
| clip_pca | CLIP 正常子空间残差图 | 残差平滑图最大值 |

PCA 输出没有接入原融合，没有与双输出、PQA 层权重、RAID 等叠加。它是一个可用于判断新信息价值的独立分支；本轮不能直接宣称改进后的完整 AdaptCLIP 已优于原模型。

## 算法与协议

1. 沿用原参考图片、顺序、重复次数和编码批次，并验证它们为正常 TRAIN、与测试路径不相交。
2. 去除 CLIP 四层已投影特征的 CLS token，固定等权平均后 L2 归一化。四层 `[S,1370,768]` 形成 `[S,1369,768]` 正常 descriptor；实际维度运行时检查。
3. 每类别一个 PCA，只使用 k 张参考的所有 patch。CPU float64 计算均值/协方差及特征分解，默认累计方差 99%；不做梯度训练，不编码其他正常训练图，不使用查询/标签拟合。
4. float32 查询特征求正交重构残差平方，固定除以 4。对单位向量，该比例有理论范围 [0,1]；这是残差量而不是概率。没有逐图/全测试集 min-max 或阈值校准。
5. 网格双线性上采样后，用原 SciPy sigma 平滑；图像取最大值。沿用原六指标、掩码和类别等权宏平均，AP 是本项目梯形 PR 面积。

与官方不同：CLIP/DPAM 而非 DINO，四层而非官方七层，518 而非 672，无 30 次旋转，增加 CLIP L2 处理，图像最大值而非默认 top1% 均值，使用本项目平滑和六指标。不能沿用官方表格数值。

## 保存内容

新结果在 `results/clip_subspace/visa_seed10/`；已存在则为 `visa_seed10_run02/` 等。每组子目录 `visa_1shot_seed10`、`visa_2shot_seed10`、`visa_4shot_seed10`。

- 父目录 `metrics_summary.csv`：三条件×三 shot 六指标；`metrics_by_category.csv` 和 `metrics_vs_baseline.csv`：逐类结果与基线差值。
- 子目录 `metrics_combined.csv`、`pixel_metrics.csv`、`image_metrics.csv`：本组指标。
- `pca_fit.csv`：参考数、token 数、维度、有效秩、主成分数、保留方差、退化状态；`pca_models/*.pt`：本轮 PCA 均值/基底等，不是被覆盖的模型权重。
- `image_scores.csv`：历史控制的八个分量、三条件相关评分及 PCA 图范围；候选开启时 Image Score=Map Max。
- `baseline_reference_comparison.csv`、`control_reference_comparison.csv`、`image_score_parity.csv`、`baseline_map_checks.csv`：历史一致性核对。
- `experiment_config.json`、`reference_samples.txt`、`source_sample_manifest.csv`、`run_status.json`、`run.log`：配置、名单、完整性、日志。
- 默认每类正常/异常各一张，24 张固定可视化；`--save_maps` 只保存这 24 张的 baseline/no_TA/clip_pca 原始与平滑矩阵。双输出像素对应 no_TA，不额外复制矩阵。

热图共用固定 [0,1] 显示范围，PCA 数值可能很小；暗图本身不代表无区分能力，应结合保存矩阵和六指标分析。显示不对评价分数做任何归一化。

## 资源与运行时保护

不增加参考/查询的编码器前向次数。只缓存当前类别的三个独立评价图；类别结束清空并回收，shot 结束退出进程。PCA 状态按类别保存在 CPU，拟合峰值与 k×1369×768 有关；原像素指标仍会占用较大内存/显存，Windows 页文件容量不保证自动缩小。

任何控制指标、逐图分数、已保存旧基线图不一致，或形状/数值异常，立即停止，不继续后续 shot。状态文件标记未完成，部分表不能当作完整结果。旧权重文件内容历史不可追溯，仅靠路径不能证明从未变化；控制结果是运行时证据。

## 文件职责与验证状态

- `clip_subspace.py`：参数、审计、独立进程、复用推理及结果保存。
- `clip_subspace_utils.py`：PCA/残差数学、参考边界、候选输出和评价缓冲。
- `tests/test_clip_subspace.py`：人工运行的程序正确性检查；验证正交残差、退化输入、批次独立性、参考隔离、关闭恢复等，不负责回退文件或证明指标提升。

仅静态 AST/编译、Ruff 和人工调用链检查；未导入运行新程序，未运行测试、模型推理或官方 SubspaceAD。

过程总结、源码审查和静态报告在 `results/clip_subspace/review_20261029/`。`results/` 默认被 Git 忽略，普通代码提交不会自动包含这些报告。
