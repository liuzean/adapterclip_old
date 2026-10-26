# VisA 分支诊断

使用项目已经配置好的 Adaptclip 环境，可以从项目根目录或 `models_test/` 运行。所有相对输入路径均以项目根目录解析。

```powershell
conda activate Adaptclip
cd E:\projects\FSAD\AdaptCLIP_old
python .\models_test\branch_diagnostics.py --pq_context --save_maps
```

不传 `--k_shots` 时，依次运行 **1-shot → 2-shot → 4-shot**；显式传入 `--k_shots 1`（或 `2`、`4`）仍可只运行一轮。在 `models_test/` 内使用 `python .\branch_diagnostics.py --pq_context --save_maps`，行为相同。

三个 Adapter 默认全部启用，也接受原来的三个启用参数。`pq_context` 现在默认为 **True**，可以保留显式 `--pq_context`；只有使用无 context 的对应权重时才传 `--no-pq_context`。不同配置比较时必须保持此选项一致。

默认权重为 `checkpoint/mvtec_three_adapters_ctx_seed10/epoch_15.pth`，对应当前新训练目录。其训练日志记录三个 Adapter 和 context 均已开启；本次入口修改没有执行权重加载或推理验证。可用 `--checkpoint_path` 指定其他来源明确、配置一致的权重。

每一轮使用独立 Python 子进程，完整结束后进行垃圾回收及 CUDA 缓存清理，再退出进程；控制进程等待其退出后才启动下一轮。进程退出会释放该轮的模型、参考特征、数据缓冲及 CUDA 上下文，避免三轮之间累积占用。三轮各自重新加载模型，不并行运行。某轮失败或 Ctrl+C 中止时停止后续轮次，已有结果保留。清理并不保证 Windows 已扩大的 `pagefile.sys` 立即缩小，也不会删除结果或系统文件。

- 默认保存每类 3 张正常和 3 张异常七栏图。`--vis_per_class 5` 调整每种标签的数量。
- `--save_visualizations` 开启图片，`--no-save_visualizations` 关闭图片；默认开启。
- `--save_maps` 开启 `.npy` 保存；不传则关闭。默认只保存每类前 `vis_per_class` 张正常/异常样本，与是否输出图片无关。
- `--save_maps --maps_scope all` 显式保存全部样本，518 像素全 VisA 的十份图约占 22 GiB。
- `--smoke_per_type 1 --save_maps` 仅对每类前 1 张正常、1 张异常进行功能检查，输出目录带 `_smoke`，不能作为完整数据集结果。
- `--num_workers 0` 可用于 Windows 数据读取排障；正式默认与原脚本一样为 4。

结果固定在 `results/branch_diagnostics/visa_<shot>shot_seed<seed>/`，重复配置自动加 `_run02` 等后缀。`run_status.json` 为 `completed` 才表示全流程完成；失败目录中的 CSV 可能只是部分结果。

Ctrl+C 中止时状态记录为 `interrupted`，保留已有输出；这些 CSV 的 Mean 只代表已完成类别，不能作为全量结果。

输出文件：

- `pixel_metrics.csv`：五分支、各类别及宏平均的平滑后像素指标。
- `image_metrics.csv`：各类别及宏平均图像指标，`Score Source` 区分来源。
- `branch_statistics.csv`：VA/TA/PQA/ALIGN 平滑前原始像素的精确合并统计，按类别和 Normal/Anomaly 分开；Std 为总体标准差。
- `experiment_config.json`、`run.log`：实际参数、评价口径、版本及日志，不记录 Git 哈希。
- `reference_samples.txt`：实际参考样本类别及绝对路径，保留重复抽到的样本；原读取器产生的 prompts 文件也保留。
- `sample_manifest.csv`：输入路径、唯一名称以及对应输出路径，包含全部实际测试样本。
- `visualizations/<类别>/<Normal或Anomaly>/<文件名及路径摘要>.png`：原图是反归一化后的实际模型输入，保证与掩码/热力图对齐。五张热图共用本图的 Turbo 数值色标，显示平滑后结果并注明原始和平滑后极值；显示不参与指标计算。
- `raw_maps/<类别>/<Normal或Anomaly>/<唯一名称>/raw/<分支>.npy`：五分支平滑前原始浮点值，包括最终融合后的 FINAL。
- 同一位置 `evaluated/<分支>.npy`：五分支平滑后评价用值，FINAL 同原脚本 `nan_to_num` 处理后的评价值。每个文件为 `[H,W]`，保留 dtype。
- `baseline_comparison.json`：完整运行时自动与 `results/<数据文件夹>/<shot>/` 现有日志及参考样本比较。历史指标只有一位小数，容差为 0.0501 个百分点。这不等同于重新运行原测试：历史日志缺少完整参数，报告会明确这一限制。

指标均为 **0–100 百分制**，不是比例；不可计算的指标写 `N/A`。某类目标只有一个标签时三个相应指标均记 N/A，宏平均如有缺失也记 N/A，不用零替代或静默忽略类别。`Mean` 是等权类别平均。

| Branch | 主要图像分数及来源 | 像素分数 |
| --- | --- | --- |
| VA | global_vl_score，native_global | local_vl_map + 原 Gaussian |
| TA | global_tl_score，native_global | local_tl_map + 原 Gaussian |
| PQA | global_pq_score，native_global | 四层 local_pq_map 算术平均 + 原 Gaussian |
| ALIGN | 平滑后距离图空间最大值，map_max（即 ALIGN_map_max） | 四层 align_score 调和平均 + 原 Gaussian |
| FINAL | 原 image_anomaly_pred，official_final | 原融合 + Gaussian + 原 nan_to_num |

另列 VA/TA/PQA/FINAL 的平滑后 `map_max` 图像结果，与原生全局得分分开；ALIGN 自身即 map_max，不重复一行。

直接复用 `tools.Evaluator`（实际来自 `tools/effecient_metric.py`），只请求六项指标，不计算 AUPRO。原项目的 AP 实际为 `metrics.AUPR` 的梯形 PR 面积，保持原实现，不换成 sklearn AP。异常图不逐张 Min-Max 归一化，原评价器内部的处理也不修改。检查发现 NaN/Inf 会报错并记录样本路径，不静默修复原始分支图。

只复用模型/数据/参考特征构建等现有函数，在独立入口中复现原 `test.py` 的必要分支聚合和融合步骤。原 `test.py`、`train.py`、模型、数据集和评价器均不修改。每轮只加载一份模型；类别结果在 CPU 缓冲后逐类评价，不将全数据集异常图留在 GPU。选图按测试顺序进行，不使用随机抽样。

以下为手动验证命令，本次修改仅做静态检查，没有自动执行。调度与参数回归验证（项目根目录；使用模拟子进程，不加载模型）：

```powershell
python -m unittest discover -s tests -p test_branch_diagnostics_sequence.py -v
```

原有模型计算回归验证（项目根目录）：

```powershell
python -m unittest discover -s tests -p test_branch_diagnostics.py -v
```

测试直接从未修改的 `test.py` AST 提取原分支推理块，对 CPU/CUDA（如可用）、1/2/4-shot、context 开关及两种原融合方式进行数值逐元素比较，并检查原评价器、N/A、路径、覆盖保护、原始统计和评价时随机状态不变。合成输入测试不能替代真实数据集的完整一致性验证。

真实小规模 FINAL 对照（先用 `--k_shots 1 --smoke_per_type 1 --save_maps` 生成完整小规模输出，再在根目录运行；不指定 shot 会运行三个小规模轮次）：

```powershell
python tests/verify_branch_diagnostics_baseline.py --diagnostic_run results/branch_diagnostics/visa_1shot_seed10_smoke
```

此验证调用原 `test.test`，只通过独立 Dataset 子类选取同一小规模测试子集、Evaluator 子类观察实际输出，不修改原文件。逐元素严格比较 FINAL 平滑图、比较六项 FINAL 指标和参考样本列表。结果保存在 `results/branch_diagnostics/verify_original_<小规模目录名>/verification.json`；不表示完成了全数据集重新运行基线的逐元素验证。
