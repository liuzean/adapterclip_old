# 实验G：SDD定位与part_glass评分误差归因

本实验解释已有候选的不足，不训练、不修改模型、融合比例或评分规则，不自动给出某个新模块一定有效的结论。

## 运行

默认从项目根目录运行：

```powershell
python .\models_test\rma_target_diagnostics.py
```

也支持在`models_test`目录内运行`python .\rma_target_diagnostics.py`。相对路径均从项目根目录解析。

默认流程：

1. 核对六数据集、1/2/4-shot的已有权重、指标、逐图分数、参考图身份/顺序、查询清单和评价配置，保存已有完整六指标与差值。
2. 依次补跑`SDD 1→2→4-shot`、`part_glass 1→2→4-shot`，始终seed=10。每条件一个子进程，共6个子进程；等待退出、清理后再运行下一个条件。
3. 共享同一批冻结CLIP查询特征，计算各套适配器的分支输出、融合阶段、最终评分和逐图区域统计；不为每套适配器重复编码查询图像。
4. 汇总诊断结果。其余四个数据集复用已有结果，不重复推理。

只整理已有文件、不推理：

```powershell
python .\models_test\rma_target_diagnostics.py --analyze_only
```

默认保存少量固定样本的可视化，每个类别最先出现的2张正常和2张异常图。默认不保存任何新的`.npy`/`.npz`矩阵。确实需要保存固定样本矩阵时：

```powershell
python .\models_test\rma_target_diagnostics.py --save_maps
```

关闭图片、只保存诊断表：

```powershell
python .\models_test\rma_target_diagnostics.py --no-save_visualizations
```

可用`--datasets SDD`或`--shots 1`显式缩小补充推理范围；已有六数据集18条件对照表始终完整保留。没有通过这些开关改变默认seed。

## 五个输出条件、三套已有模型权重

| 名称 | 含义 | 权重 |
|---|---|---|
| A | 原AdaptCLIP | 原三适配器epoch_15.pth |
| T_three | 从训练开始就没有RMA的新流程三适配器对照 | 该训练自己源域验证选出的best.pth，记录实际epoch |
| D_off | 新D共同训练后，测试时关闭RMA | 同一新D第14轮best.pth |
| G0_L1 | 旧三支全局项＋25%局部融合后的原完整评分/图 | 同一新D第14轮best.pth |
| E | G0_L1完整图像分数＋12.5%定位图，经ALIGN和平滑 | 同一新D第14轮best.pth |

新D与T_three不是同一份权重。D_off也不是独立训练的三适配器对照。程序从已经完成的历史测试campaign定位各自权重，不假设三适配器的最好权重也在第14轮。

默认读取以下已有实验目录；均可通过相应`--*_run`参数明确替换：

- `results/test_rma_scratch_new/six_datasets_seed10`
- `results/new_d_without_rma/six_datasets_seed10`
- `results/rma_output_ablation/six_datasets_seed10`
- `results/test_new_d_three/six_datasets_seed10`
- `results/rma_fixed_dual_output/six_datasets_seed10`

## 结果保存

所有本实验输出位于：

```text
E:\projects\FSAD\AdaptCLIP_old\results\rma_target_diagnostics\experiment_G_seed10\
```

重复运行自动创建`experiment_G_seed10_run02`等新目录，不覆盖旧结果。

根目录的重要文件：

- `historical_metrics.csv`：六数据集18条件、五种输出、所有类别及Mean的完整六指标，注明历史来源。
- `historical_deltas.csv`：新流程三适配器减A、D_off减三适配器、加入25%局部图、E减D_off、E减A的差值。
- `history_report.md`：对照身份、指标来源与解释边界。
- `region_summary.csv`、`paired_error_summary.csv`、`gap_summary.csv`：成功补充轮次的区域诊断汇总。
- `ranking_summary.csv`、`image_metric_checks.csv`：成功轮次的排序诊断及图像指标核对。
- `campaign_config.json`：实际运行选项、三套权重身份、原始配置/清单指纹和代码指纹；不记录Git提交哈希。
- `run_status.json`：当前阶段、数据集/shot、已完成项和各轮实际目录。

每轮目录的重要文件：

- `region_stats.csv`：逐图缺陷/背景的均值、P95/P99、最大值等统计。
- `foreground_background_gaps.csv`：缺陷与背景的分离情况。
- `paired_errors.csv`：相同位置在不同条件下新增/消除高响应、缺陷漏响应情况。
- `region_summary.csv`、`paired_error_summary.csv`、`gap_summary.csv`：区域与配对错误汇总。
- `image_scores.csv`：同一批图像的五种最终分数。
- `score_components.csv`：VA/TA/PQA全局分数、全局均值、评分图最大值、显示图最大值，以及峰值是否落在缺陷区域。
- `ranking_summary.csv`：最终分数及各评分分量的AUROC、正确/错误/同分配对数量。
- `ranking_changed_pairs.csv`：正常—异常图像配对中，哪些排序变好或变差；同分计0.5。
- `ranking_error_images.csv`：哪些图像贡献了更多新增排序错误；是后验诊断，不代表无偏挑选的展示样本。
- `historical_score_checks.csv`：五种最终分数逐图对照历史，绝对容差1e-6。
- `score_decomposition_checks.csv`：全局均值与评分图最大值能否重建实际最终分数；只核对，不替换推理输出。
- `image_metric_checks.csv`：用原评价器重新计算图像三指标并核对历史，容差0.001个百分点；另用配对排序独立核对图像AUROC。
- `selected_samples.csv`、`visualizations/`、`branch_visualizations/`：固定少量样本及最终图/分支图；`--save_maps`才增加`selected_maps/`。
- `sample_manifest.csv`、`references.csv`、`input_audit.json`：完整查询与参考身份。
- `run_status.json`、`diagnostics.log`：进度和错误信息。

## 怎么解释

SDD要同时观察缺陷响应是否充分、正常背景是否出现高响应。区域均值并非越高越好。P99等分位数是每张图内部区域的分位数，其汇总均值不是把所有像素混合后的分位数。AP沿用原项目PR曲线梯形积分口径。

阶段划分：`native`为单分支/融合前输出；`branch_smoothed`为单支路经过原sigma平滑；`aligned`为完整局部融合＋ALIGN、尚未平滑；`evaluated`为最终平滑图。单分支及RMA_only未经过完整融合，不能仅凭更红或区域绝对分数更大判断优劣。

part_glass重点看新增排序错误究竟由全局项、局部最大值或两者组合引起。E的`ScoringMapMax`来自G0_L1的25%图；`DisplayedMapMax`来自E的12.5%图。不能用后者解释为E图像分数的直接输入。

固定0.25/0.5/0.75分数切点只用于描述错误，不是源域选出的部署阈值，不从目标测试标签选阈值/参数。保留全测试集统计；固定样本图用于解释，不代替全测试集指标。

训练流程、联合训练及各自最佳epoch可能共同影响结果。三适配器与四适配器的对照帮助定位问题范围，不把差值直接写成唯一原因的证明。只使用seed=10，不声称跨seed稳定。

## 内存、缓存与验证范围

每张图立刻归约为统计数值，内存主要保留当前batch及少量标量行，不构建新的全测试集像素矩阵缓存。子进程退出释放其模型、内存和CUDA上下文；`empty_cache`只释放未使用显存，不会强行删除活跃张量或控制Windows分页文件。

完整像素三指标直接复用历史已完成的全测试集评价，不声称本轮重算。此次实际重新计算图像三指标，逐图核对历史分数，检查冻结状态和输入/权重/代码一致性；中间区域统计覆盖补跑数据集的全部查询。

## 中断与续跑

只查询进度：

```powershell
python .\models_test\rma_target_diagnostics.py --status .\results\rma_target_diagnostics\experiment_G_seed10
```

继续同一实验：

```powershell
python .\models_test\rma_target_diagnostics.py --resume_from .\results\rma_target_diagnostics\experiment_G_seed10
```

请换成终端实际打印的目录（可能带`_run02`）。续跑恢复该次记录的选项，核对历史文件/权重/代码，复用已通过检查且输出指纹仍一致的轮次。未完成轮次会在该数据集下创建新的`*_run02`尝试目录，旧失败文件保留；成功汇总只读取`job_results`指向的结果。

`--status`不运行推理。`--analyze_only`结果不能用`--resume_from`转换成诊断推理，应重新正常启动。失败会打印当前阶段、数据集/shot、已完成项、总目录和当前子任务目录，然后停止后续任务。

## 开发验证

新增契约测试位于`tests/test_rma_target_diagnostics.py`，覆盖排序同分、异常分数、参考顺序、固定样本选择、区域错误区分、评分分解和只读进度查询。遵循当前静态验证约定，开发阶段不自动运行pytest、项目模块、`--help`、训练或推理；实际完成与未完成的检查以开发汇报为准。
