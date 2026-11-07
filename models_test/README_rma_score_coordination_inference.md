# 实验H真实推理一致性验证

这是实验H的实现验证，不另起实验I，不训练、不重新选择系数。

固定：新D第14轮best.pth、seed=10、lambda=0.25、E的12.5%局部图比例，以及原参考图和查询。

## 运行

项目根目录：

```powershell
python .\models_test\rma_score_coordination_inference.py
```

也支持在models_test中执行`python .\rma_score_coordination_inference.py`。默认Visa→BTAD→MPDD→SDD→DTD→part_glass，每个数据集依次1/2/4-shot，共18轮真实GPU推理；每轮独立进程，等待退出并清理后继续。

不再运行MVTec源域推理或系数网格。只读取已完成H的源域锁定记录，复核固定系数确实为0.25。

## 本次真正计算什么

同一次冻结CLIP编码，计算原A、新D分支及RMA，复用E原有融合函数：

1. 重建25%局部融合＋ALIGN＋平滑的G0_L1图及完整图像分数；该分数就是SE。
2. 重建12.5%局部融合＋ALIGN＋平滑最终图，作为本次E定位图。
3. 从真实分支输出直接取得G0，计算`SH=0.75*SE+0.25*G0`，不从最终分数反推G0，不用12.5%图最大值重算SE。
4. H的原始融合图和最终平滑图直接引用本次E图。逐图检查原始图与最终图相等；H只有图像分数改变。
5. 按原评价器重新计算整个测试集的像素三指标；H/E只保存三份唯一像素缓冲，不额外复制H/E全测试矩阵。H图像三指标独立计算，不能复用E图像指标。

单类别评价后释放缓冲，单轮进程退出后释放模型、内存和CUDA上下文。默认不生成新矩阵或图片；需要固定少量样本时显式传`--save_maps`或`--save_visualizations`，不会保存完整测试集矩阵。

## 核对内容

- 原A/G0_L1/完整新D的原端点评分、六指标、原参考与查询身份。
- 真实G0、完整SE和SH逐图对照已完成H的离线记录，绝对容差1e-6。
- 本次H/E原始及最终像素图逐图相等；像素指标精确相同。
- 本次E的类别与均值六指标对历史E、本次H六指标对离线H，容差0.001个百分点。
- 冻结参数、缓冲、权重和历史文件指纹不变。任何检查失败，停止后续条件，不能将失败写成验证通过。

历史E没有保存全测试像素数组，所以不能声称本次与历史全测试像素逐值重比。这里保证的是本次H/E逐像素相等、相同已核对的E推理路径，以及历史完整指标一致。原A已有选定样本矩阵会继续由现有工具检查。

AP继续使用原项目评价器口径；不改成sklearn.average_precision_score。该验证确认实现与离线结果一致，不提供新的独立盲测或跨seed稳定性证据。

## 结果

```text
E:\projects\FSAD\AdaptCLIP_old\results\rma_score_coordination_inference\experiment_H_inference_seed10\
```

重复启动自动增加`_run02`，不覆盖旧结果。每轮默认在`<数据集>/<shot>shot_seed10/`。

- `image_scores.csv`：原A、G0_L1、完整新D、局部候选、E、H的真实分数。
- `H_score_and_pixel_checks.csv`：实际G0/SE/SH及逐图历史差值、本次H/E原始/最终图相等记录。
- `metrics_combined.csv`：本次重新评价的完整六指标、各类别和Mean。
- `H_offline_metric_checks.csv`：H对离线H、E对历史E的六指标差值。
- `baseline_reference_comparison.csv`、`endpoint_*checks.csv`：复用原端点回归检查。
- `references.csv`、`sample_manifest.csv`、`input_audit.json`：查询、参考身份及协议。
- `run_status.json`：本轮是否完整完成、冻结和三类H一致性检查是否通过、输出指纹。

根目录保存`metrics_summary.csv`、`dataset_deltas.csv`、`retention_by_shot.csv`、`offline_metric_checks.csv`与`verification_summary.json`。E是直接对照，同时报告原A、G0_L1及完整新D。六集三个shot未跑齐时不作整体保留判定。

已有离线H默认目录为`results/rma_score_coordination/experiment_H_seed10`，可用`--h_run`指定另一份完成记录；固定lambda仍必须是0.25。权重从已完成记录读取，不通过CLI切换权重或比例。

## 查询进度与续跑

只读状态，不开始/恢复推理：

```powershell
python .\models_test\rma_score_coordination_inference.py --status .\results\rma_score_coordination_inference\experiment_H_inference_seed10
```

恢复同次任务：

```powershell
python .\models_test\rma_score_coordination_inference.py --resume_from .\results\rma_score_coordination_inference\experiment_H_inference_seed10
```

使用实际打印的目录。续跑恢复原选项，核对历史/代码/权重，只复用通过全部检查且指纹一致的完整轮次。未完成轮次保留并在`retry_attempts/`新目录重跑；根记录`job_results`指向实际结果，汇总不会混入失败尝试。中断打印当前数据集/shot、已完成项、总目录和当前子任务目录。

## 已完成运行验证

截至2026-11-07，本入口六数据集1/2/4-shot的18轮真实推理已完成，15408次查询的G0/SE/SH对离线分数最大差0，本次H/E图逐像素相同，1476项六指标核对通过。H相对E和原A三个shot均符合当前保留规则，下降项全部保留。

之后run_test_h_sequence完成精简test_h的H/E各18条件回归，属于相同方法的实现验证。当前结果、证据边界和主线省略过程见README_H_mainline.md；不称为新seed或独立盲测。

## 开发边界

新增主脚本、验证工具、说明和`tests/test_rma_score_coordination_inference.py`；复用既有推理钩子及指标工具，不修改E/F/G/H既有实现、正式入口或模型代码。契约测试覆盖真实G0使用、完整E评分、像素引用、H独立图像评价、历史差异检测和只读进度；开发阶段只做静态检查与历史文件接口审查，不自动运行pytest、项目模块或GPU推理。
