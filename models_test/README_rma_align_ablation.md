# 实验J H同权重ALIGN定位对照

2026年11月10日实际运行补充：J已完整完成18条件。有ALIGN的H核对通过；关闭ALIGN后相对H三个shot均0/6数据集净正，六指标净和为-66.201267/-60.690705/-68.707540，当前不保留关闭ALIGN。下文开发时的静态验证说明不替代这次真实结果，故意中断/文件锁恢复等尚未做故障注入验证。

本实验不训练、不校准、不搜索比例。固定新D第14轮权重、seed=10、H评分和E的12.5%局部混合，只检验定位路径经过ALIGN的影响。原始test.py、train.py、adaptcliplib和历史H代码不变。

可通过`python .\models_test\run_align_cra_sequence.py`先运行本实验，再运行独立CRA训练/测试。J仅增加内部结果目录回执接口，预测公式不变；不会将J的结果自动叠加到CRA。详见README_cra.md。

| 条件 | 图像分数 | 定位图 |
|---|---|---|
| H | 实时H最终分数 | 12.5%局部混合 → 与ALIGN调和融合 → sigma4平滑 |
| J_no_ALIGN | 与本次H完全相同的分数张量 | 同一12.5%局部混合 → 同样sigma4平滑 |

三分支图来自新D的VA/TA/PQA权重，不是原A。混合公式为`0.875 × 三分支图 + 0.125 × RMA异常概率图`。ALIGN本身的四层聚合保持原状；关闭的是最后定位图与ALIGN图的融合。为计算H原图像分数，仍然需要ALIGN，因此本实验不能解释为完全不计算ALIGN。

H图像分数固定`SH=0.75SE+0.25G0`。SE继续使用G0_L1的25%局部图经过ALIGN和平滑后的完整评分，G0是新D三适配器原生全局均值。绝不根据J的无ALIGN定位图最大值重算图像分数。因此J−本次H的三个图像指标必须精确为零，变化只来自像素指标。

## 运行

在项目根目录：

```powershell
python .\models_test\rma_align_ablation.py
```

也支持进入models_test后执行`python .\rma_align_ablation.py`。默认顺序Visa→BTAD→MPDD→SDD→DTD→part_glass，每集1→2→4-shot，共18个条件。每个dataset/shot只做一次查询编码、一次旧三适配器和一次RMA前向，计算两组输出。一个子进程结束并释放其CPU/CUDA资源后才启动下一条件；类别评价完成也会清理可释放缓存。

固定权重默认`results/train_rma_scratch_new/mvtec_seed10/best.pth`，校验H原第14轮权重SHA256，不能通过参数换成其他权重或改变seed/比例/平滑。参考图默认复用`results/rma_score_coordination_inference/experiment_H_inference_seed10`的历史清单，不重新抽样。核对正常TRAIN身份、顺序、重复次数、shot、查询清单、元数据、历史指标/分数指纹。

可选缩小范围仅供排查，不能据此宣称六集保留判定通过：

```powershell
python .\models_test\rma_align_ablation.py --datasets SDD --shots 1
```

## 结果

首次输出：`results/rma_align_ablation/J_seed10/`。重复运行自动使用`J_seed10_run02`等新目录，不覆盖已有结果。单轮目录示例为`Visa/1shot_seed10/`。

主要汇总文件：

- `metrics_summary.csv`：每数据集、shot的H/J和历史原A全部六指标。
- `metrics_by_category.csv`：各类别六指标。
- `deltas_vs_H.csv`：J相对本次H的六指标差值，主要结论依据。
- `deltas_vs_original.csv`：J相对历史原A差值，保留全部下降。
- `deltas_by_category.csv`：包含类别及Mean的两种比较。
- `retention_by_shot.csv`、`verification_summary.json`：各shot与整体判定。每shot至少3/6数据集六指标净和严格为正且六集总净和严格为正，至少两个shot通过才考虑保留。

每轮还保存`metrics_combined.csv`（本次H/J）、`metrics_with_reference.csv`（另附历史原A）、`image_scores.csv`、`references.csv`、`sample_manifest.csv`、实际配置与状态，以及：

- `score_checks.csv`：实时H分数与历史H的逐图核对，容差1e-6。
- `history_metric_checks.csv`：本次H与历史H完整六指标核对，容差0.001个百分点。
- `image_metric_identity_checks.csv`：本次J与本次H图像三指标必须精确相等。
- 推理时逐像素检查H定位图与本次现有H/E12.5%实现相同，失败即停止；历史完整像素数组未保存，不声称与历史全量数组逐值核对。

原A为已经验证过的历史指标，不重新加载原A模型。默认不生成图片、npy或全测试集像素矩阵文件；评价时在内存保留单类别的两组图，类别完成后释放。Windows仍可能因内存需求使用系统分页文件，清理CUDA缓存不等于将系统分页文件迁移到E盘。

## 中断与继续

失败会在终端和`run_status.json`记录当前数据集/shot、已完成条件及错误，后续停止。查看状态不会启动模型：

```powershell
python .\models_test\rma_align_ablation.py --status .\results\rma_align_ablation\J_seed10
```

继续同一轮实验：

```powershell
python .\models_test\rma_align_ablation.py --resume_from .\results\rma_align_ablation\J_seed10
```

使用实际打印目录。恢复会核对代码/权重/输入和完成输出的指纹，复用完整条件；未完成条件从头推理，使用新的尝试子目录，保留旧文件。不能恢复到单个batch，不会训练。源代码或历史输入改变时停止，不能把两种实现混入同一轮结果。

## 验证范围

开发时只做静态语法、调用/接口、路径与已有文件契约检查，不自动运行模型或pytest。已独立核对18个历史条件、15408次查询对应的清单/标签/参考图、输入文件存在性、元数据及历史指标/分数指纹，并核对第14轮适配器和CLIP权重摘要。记录在`results/rma_align_ablation/development_check_20261109/report.json`；这不是本轮实际推理结果。

`tests/test_rma_align_ablation.py`是CPU小张量、输出隔离与判定逻辑回归测试，已编写但尚未运行。18条件真实推理、H历史一致性及J指标必须以用户实际运行结果为准。

结果用于检验当前固定H下ALIGN的净贡献；即使J整体上升，也不能单凭这一项推导ALIGN在其他模型无效，或解释全部SDD训练退化。
