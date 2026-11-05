# 新D RMA全局项与局部图独立开关

在项目根目录执行：

```powershell
python .\models_test\rma_output_ablation.py
```

也可在models_test目录直接运行。默认六数据集Visa→BTAD→MPDD→SDD→DTD→part_glass，
每数据集依次1/2/4-shot，seed10。只推理，不重新训练，不更换新D第14轮权重。
自动由已完成新D测试的campaign_config.json绑定其准确best.pth和哈希，不猜测最新目录。

| 条件 | 全局分数融合 | 局部图融合 |
| --- | --- | --- |
| G0_L0 | 新D的VA/TA/PQA三分支平均 | 三分支图平均＋ALIGN＋平滑 |
| G1_L0 | 加入RMA，四分支平均 | 三分支图平均＋ALIGN＋平滑 |
| G0_L1 | 新D的VA/TA/PQA三分支平均 | 加入RMA，四分支图平均＋ALIGN＋平滑 |
| G1_L1 | 加入RMA，四分支平均 | 加入RMA，四分支图平均＋ALIGN＋平滑 |

G是native全局项开关，不是最终图像分数的独立开关。
四条件都用本条件全局项与本条件平滑后异常图最大值做原调和融合。
因此只开局部图也可能改变图像指标，不是复用原A图像分数的双输出。
三个旧适配器均来自同一新D权重；全关不代表恢复原A训练权重。

每查询batch只编码一次CLIP、新D三个适配器和RMA各计算一次，四条件共享结果。
另计算原A三个适配器作独立基线，复用同一CLIP特征与参考库。
ALIGN、sigma、图像尺寸、batch、查询/正常参考名单和评价协议与历史一致。
核对G0_L0与实验十四、G1_L1与完整新D的逐图分数及每类别/均值六指标；
同时核对原A分数、指标、选定样本矩阵。不一致停止，不写成完整通过。

## 内存和结果

每dataset/shot独立子进程，退出并清理后再继续；失败停止并显示数据集、shot和已完成数量。
每类别评价后清理空闲缓存；只缓存原A和两套唯一候选图，不重复缓存四套图。
四套图像分数分别评价；像素图完全相同的条件，检查别名/张量一致后复用其像素指标。
进程退出释放本轮模型和张量，不保证操作系统分页文件立即缩小。

结果写入results/rma_output_ablation/six_datasets_seed10，重复运行新建编号目录，不覆盖。

- metrics_summary.csv、metrics_by_category.csv：原A及四条件的完整六指标。
- deltas_vs_original.csv、deltas_vs_new_D.csv：各条件减原A或完整新D。
- factor_effects.csv：固定另一开关时，打开全局项或局部图的指标变化，Left减Right。
- retention_by_shot.csv：各条件相对原A，按每个shot分别判断。
- image_scores.csv：逐图四条件分数及G0/G1全局项、L0/L1异常图最大值。
- endpoint_score_checks.csv、endpoint_metric_checks.csv：历史两端点回归核对。
- references.csv、sample_manifest.csv、input_audit.json：固定输入身份和来源记录。
- run_status.json：完成状态、查询编码/适配器调用批次数及核对状态。

默认保留固定少量样本可视化，不默认新增矩阵。需要选定样本矩阵可运行：

```powershell
python .\models_test\rma_output_ablation.py --save_maps
```

四条件预先固定，全部结果保留；不按目标标签搜索权重或按类别挑开关。
原train.py、test.py、adaptcliplib及已有实验入口均不修改。
开发只做静态检查和历史输入审计，未运行推理或pytest，未确认四条件运行结果。
