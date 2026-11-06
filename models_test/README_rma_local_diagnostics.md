# MVTec源域局部输出误差归因

项目根目录执行：

```powershell
python .\models_test\rma_local_diagnostics.py
```

也可以在models_test内执行 `python .\rma_local_diagnostics.py`。
默认原MVTec源域验证集全部345张查询、15类、1/2/4-shot、seed10。
固定新D第14轮best.pth，原A使用其原epoch15三适配器；全部冻结，没有优化器或权重更新。
每个shot独立子进程，完成并退出后再清理/进入下一shot；中断停止，显示当前shot、已完成shot和结果位置。

## 四种局部输出

| 标签 | 输出 |
|---|---|
| baseline | 原A三适配器均值图＋ALIGN，再sigma4平滑 |
| G0_L0 | 新D自己的三适配器均值图＋ALIGN，再sigma4平滑，RMA不参与融合 |
| RMA_only | RMA独立local logits softmax异常通道；无ALIGN、无其他分支；另保存sigma4平滑版作诊断 |
| G0_L1 | 新D四分支等权，RMA25%＋ALIGN，再sigma4平滑；native全局项不使用RMA |

raw为平滑前；evaluated为按原sigma4平滑后的图。RMA_only没有ALIGN，所以不能把它的绝对值
与最终融合图当成同一种概率校准结果。重点用G0_L1相对G0_L0的配对变化判断加入RMA后的影响。
另评价G1_L1完整新D作历史回归端点，像素图与G0_L1相同，不额外缓存一套。

六项指标全部记录。原A、G0_L0、G0_L1图像分数沿用各自原生全局项与本条件平滑图最大值的调和融合；
RMA_only的辅助图像分数由RMA自身global softmax异常分数与其独立平滑图最大值调和融合。
RMA_only只是支路诊断，不是新完整算法，也不据此评定保留资格。

## 分清哪类错误

分别统计：正常图背景、异常图背景、GT缺陷区域。无缺陷区域不填写0分，而标N/A；
异常标签但预处理后GT缺陷为空会在逐图DefectPixels中显示0，不伪造缺陷像素。

- 逐图各区域：像素数、均值、标准差、P95/P99、最大值、Top1%均值。
- 固定分数切点0.25/0.5/0.75：背景高响应比例和缺陷低响应比例。
  这些只是诊断分数切点，不从目标集或源域搜索最优阈值，不解释为校准后概率或部署阈值。
- 前景背景间隔：缺陷均值减背景均值，以及缺陷均值减背景P99。越低表示越难分离，但不自动推断病因。
- 配对归因：G0_L0减原A，观察适配器权重变化；G0_L1减G0_L0，观察加入RMA图后的增量。
  逐像素记录新增/消除背景高响应、新增缺陷低响应/恢复缺陷的比例。
  paired_error_summary的NetErrorRateChange为正表示该切点下错误净增加，负为减少。

源域全部像素参与统计，不以固定展示样本代替全验证集。
默认每类选前2张正常、前2张异常作跨shot一致的可视化；不按本次错误高低挑展示样本。
`background_tail_ranking.csv`和`weak_defect_ranking.csv`可定位高背景响应、弱缺陷响应对应的完整图像路径。

## 路径和可追溯性

输出统一到 `E:\projects\FSAD\AdaptCLIP_old\results\rma_local_diagnostics\mvtec_seed10\`，重复运行自动编号。

- 每个 `<shot>shot_seed10/`：
  - region_stats.csv、region_summary.csv：逐图区域统计及每类别/All_images汇总。
  - foreground_background_gaps.csv、gap_summary.csv：缺陷背景间隔。
  - paired_errors.csv、paired_error_summary.csv：配对错误变化。
  - metrics_combined.csv：六指标；image_scores.csv：逐图分数与缺陷像素数。
  - references.json保存60张编码参考；effective_references.json保存实际first-S参考，顺序与原验证一致。
  - source_audit.json、sample_manifest.csv、endpoint_metric_checks.csv：来源身份和回归结果。
  - selected_samples.json、visualizations：固定展示样本名单和四图同一0至1色标对比。
- 根目录合并三个shot的指标、区域汇总、配对汇总和间隔汇总。

统计表的AverageImageP99/Top1Mean是逐图区域统计的等图像平均，不是全像素池化分位数；
PooledMean和PooledHighRate按像素数加权。All_images也不是等类别宏平均；每类别结果全部保留。
六项评价指标仍使用原tools.Evaluator和等类别宏平均，不被上述统计加权方式替换。

默认不新增样本矩阵。需要保存固定展示样本的raw/evaluated四图及GT：

```powershell
python .\models_test\rma_local_diagnostics.py --save_maps
```

不保存全验证集npy，不把全部图留在内存；评价最多缓存一个类别的四套唯一像素图。
每类别评价后清理空闲缓存；操作系统分页文件管理仍由环境决定。

## 验证及解释边界

复用新D原source_split：1380张训练查询不进入本轮；345张验证查询的顺序不变，60张参考在batch2编码后取各shot前缀。
检查原源域metadata/split及权重身份；G0_L0/G0_L1/G1_L1逐类别与均值六指标核对已完成固定比例源域结果。
默认历史目录为 `results/rma_fixed_fusion/six_datasets_seed10/calibration`，可用 `--source_history` 明确更换。

重要：原A训练时曾使用这些MVTec查询，新D则把它们作为验证留出。
原A在此处作为输出行为参照，不能把双方源域指标差直接解释为公平的泛化优劣。
该验证集此前也用于选epoch14，诊断结果不属于独立无偏评估。
本轮只观察源域，不在六个目标测试集上确定训练约束、阈值或新超参数；之后若补目标比例消融，仍需分开记录其探索性质。

本代码不自动宣布“背景误报”或“缺陷漏检”是唯一原因；需结合分区、配对统计和可视化再确定。
原train.py/test.py/adaptcliplib及已有实验代码不修改。
开发仅静态检查，tests/test_rma_local_diagnostics.py为必要逻辑验证，未自动运行；未启动推理或确认指标效果。
