# 精简H测试入口与E回退

此入口是H的代码整理，不是新实验I。原始test.py、train.py和adaptcliplib不变。

## 运行

项目根目录默认H：

```powershell
python .\models_test\test_h.py
```

切换E，只改变最终图像评分，定位路径不变：

```powershell
python .\models_test\test_h.py --mode E
```

进入models_test目录后，也可以运行`python .\test_h.py`或`python .\test_h.py --mode E`；根目录解析、子进程入口和文件指纹均基于实际脚本位置。

H：`SH=0.75SE+0.25G0`；E：直接使用SE。SE为G0_L1的完整最终分数，仍包含25%局部图最大值的贡献；G0为新D旧三分支原生全局均值。两种模式均使用E的12.5%局部融合＋ALIGN＋sigma4平滑定位图。

与rma_score_coordination_inference.py的关系：后者是H多条件一致性验证入口；本文件是相同H方法的精简单模式使用入口。默认只评价H，传mode E才评价E，不会一次汇总运行之前所有实验，也不构成下一轮新算法实验。目录及加载路径整理后的真实回归已于2026-11-07完成。

默认六集Visa→BTAD→MPDD→SDD→DTD→part_glass，每集1→2→4-shot、seed10；每条件独立子进程，退出清理后继续。无需重新训练或源域校准，不能通过参数改变lambda和局部比例。

## 加载与依赖

- 只加载CLIP预训练权重及新D第14轮四适配器best.pth，不再构建原A模型、也不加载原A的epoch15适配器。
- 使用`results/train_rma_scratch_new/mvtec_seed10/best.pth`及同目录model_manifest.json。固定权重SHA256校验防止误用其他训练版本；`--checkpoint`只允许同一权重文件的位置变化。
- 参考图和查询协议默认读取已经完成的`results/rma_score_coordination_inference/experiment_H_inference_seed10`。检查记录指纹、正常TRAIN参考身份、顺序、重复次数、shot和查询清单；不会重新随机抽参考。
- 复用官方数据集、变换、模型模块、参考编码、分支推理、融合和平滑工具，没有复制adaptcliplib或整个test.py。
- `models_test/h_inference.py`提供小型加载、预测及H/E评分选择接口。预测函数不读取历史分数或目标标签，历史只在入口中用于输入核对和输出验证。

## 验证状态

此前rma_score_coordination_inference.py完成18轮；2026-11-07的run_test_h_sequence又完成精简test_h的H/E各18轮，共36轮。30816项逐图历史评分与1476项历史六指标核对最大差均为0；H/E像素三指标按类别及Mean相同。直接加载四适配器及模式切换已通过本次真实回归。

默认继续开启历史核对：逐图分数容差1e-6、完整六指标容差0.001个百分点。历史完整像素数组未保存，不声称全部历史像素逐值重比。同条件重跑属于实现验证，不另计一次算法提升或独立泛化试验。

新入口回归通过后，若只需运行评估且不再保留历史分数/指标，可使用`--no-verify_history`；此时仍保留参考图与查询清单及指纹检查，结果不会宣称历史一致性已验证。没有完整历史像素数组，不声称新图与全部历史像素逐值比较。

## 保存

```text
results/test_h/H_seed10/
results/test_h/E_seed10/
```

重复运行自动编号，不覆盖结果。每轮保存完整六指标metrics_combined.csv、image_scores.csv、实际参考与查询、配置、状态；默认历史核对还保存score_checks.csv和metric_checks.csv。根目录metrics_summary.csv合并全部类别/Mean，不混入未完成尝试。没有默认图像、矩阵或全测试集文件缓存；评价缓冲只保留单类别的一份定位图。

## 进度与恢复

```powershell
python .\models_test\test_h.py --status .\results\test_h\H_seed10
python .\models_test\test_h.py --resume_from .\results\test_h\H_seed10
```

使用实际打印目录。status只读；resume恢复原参数，核对权重/代码/输入和完整输出指纹，跳过完整轮次，在新尝试目录重跑未完成条件，保留旧文件。失败输出数据集/shot、已完成项及目录，停止后续条件。

## 论文结果链

最新整理在`results/test_h/final_evidence_20261107/report.md`，核对精简入口后固定H，不新训练。dataset_metrics.csv保留108行数据集Mean六指标；all_category_metrics.csv保留738行类别/Mean；incremental_and_baseline_deltas.csv分别列出相对前一步和原A的变化；retention_by_transition.csv报告每shot判定。另保留权重/代码冻结清单、选参来源、主线省略步骤和未补缺口。代码内的简要说明见README_H_mainline.md。

## 本轮H实现验证的Git备注建议

```text
完成实验H真实推理一致性验证并保留评分协调候选

固定新D第14轮、lambda=0.25及E的12.5%定位路径。
完成六数据集1/2/4-shot、seed=10的18轮真实推理。
15408次查询的G0/SE/SH与离线分数差为0，H/E本次定位图逐像素相等；1476项六指标核对通过。
H相对E三个shot均满足保留规则；DTD及MPDD部分条件仍下降，保留全部代价。
```

本次同时提交代码整理时可说明“完成H/E独立入口36条件真实回归，历史逐图评分与六指标核对通过”；此前开发交付时的“精简入口尚未GPU回归”状态已由本次运行更新。
