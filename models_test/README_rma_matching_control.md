# 实验I 同Top8候选的固定余弦加权对照

目的：检验当前RMA的可学习候选权重是否带来增量，不追加其他模块，也不保证指标提高。

| 条件 | 冻结CLIP的Top8检索 | Top8内部权重 |
|---|---|---|
| 已完成新D/H | 相同 | softmax(cosine + QK/sqrt(d)) |
| 新对照I | 相同 | softmax(cosine) |

Q投影还用于关系特征，保留训练；V、关系卷积、全局/局部头以及VA/TA/PQA仍从头共同训练。仅用于已移除加权项的K投影保留checkpoint布局，冻结且不进入优化器。有效训练参数减少属于被消融项，不能称为完全相同有效参数量。

## 一次运行训练和测试

项目根目录执行：

```powershell
python .\models_test\run_rma_matching_control.py
```

也支持在models_test内执行`python .\run_rma_matching_control.py`。启动前检查源域、新D训练记录及全部目标参考和历史H/A指标；先训练，再自动使用本次实际目录的best.pth测试。不会猜测最新目录或误用旧best。

独立运行：

```powershell
python .\models_test\train_rma_cosine.py
python .\models_test\test_rma_cosine.py --checkpoint .\results\train_rma_cosine\mvtec_seed10\best.pth
```

如训练目录有_run02编号，测试命令应替换为本次打印路径。总启动器自动传递正确路径。

## 公平条件

- 对照从CLIP预训练开始，四适配器全新初始化，不从H第14轮继续训练。
- 匹配实际新D训练的源域、CLIP、原A身份、80/20划分、查询、正常参考池及每一条有序episode。任何不一致停止，不静默重新采样。
- 固定seed10、参考数轮换1/2/4、batch2/累积4、AdamW、lr1e-4、betas(.9,.999)、weight_decay1e-4、融合损失1、梯度裁剪1、15轮预算。
- 沿用新D的完整四分支源域六指标与三个shot平均选轮规则；对照独立选best，可能不是第14轮。每轮权重、best和last及训练曲线继续保存。
- 目标测试固定H公式0.75SE+0.25G0；定位12.5%→ALIGN→sigma4。SE/G0和图都来自对照自己的新权重，不直接拿历史H分数当预测。
- 历史H是直接比较对象，原A是共同基线。使用已经验证的H参考图身份、顺序及同一测试查询；历史标签仅用于身份核对和评价。
- 默认Visa→BTAD→MPDD→SDD→DTD→part_glass，各1→2→4-shot，共18条件，只报告原六指标。

## 缓存 进度与中断

训练每epoch清理未使用缓存，保留训练所需模型及优化器；不会在epoch间清空权重。训练进程结束后才启动测试。测试每dataset/shot独立进程，等退出释放内存/显存后继续。没有新增固定样本矩阵或全测试像素缓存；指标缓冲按类别释放。

这里一轮指一个数据集的一个shot，例如BTAD 1-shot结束并清理后才启动BTAD 2-shot。异常退出也在finally中执行缓存清理，失败后停止后续条件。

中断提示、停止摘要和stop_progress只记录：训练时epoch、batch；测试时数据集、shot。训练逐batch原子写入training_progress.json；正常运行的内部状态仍保留校验、续测所需信息，不在中断摘要中展开。

进度JSON使用独立临时文件及有界重试，避免Windows读取/扫描时的短暂共享冲突。训练与测试的显示进度写入失败只警告一次、保留上次记录并继续，持续失败会短暂降低尝试频率；不会为进度文件报错丢掉训练。配置、正式结果和完成标记仍属于关键写入，重试后仍失败会停止，不能把未保存的结果标为完成。

可捕获错误或Ctrl+C会打印并保存总目录的stop_summary.txt、run_status.json中的stop_progress，以及各次stop_records/记录。突然断电/强制终止不能保证执行停止处理，但可用--status读取最后一次已经写入的进度；不声称知道断电的精确时刻或原因。进度记录不等于训练batch断点恢复。

只读查询总进度，不启动模型：

```powershell
python .\models_test\run_rma_matching_control.py --status .\results\run_rma_matching_control\I_cosine_seed10
```

训练完整完成后、测试中断，可恢复同一序列，复用这次训练及已验证测试轮次：

```powershell
python .\models_test\run_rma_matching_control.py --resume_from .\results\run_rma_matching_control\I_cosine_seed10
```

也可对test_rma_cosine单独使用`--resume_from <实际测试目录>`。恢复会检查代码/输入/权重及完成输出指纹；未完成条件另建编号目录，保留旧文件。总序列`--resume_from`用于训练完整完成后的测试续跑；训练未完成请使用下面的训练恢复参数。

### 从第13轮继续并自动测试

当前失败目录已完整保存前12轮。执行：

```powershell
python .\models_test\run_rma_matching_control.py --resume_training_from .\results\train_rma_cosine\mvtec_seed10
```

新建编号训练及总记录目录，复制已完成的检查点、best/last和完整历史，不修改原失败目录。恢复第12轮四适配器及AdamW优化器，第13轮重新完整执行，训练到15轮；将前12轮与13至15轮共同按原源域规则选best，之后自动测试六集各1/2/4-shot。最优轮次仍可能是第10轮，不强制选择新训练的三轮。

旧last没有随机数状态：程序仅对已知原实现、workers0、确定性变换和无活动dropout的情况做CPU元数据采样回放，逐条核对原查询、shot及正常参考，恢复采样器和参考RNG的前进位置。不会重复前12轮模型前向、反向或优化。数值代码/参考或采样回放不一致立即停止，不跳过核对。是否与连续训练逐位一致尚未实测，不作此承诺。

后续每轮last.pth在epoch结束时保存Python、NumPy、Torch CPU/CUDA及参考采样随机状态，之后优先直接恢复这些状态。仅支持完整epoch边界，不从中断batch恢复。新目录保存resume_source.json及resume_audit.json，记录来源、原文件指纹、恢复方式和未验证边界。原先代码指纹不篡改为新值。

## 输出

- `results/train_rma_cosine/mvtec_seed10/`：epoch权重、best/last、训练日志、曲线、源域验证六指标、逐查询参考episode、matching_control_audit.json。
- `results/test_rma_cosine/I_cosine_seed10/`：各dataset/shot完整六指标、逐图分数、参考/查询清单、H/A对照、六指标差值、图像/像素净和、分shot保留判定。
- `results/run_rma_matching_control/I_cosine_seed10/`：两阶段状态、实际子任务目录和本次best身份。

重跑自动编号，不覆盖。每shot仍要求至少3/6数据集六指标净正、总净和正；完整三个shot中至少两个通过才判可保留。部分条件不能宣称完整达标。差值方向为I减H/A，负值有利于原学习匹配；不能强行将消融结果写成正贡献。

为直接检查学习项贡献，另外保存learned_H_minus_control_deltas.csv与learned_H_retention_by_shot.csv，反向报告H减I及同规则判断；正值有利于学习匹配。不能仅因新对照没有通过，就宣称学习匹配已经通过了反向保留规则。

## 修改范围与验证边界

新增训练入口、测试入口、两阶段启动器及纯数据检查工具，测试放在tests/test_rma_matching_control.py。复用既有训练/损失/选轮/指标/H预测逻辑。

三个实验共享文件作小改动：rma_model新增cosine分支（原learned公式保留）；rma_scratch_common增加显式匹配对照与冻结参数过滤；h_inference允许对照显式提供自己best身份，默认H仍锁原第14轮及learned。原test.py、train.py、adaptcliplib不改。

共享文件指纹已变化，历史完成结果不修改；旧任务不能在代码变化后强行跳过指纹续跑。必要时新启动相应入口。回退可撤销本次三个共享文件改动及新增文件，不改已有权重或结果。

本次通过8个Python文件的AST检查及Ruff语法/未定义变量检查；原learned权重分支与H预测函数的AST未变。已核对新D源域1380/345划分、20700条历史episode、CLIP/新D权重身份和18条件H/A指标及参考清单指纹。开发记录在results/run_rma_matching_control/development_20261107/static_check.json。

开发交付时仅编写数值、梯度、同候选集和不完整条件判定测试，没有自动启动pytest、训练或GPU推理。之后用户完成的实际训练/测试状态见下节；中断故障注入、续训及跨seed稳定性不能由正常完成自动推定已验证。

## 2026年11月9日实际结果

清理原中断结果后，本次从头完成15轮，未使用续训；全部20700条源域查询/参考episode与新D一致。独立源域验证选出第14轮，综合分89.919049，高于学习匹配新D的89.698437，但目标域总体落后。

六数据集18个条件、15408次查询完成。108项输出指纹、1476项历史H/A六指标核对通过；逐图固定H公式最大误差约4.47e-8。标准库独立重算210项图像AUROC/F1，最大差约0.00001033个百分点。AP与像素指标采用实际运行及原评价器口径，不声称独立重新计算全像素数组。

| 比较方向 | 1-shot净正数据集/净和 | 2-shot净正数据集/净和 | 4-shot净正数据集/净和 | 达标shot |
|---|---|---|---|---|
| I-H | 2/6，-7.494696 | 2/6，-10.101759 | 2/6，-11.188807 | 0/3 |
| I-A | 3/6，+13.173779 | 2/6，+15.537485 | 2/6，+3.178502 | 1/3 |
| H-I | 4/6，+7.494696 | 4/6，+10.101759 | 4/6，+11.188807 | 3/3 |

净和为六数据集六指标百分点差之和，不能解释成某一个AUROC提升。I不替换H，保留为结构消融；当前协议支持学习加权设计总体有价值。BTAD、SDD的I三个shot净正，VisA、MPDD、DTD、part_glass三个shot净负；不能说学习加权所有数据集都更好。SDD相对A的像素AP/F1仍下降，固定余弦没有解决该定位问题。

证据边界：四适配器共同训练，差异包含其共同适应；K冻结减少98432个有效训练参数，未验证真实匹配正确率、权重分布或单独的尺度/容量作用。只有seed10；不能把结果解释成所有配置下固定余弦均无效，也不声称统计显著。此次没有运行续训路径；恢复模块的RNG捕获用于正常epoch保存，但不等于旧检查点重建已实测。

完整报告、全部六指标差值和Git文件分类在results/test_rma_cosine/I_cosine_seed10/analysis_20261109/。后续继续固定H，将I加入论文关键消融；若继续算法实验，优先预定义一次同权重定位ALIGN有/无对照，固定H图像分数及其他步骤，不立即继续扩大RMA选参。该后续尚未生成或运行。
