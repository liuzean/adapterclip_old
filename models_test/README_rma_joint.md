# C：冻结 CLIP，联合训练 VA、TA、PQA、RMA

本轮是独立实验候选。只新增文件；原 train.py/test.py、原三个适配器源码、B方案脚本及已有结果均保持原样。
没有执行训练、模型推理或自动化数值测试；收益尚待真实运行。

## 在项目根目录运行

```powershell
cd E:\projects\FSAD\AdaptCLIP_old
python .\models_test\rma_joint_pipeline.py
```

流水线先运行 `train_rma_joint.py`，成功退出后取本次训练明确返回的 `best.pth`，
再运行 `test_rma_joint.py`。不需要 B 先完成，也不读取 B 训练后的 RMA。
默认 Visa→BTAD→MPDD→SDD→DTD→part_glass，每个数据集按1→2→4-shot、seed=10测试，共18轮。
每轮为独立子进程；执行缓存清理并等待进程完全退出后才开始下一轮，释放该进程全部 CPU/GPU 资源。
操作系统文件缓存/分页文件不保证立即缩小；没有修改系统设置。
失败/中断停止后续任务，保留已生成结果。每次启动分配新会话目录，不覆盖旧训练/测试会话。

单独运行：

```powershell
python .\models_test\train_rma_joint.py
python .\models_test\test_rma_joint.py --checkpoint .\results\train_rma_joint\mvtec_seed10\best.pth
```

重复训练会得到 `_run02` 等目录，测试请指定终端打印的实际路径。
从 models_test 目录也能直接执行脚本；数据、权重、输出路径以项目根目录解析。

## C 与 B 的区别

两者都从原始 `checkpoint/mvtec_three_adapters_ctx_seed10/epoch_15.pth` 加载 VA/TA/PQA，
RMA 均从随机初始化开始；主干权重不更新。C 让四个适配器全部训练，B 只训练 RMA。
C 的旧适配器是独立副本，原始 A 的三个适配器只用于对照，保持冻结/eval，不加入优化器。
CLIP 只有一份，共享冻结的查询与正常参考图像特征；没有额外目标图像、PCA、1024接口、RAID或层加权模块。

训练时只在 CLIP 图像编码上使用 no_grad；每批重新生成 TA 可学习文本特征，保留提示向量经冻结文本编码器反传的路径。
四个适配器都保持 train，VA/PQA 的 BN 统计会更新；验证/测试统一 eval。
运行时核对四个模块梯度非零且有限、优化器参数恰好属于四个适配器，并检查冻结 A/CLIP 的参数版本与 buffer 不变。

默认15轮、batch_size=2、累积4批；RMA学习率1e-4，旧三适配器1e-5，AdamW weight_decay=1e-4。
累积只改变梯度更新频率，BN 看到的仍是实际小批量。尾批若只剩一张，合入前一批，最多 batch_size+1，避免 BN 报错且不漏样本。
这些是首版预设，未在目标测试集调参，资源峰值尚未实测。
可以显式改参数，例如：

```powershell
python .\models_test\rma_joint_pipeline.py --epochs 15 --adapter_learning_rate 0.00001 --learning_rate 0.0001
```

训练选项可通过用户手动执行 `rma_joint_pipeline.py --train_help` 查看。

## 数据、损失和选优

复用 B 的 source_split/source_spec，固定 seed=10 和默认80/20分层划分。
MVTec meta.train 提供正常参考；meta.test 的带标注源域查询分成新增训练/验证，正常参考与查询身份不交叠。
默认当前数据为1380张训练查询、345张验证查询；逐批轮换1/2/4张源域支持，并记录实际参考身份。
源域验证查询/嵌套1/2/4张参考每轮固定，与 B 相同规则生成。
旧三个适配器原先已用过 MVTec 源域查询，所以这不是整套原模型从未见过的验证数据。
六个目标数据集只做最终评价；前置读取历史元数据/结果用于身份和协议审计，不用来训练或选轮。

C 保留原 train.py 对 VA、TA、PQA 的图像交叉熵和像素 Focal+双 Dice 监督，另外添加 RMA 的同类监督及完整融合图监督。
PQA 四层各自损失相加，保留原训练口径，因此其四层损失并不是先平均成一项。
默认完整融合图监督权重1（--fusion_weight）；监督高斯平滑前的最终概率图。
训练中两输入谐波融合采用等价的 2ab/(a+b) 形式，避免零 ALIGN 导致反传数值问题；正式评价复用原融合/平滑/图像评分。
日志保存每分支 CE、Focal、两项 Dice，以及合计 global_loss/local_loss/fusion_loss/total_loss。
与 B 相比，C 同时新增旧三分支的梯度监督，损失绝对值不能直接相互比较。

每轮验证完整 C：四局部分支等权平均后与原 ALIGN 谐波融合；四原生全局概率平均后与平滑图最大值谐波融合。
六项原始指标均保留。选优为每个shot类别等权宏平均→六指标平均→1/2/4-shot平均，严格更好才更新 best，平分留早轮。
选择只用源域验证，不按训练损失或目标测试指标选 best。

`best.pth` 包含 VA/TA/PQA/RMA 四个状态、模型配置、原初始化权重/CLIP校验和及选优信息。
`last.pth` 另外保存优化器和部分随机状态，当前没有续训入口。`--save_every_epoch` 可保存每轮完整四适配器状态。
不重复保存整份冻结 CLIP；正式测试仍需保留原 CLIP 与原 A 权重，后者用于独立对照。
B 与 C checkpoint 格式分开，不能混用测试入口。

## 测试输出与解释

同一查询编码和同一正常参考下分别计算 `baseline`（原始 A）和 `C_joint`（联合训练完整模型）。
原始 A 不加载 C 的适配器状态；逐图分数、固定样本矩阵及全精度六指标与历史结果核对，不一致报错。
沿用历史原始 Evaluator，AP仍为项目梯形PR面积，不换评价口径。

- 训练：`results/train_rma_joint/<独立会话>/`。包含 best/last、源域划分、实际训练参考、逐步/每轮损失、验证六指标和曲线。
- 测试：`results/test_rma_joint/<独立会话>/<Dataset>/<shot>shot_seed10/`。包含两条件六指标、逐图分数、参考/查询清单、一致性核对、日志。
- 测试会话根：metrics_summary.csv、metrics_by_category.csv、deltas_vs_original.csv、retention_by_shot.csv。
- 流水线：`results/rma_joint_pipeline/<独立会话>/`。记录训练/测试实际路径、选定权重和执行状态。

所有这些目录均在项目根目录 results 下的 E 盘。默认输出每类正常/异常各一张固定样本图；不保存新的矩阵，除非显式 --save_maps。
固定样本图用于解释个案；有效性仍由完整测试集六指标判断。各shot分别按六数据集净差规则汇总，不将三个shot混在一起判断。

可选 `test_rma_joint.py --disable_rma` 输出 `C_joint_without_RMA`：保留联合更新后的三适配器，仅移除 RMA。
这不能恢复原始 A；A 始终由另外的原权重路径提供。
它也不能代替“只继续训练三适配器”的独立训练对照，因为它的三个适配器训练时已经与 RMA 配合过。

## 验证边界

新增 tests/test_rma_joint_contracts.py：覆盖训练尾批不漏样本、四适配器梯度、冻结文本编码器的提示梯度、
原 A 参数隔离、零 ALIGN 数值问题及拒绝混用 B/C 权重。仅编写，未自动执行。
开发只做 AST/内存编译、Ruff 与调用链静态审查，不运行 --help、pytest、训练、推理或小模型数值验证。
数据实际加载、显存、训练收敛、最好轮次和目标指标均待用户运行后确认。
