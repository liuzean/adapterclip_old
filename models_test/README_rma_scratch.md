# 从头训练四适配器：两个独立训练流程

两个训练入口和两个测试入口共享模型计算、损失及评价辅助代码。
没有复制原train.py整份脚本；后续修正公共模型逻辑时只改一处。
四个入口的文件开头均注明了流程和对照范围。

## 初始化与对照

两组均加载同一CLIP预训练权重，冻结CLIP；新建VA、TA、PQA、RMA，第一轮起全部共同训练。
没有将epoch15、B或C权重载入训练。baseline_checkpoint仅记录原A比较对象的身份哈希，不读取其参数。
测试时单独载入原A三个适配器，作为冻结对照；候选四适配器从本次检查点加载。

| 条件 | original 原流程组 | new 新流程组 |
|---|---|---|
| 训练查询 | 原Dataset的全部MVTec meta.test源域查询 | 同一源域查询按类别/标签固定划分80%训练、20%验证 |
| 正常参考池 | 原meta.train，要求全为正常图 | 原meta.train正常池 |
| 每查询训练参考数 | 原Dataset.__getitem__随机1张 | 每小批轮换1/2/4，逐查询无放回随机采样 |
| 实际batch / 累积 | 8 / 1 | 2 / 4 |
| 优化器 | Adam | AdamW |
| 学习率 | 四适配器统一1e-3 | 四适配器统一1e-4 |
| betas / weight_decay | 0.5,0.999 / 0 | 0.9,0.999 / 1e-4 |
| 监督 | 原VA/TA/PQA分支损失＋RMA全局/局部分支损失 | 同样四分支损失＋融合图监督，权重1 |
| 梯度裁剪 | 无 | norm1 |
| epoch / seed | 15 / 10 | 15 / 10 |
| 保存 | 每轮、last.pth、训练损失最低的best.pth | 每轮、last.pth、源域验证best.pth |
| 主测试权重 | 每轮global_loss均值＋local_loss均值最低的best.pth | 源域验证best.pth |
| 三适配器对照 | 直接使用已有原A，不重新训练 | 暂未安排同样新流程的三适配器，先看结果再决定 |

新组四适配器均从头学习，没有预训练好的旧适配器，因此统一lr=1e-4，
不沿用warm-start C给旧三适配器设置的1e-5；这是显式的初始化适配选择，不是已经验证的最佳超参数。
两组保持相同RMA结构：768维、hidden128、Top8、四层共享参数。
CLIP不更新，但TA学习提示仍通过冻结文本编码器反传。

原流程组沿用原Dataset、数据范围和随机1参考方式，保留原分支监督，并新增RMA自己的监督。
因此它研究的是“增加RMA及其必要训练/四分支融合”的整体效果；不是只改变一个数学算子的消融。
原train.py记录每轮global_loss和local_loss的小批均值、按轮次保存，但没有自动选优功能。
按用户要求，原流程组增加训练损失选优：每轮两个均值相加，严格小于历史最小值时更新best.pth；
损失完全相同时保留较早轮次。最后一轮仍保存到epoch_N.pth和last.pth，不强制作为主测试权重。
best_training_loss和best_epoch写入权重、运行状态和日志，validation_score保持为空。
这是训练损失最小权重，不代表源域验证最好或目标测试指标最好，不使用目标标签选轮。
已有原A日志中第15轮global_loss=1.7013、local_loss=3.2099，合计4.9112，是记录的15轮最低值；
因此原A对照仍使用其epoch15。第一组四适配器和原A三适配器的绝对损失不直接作优劣比较。
显式改epochs后应报告预算变化，不再声称与原15轮训练预算相同。
原流程正常参考由原Dataset内部随机抽取，不额外改其返回值；保存正常参考池和查询范围，
不声称记录了每条原流程训练episode的实际参考身份。新流程保存逐查询参考采样记录。

原流程默认num_workers4，保留原DataLoader行为；新流程num_workers0。
原流程若出现尾批只有1个查询则明确报错；新流程沿用FullBatchSampler合并尾部单例，不丢弃查询。
RMA初始化后恢复PyTorch随机状态，避免额外初始化直接改变后续查询shuffle随机状态；
不承诺四适配器训练与历史三适配器训练逐步数值一致。

## 从根目录分别运行

全部任务也可以用一个入口顺序运行：

```powershell
python .\models_test\run_rma_experiment_sequence.py
```

固定顺序：rma_dual_output.py --save_maps → 第一组训练 → 第一组best.pth测试 → 第二组训练 → 第二组best.pth测试。
两组训练都用默认15轮；三个测试活动均为六数据集、1/2/4-shot、seed10，共54个测试条件。
仅第一项默认保存选定样本矩阵；两个scratch测试保持各自原默认，不新增矩阵。
总入口通过子任务写入的准确会话路径读取完成状态和权重，不查找“最近文件夹”，不误用旧训练权重。
总记录保存在results/run_rma_experiment_sequence/dual_original_new_seed10/，重复运行创建编号目录。
各训练/测试结果仍保存至原入口同名结果目录。失败或中断停止，输出阶段、训练最近完整epoch或测试数据集/shot，
并写run_status.json、stop_summary.txt和各阶段进度回执。默认不自动续跑。

缓存清理：训练每epoch执行gc.collect和CUDA空闲缓存释放，保留继续训练所需的模型和优化器；
每个测试dataset/shot由独立子进程运行，等待退出后再进行下一条件；五个阶段也分别等待子进程退出。
不能在训练epoch之间清除模型或优化器状态，否则无法继续同一训练。操作系统文件缓存/分页文件不保证立即缩小。
该总入口只完成静态检查，未自动执行长时间实验。

### 显式从上次失败阶段继续

2026-11-03修复：共享scratch训练模块未将项目根目录加入sys.path，直接运行训练入口时会找不到adaptcliplib。
训练/测试共享模块现在都在导入项目模块前设置根目录，不依赖当前工作目录或PYTHONPATH。
同日补充修复：原模型类型识别函数接受字符串，初始化调用现在显式将预训练路径Path转为str，
避免把字符串替换误调用为Path.replace文件移动方法。权重路径、内容及训练设置均不改变，原train.py不修改。

若上次已完成双输出、第一组训练在初始化时报错，可运行：

```powershell
python .\models_test\run_rma_experiment_sequence.py --resume_from .\results\run_rma_experiment_sequence\dual_original_new_seed10
```

这会核对已完成阶段的回执、18条件状态、原A/双输出一致性及权重哈希，跳过双输出；
然后在新编号目录重新进行第一组训练，再继续第一组测试和第二组训练/测试。
旧失败会话、已完成双输出结果不覆盖；新的总记录写明resume_from及reused_stages。
先复用连续完成的阶段；未完成训练从头重跑，第一组未完成测试按下节的数据集级恢复规则处理。这不是训练epoch/优化器状态断点恢复。
例如已有失败的mvtec_seed10训练目录，会创建mvtec_seed10_run02；测试权重自动来自本次新训练。
不带--resume_from仍按原规则从第1项开始，重复运行会重做双输出。

### 重启后第一组从DTD继续测试

2026-11-03补充：第一组测试支持数据集级恢复，训练权重不重新训练。
例如本次第一组训练完成、前四数据集全部完成，DTD状态文件损坏，可直接运行：

```powershell
python .\models_test\run_rma_experiment_sequence.py --resume_from .\results\run_rma_experiment_sequence\dual_original_new_seed10_run03
```

总入口核对并复用旧B双输出和第一组训练；将上次original_test_progress.json中的准确测试目录传入第一组测试。
测试核对同一权重、flow、seed、协议、历史输入、查询/参考身份、六指标覆盖及记录检查，
复制连续完整数据集前缀到新的编号结果目录，再从首个不完整数据集运行全部shot。
本次复用Visa/BTAD/MPDD/SDD共12组，重跑DTD 1/2/4和part_glass 1/2/4共6组。
DTD 1-shot虽有指标，但其完成状态为全零字节，不作为已验收结果复用。
旧目录和损坏文件均保留，不覆盖、不改写；根目录的损坏汇总表不读取，使用各组指标重新汇总。
新会话同时包含复用的12组和新增6组，最终仍输出完整六数据集指标与各shot保留判断。
reuse_manifest.json记录源测试目录、复用条件以及从哪个数据集重跑。
每个实际补跑条件仍是独立子进程，等待退出和清理后再启动下一条件。
第一组测试完整结束后，继续第二组训练、测试。

单独进行第一组测试恢复也可使用：

```powershell
python .\models_test\test_rma_scratch_original.py --checkpoint .\results\train_rma_scratch_original\mvtec_seed10_run03\best.pth --resume_test_run .\results\test_rma_scratch_original\six_datasets_seed10
```

本次仍只做静态检查和已有文件核对，未自动开始补跑或执行pytest。

第一组训练：

```powershell
python .\models_test\train_rma_scratch_original.py
```

训练结束会输出实际测试权重路径。第一次默认会话对应：

```powershell
python .\models_test\test_rma_scratch_original.py --checkpoint .\results\train_rma_scratch_original\mvtec_seed10\best.pth
```

第二组训练：

```powershell
python .\models_test\train_rma_scratch_new.py
```

第二组测试：

```powershell
python .\models_test\test_rma_scratch_new.py --checkpoint .\results\train_rma_scratch_new\mvtec_seed10\best.pth
```

重复训练创建_run02等新会话；使用终端实际输出路径，不猜测最近权重。
测试入口只接受该流程已完成训练所声明的selected_checkpoint，拒绝B/C或另一流程权重。
也支持从models_test目录启动；权重、数据、结果均按项目根目录解析。

两组测试均默认按Visa、BTAD、MPDD、SDD、DTD、part_glass，每个数据集1/2/4-shot，seed10。
每条件单独子进程，等待退出再启动下一个。失败/中断停止后续任务，保留当前数据集/shot和已完成列表。
默认保存固定少量样本可视化，不新增npy矩阵；需要可额外--save_maps，仅保存选定样本。
目标测试参考和查询使用历史固定名单，原A逐图分数、六指标和选定样本矩阵均核对历史。

## 保存位置

- results/train_rma_scratch_original/mvtec_seed10/：每轮四适配器权重、训练日志、training_steps.csv、training_history.csv、配置、model_manifest.json、训练查询身份。
- results/train_rma_scratch_new/mvtec_seed10/：上述内容以及source_split.json、training_episodes.csv、源域验证指标和training_curves.png。
- results/test_rma_scratch_original/six_datasets_seed10/：原流程组六数据集测试结果。
- results/test_rma_scratch_new/six_datasets_seed10/：新流程组六数据集测试结果。

每组测试根目录均输出metrics_summary.csv、metrics_by_category.csv、deltas_vs_original.csv和retention_by_shot.csv。
完整保留I-AUROC、I-AP、I-F1max、P-AUROC、P-AP、P-F1max；逐shot分别判断，禁止混合shot挑最好权重。

## 解释范围

原流程组减A：在主要原训练条件下，增加RMA及其配套监督/融合后的效果。
新流程组减A：增加RMA并改变训练流程后的整体效果。
新组减原流程组：已有RMA条件下，两整套训练策略的差别，包含数据量、参考数、优化器、损失、选轮等变化。
缺少“新流程三适配器”时，不能独立分离新流程本身的作用或RMA与流程的交互作用。
本次不自动补该对照，也不自动启动任何训练或推理。

## 开发检查及回退

仅AST解析、内存编译、Ruff静态检查；新增tests/test_rma_scratch_contracts.py未执行。
未验证四适配器从头训练的显存、收敛、实际测试效果，不声称性能提升。
原train.py、test.py和adaptcliplib不修改。必要复用改动仅限：
MetricsBuffer支持可选conditions（默认原两条件）；test_rma_joint.inference支持可选验证器和候选名（默认旧C路径）。
回退时移除这些可选扩展并停止使用新入口即可；历史权重与结果不覆盖、不删除。
