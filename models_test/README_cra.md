# 实验K CRA独立训练候选与J到K串行运行

2026年11月10日实际运行补充：CRA已从头适配15轮，按源域验证选择第9轮（89.705072；第0轮89.646809），六集18条件已完成。逐图H评分和738项历史H指标核对差为0，运行中冻结/梯度/零初始化检查通过。K相对H在2/4-shot通过、1-shot未通过，按至少两个shot规则可保留；相对原A三个shot通过。收益集中part_glass图像检测，六集平均像素三项三个shot均下降，SDD未修复。完整分析在results/test_cra/K_seed10/analysis_20261110/report.md。下面“开发交付”说明是最初交付时范围；实际推理状态以本段及本次结果为准，pytest和故意中断恢复故障测试仍未执行。

下一步为固定第9轮CRA的完整图像分数和第14轮H的完整定位图，补充双输出一致性验证，不重训；它不计入本次已经完成的K结果。

本实验只借鉴`Reference/TGSA/module.py`中的零门控3×3深度卷积残差设计，接在原PQA匹配差异表示`q+abs(q-p)`与每层sharebn之间。没有移植TGA、SPL、提示词或其他TGSA模块。通过独立的归一化前置钩子接入，原PQA源码不变。

## 训练与测试设计

加载H正在使用的新D第14轮四适配器权重，冻结CLIP、VA、TA、原PQA、RMA的参数及运行缓冲，**仅训练CRA的27,652个新参数**。这不是从头重训五个模块，也不是重新训练CLIP。

CRA初始门控为0，初始输出与原特征相同。会先完整验证MVTec的1/2/4-shot零初始化预测与CRA关闭时逐像素/逐图相同。新增模块放在原PQA共享的局部/全局预测头之前，所以它可能同时影响PQA的局部图和全局项。

| 项目 | 本实验设置 |
|---|---|
| 直接对照 | 原H：同第14轮权重、关闭CRA |
| 候选 | K_CRA：原H加训练后的CRA |
| 源域 | MVTec；复用新D的1380训练查询/345验证查询及正常参考池 |
| 训练采样 | 按新D保存的15轮20700条episode，逐批复用查询及参考，正常参考数量1/2/4轮换 |
| Batch Size/累积 | 2 / 4批，尾部不足4批按实际累积批数归一化 |
| 优化器 | AdamW；lr=1e-4，betas=(0.9,0.999)，weight_decay=1e-4 |
| 轮数/梯度裁剪 | 15轮仅CRA适配；梯度范数上限1 |
| 损失 | 沿用新D已有四分支分类、Focal/Dice及融合损失定义；旧模块冻结，只有CRA接入的PQA路径有可训练梯度 |
| 选权重 | 源域验证的类别宏平均六指标，再对1/2/4-shot等权平均；严格变好才替换最优，同分留较早轮次 |
| 第0轮 | 纳入候选；若训练未超过零改动源域分数，best可为0，并明确标为no_op_selected |
| 目标评价 | Visa→BTAD→MPDD→SDD→DTD→part_glass，每集1→2→4-shot，seed10 |

CRA训练有新的小权重文件。测试同时加载固定的H四适配器best.pth和本实验的CRA best.pth；不会用CRA的小权重替代H旧四模块权重。

目标测试保持H的完整公式：图像分数`SH=0.75SE+0.25G0`，SE仍由25%局部图参与原G0_L1完整评分；定位图仍是12.5%局部混合→ALIGN→sigma4平滑。CRA打开后用修改后的PQA输出按同一公式重新计算SE/G0，**不是把历史H图像分数直接拷给候选**。图像三指标因此允许变化。

H与K共享每批CLIP编码和冻结RMA输出，分别关闭/打开CRA计算旧三适配器输出。检查参考ALIGN不变、CRA确实作用于四层、共享特征未被原地修改。H关闭对照要通过历史逐图评分与完整六指标核对；旧模型参数和BN缓冲需始终保持冻结。

本实验始终保留ALIGN。即使先运行的J显示关闭ALIGN有收益，也不会自动将该改动叠加到CRA，避免混淆两项作用。

## 运行命令

项目根目录只运行CRA训练和测试：

```powershell
python .\models_test\cra_pipeline.py
```

按本次要求依次运行J，再运行CRA训练和测试：

```powershell
python .\models_test\run_align_cra_sequence.py
```

实际顺序：J六集18条件 → CRA零初始化源域验证 → CRA训练15轮并每轮源域验证/保存 → 自动加载本次CRA源域最优权重 → K六集18条件。父调度器不加载模型；各阶段独立进程，测试每数据集/shot独立进程并等待退出后继续，训练每epoch清理可释放缓存。训练中的模型/优化器会保留到该训练进程结束。

分开运行时：

```powershell
python .\models_test\train_cra.py
python .\models_test\test_cra.py --checkpoint .\results\train_cra\mvtec_seed10\best.pth
```

训练重复运行会编号；单独测试须用终端实际打印的best路径。pipeline会传递本次训练的确切路径，不会误选旧目录。以上入口也支持在models_test内直接运行，路径均基于项目根目录解析。

## 结果文件

```text
results/train_cra/mvtec_seed10/              # CRA训练与小权重
results/test_cra/K_seed10/                  # 六集完整测试
results/cra_pipeline/K_seed10/              # CRA训练→测试总记录
results/run_align_cra_sequence/J_then_K_seed10/ # J→CRA总记录
```

全部路径在当前E盘项目内。重复运行自动编号，不覆盖其他实验。自身训练中的best.pth、last.pth和训练进度按训练正常更新；各epoch/中断尝试的目录保留，不删除历史尝试。

- 训练：`training_history.csv`记录第0～15轮源域分数、全局/局部/总损失均值和四层有效门控；`epochs/`保存每轮权重及逐批损失，`validation/`保存每轮三个shot的H/K六指标和逐图分数；`best.pth`仅含CRA和身份信息，`last.pth`另含优化器/RNG/完整轮次历史。
- 测试：`metrics_summary.csv`、`metrics_by_category.csv`完整报告本次H/K及历史原A六指标；`deltas_vs_H.csv`为直接比较，另存`deltas_vs_original.csv`、类别差值、`retention_by_shot.csv`和完整核对文件。
- H是本次新推理的CRA关闭对照；原A仅引用已有已核对指标，不额外加载原A推理。没有历史全量像素矩阵，不声称与历史所有像素逐值重比。
- 不默认保存新的固定样本矩阵或热图，结果指标用全测试集，评价缓冲只保留单类别的两组预测图。

## 中断进度与恢复

失败会停止后续阶段。训练只需查看当前epoch/batch；测试查看数据集/shot。总串行入口可递归显示内部CRA训练/测试的位置：

```powershell
python .\models_test\run_align_cra_sequence.py --status .\results\run_align_cra_sequence\J_then_K_seed10
python .\models_test\run_align_cra_sequence.py --resume_from .\results\run_align_cra_sequence\J_then_K_seed10
```

只运行CRA的序列同理：

```powershell
python .\models_test\cra_pipeline.py --status .\results\cra_pipeline\K_seed10
python .\models_test\cra_pipeline.py --resume_from .\results\cra_pipeline\K_seed10
```

使用实际输出目录。完整阶段和测试条件核对指纹后跳过；测试未完成条件在新尝试目录从头推理。训练从last.pth最后完整保存的epoch继续，未完成epoch重跑；不是batch级训练恢复。没有完整训练轮次可恢复时，从零初始化验证开始，保留旧尝试目录。代码/输入身份改变则拒绝静默恢复。状态命令只读取记录，不判断进程是否仍然运行。

## 证据边界

此设计回答“固定H旧模块后，增加并训练CRA能否带来增量”，不等于同预算从头共同训练的结构对照。新增15轮只训练CRA，不能把结果写成原H训练预算完全相同。源域验证沿用新D已用过的划分，不是新的独立盲测；不会用六目标测试标签挑权重、融合参数或按数据集切换。

完整保留六指标和下降。以H为直接对照，每shot要求至少3/6数据集六指标净和严格为正，且六集总净和严格为正；至少两个shot通过才考虑保留。相对原A另报告。若best选中第0轮，表示本次源域准则没有支持已训练CRA优于零改动，不应描述为有效改进。

开发交付只做静态语法、接口、文件身份和预定义episode检查；不自动启动训练、测试或pytest。已核对1380/345源域查询、15轮20700条episode、每轮690批次及默认训练参数，记录在`results/cra_pipeline/development_check_20261109/report.json`，这不是本轮训练或测试结果。真实训练梯度、运行显存、源域最佳轮次和目标收益均需以用户实际运行结果为准。回归测试已编写在`tests/test_cra_candidate.py`，避免与实验测试入口test_cra.py同名。

## 文件职责

- `cra_model.py`：CRA残差模块和PQA低侵入接入/关闭接口。
- `cra_utils.py`：源域episode、权重身份、H/K共享推理与指标比较。
- `train_cra.py`：仅CRA训练、源域选权重、损失/门控记录及完整epoch恢复。
- `test_cra.py`：同参考六集1/2/4-shot测试、历史H核对及差值汇总。
- `cra_workflow.py`：两个实际流程共用的阶段调度、进度、完成核对和恢复。
- `cra_pipeline.py`：先训练CRA，再自动测试本次源域最优权重。
- `run_align_cra_sequence.py`：先J，再CRA训练和测试。
- `tests/test_cra_candidate.py`：零门控恒等、冻结梯度、关闭恢复、选权重和进度的CPU回归契约。
- 原`rma_align_ablation.py`只增加内部进度回执参数，支持串行程序准确记录J输出目录，预测公式不变。
