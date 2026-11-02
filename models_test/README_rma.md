# RMA：冻结原模型，新增可训练参考匹配分支

## 实验九的两组训练设置 2026年11月2日补充

同一份 train_rma.py 通过 `--training_profile` 切换，默认 legacy_rma 保持历史实验九设置。
不复制训练脚本、不覆盖旧权重。新的 adaptclip 组仅对齐用户指定的七项训练设置，并非完整复现原训练：
仍加载旧 epoch_15，冻结 CLIP/VA/TA/PQA，只训练新 RMA；保留原实验九的源域划分、融合监督、梯度裁剪、验证选优和测试流程。

| 设置 | legacy_rma 历史实验九 | adaptclip 参数对齐组 |
|---|---|---|
| 每查询训练参考数 | 每小批轮换1/2/4 | 每查询随机1张 |
| 实际batch | 2 | 8 |
| 梯度累积 | 4批 | 1批 即不累积 |
| 优化器 | AdamW | Adam |
| 学习率 | 1e-4 | 1e-3 |
| betas | 0.9,0.999 | 0.5,0.999 |
| weight_decay | 1e-4 | 0 |

训练参考数通过独立 `--train_reference_shots` 管理；测试 `--shots` 与源域验证1/2/4-shot均不改变它。
单参考用 random.Random.choice，多参考按原历史代码用 sample；同seed不保证跨batch设置得到相同训练episode。
两组源域划分与验证参考身份固定不变。目标六数据集的历史参考名单和评价协议也不变。
`profile_overrides` 会记录显式参数覆盖；例如改成batch4后不能声称七项全部与官方一致。

在项目根目录运行新的训练、保存每轮权重、测试18个条件并自动比较历史B：

```powershell
python .\models_test\rma_pipeline.py --training_profile adaptclip --save_every_epoch --comparison_run .\results\test_rma\pipeline_20261030_210844_733256
```

本次只使用B专用流水线，不需要运行会接着训练C的 run_rma_pipelines.py。
这里“一次训练”为默认15个epoch；若明确只试1个epoch，可额外 --epochs 1，但其训练预算与旧15轮不相等。

单独训练或复用历史设置：

```powershell
python .\models_test\train_rma.py --training_profile adaptclip --save_every_epoch
python .\models_test\train_rma.py --training_profile legacy_rma
```

新流水线/训练/测试目录名带 adaptclip 前缀，均仍在各入口同名 results 目录内，重复运行创建新会话。
`best.pth`按原实验九的源域验证选优；每轮权重只作记录，不根据目标测试指标挑轮次。
新测试目录保留新B减原A的差值；流水线会话的 `training_setting_comparison/` 保存新B减旧B的六项差值、逐类别差值与每shot保留判断。
对比前核对源域划分、冻结初始化权重、模型结构、目标协议、参考与查询身份。七项同时改变，不能据此归因到某一个参数。
若终端已显示 Evaluation completed，而之后在比较汇总处报错，先核对训练和测试各自的 run_status.json；
不应直接重新运行整条流水线。2026-11-02 已修复汇总错误读取 train_options.seed 的问题，seed 实际存放于 model_manifest.json 顶层。
本次现有结果的离线补汇总及状态说明见 results/rma_pipeline/adaptclip_six_datasets_seed10/postprocessing_recovery.json；
原 failed 状态保留为历史记录，训练、测试和权重没有重跑或覆盖。
新组尚未实际训练，batch8峰值显存未实测；若不足，应明确记录实际覆盖值，不自动降batch冒充官方配置。

这是待实跑的实验候选，不是已证实的提升，也不是完整复现 RAID/DictAS。
原始 `train.py`、`test.py`、三个适配器源码及所有历史实验均不修改。

## 根目录直接运行

```powershell
cd E:\projects\FSAD\AdaptCLIP_old
python .\models_test\rma_pipeline.py
```

先训练一份通用 RMA 权重；训练进程成功退出后，自动加载该次训练的 `best.pth`。
按 Visa→BTAD→MPDD→SDD→DTD→part_glass 顺序，每个数据集依次运行 1/2/4-shot、seed=10，共18个测试子进程。
每次等待当前子进程完成、退出，才启动下一个。进程退出释放该进程所有 CPU/GPU 资源；
`gc.collect()`/`empty_cache()`仅辅助释放未使用分配，不承诺操作系统缓存或分页文件立即缩小。
失败或中断则停止后续运行，不把未完成结果标为完成，也不自动覆盖已有会话。

单独训练及测试：

```powershell
python .\models_test\train_rma.py
python .\models_test\test_rma.py --checkpoint .\results\train_rma\mvtec_seed10\best.pth
```

独立训练重复运行会产生 `_run02` 等新目录；请使用终端显示的实际 `best.pth` 路径。
测试不猜测“最近”的权重，必须显式指定。
流水线可直接接收训练参数，例如 `--epochs 15 --batch_size 2 --accumulation 4 --learning_rate 0.0001`。
`--datasets Visa --shots 1` 可显式限制最终目标测试；源域选优仍固定使用1/2/4-shot。
训练参数见 `train_rma.py` 参数定义，或用户手动运行 `rma_pipeline.py --train_help`。

## 模型与训练

保留预训练 CLIP、VA/TA/PQA 和原 ALIGN，显式冻结参数并保持 eval，检查参数版本及所有注册 buffer 不变。
不重训 CLIP，也不联合更新旧适配器。优化器只包含新 RMA。
旧权重默认 `checkpoint/mvtec_three_adapters_ctx_seed10/epoch_15.pth`，不覆盖。

RMA 输入当前四层768维投影特征；冻结余弦特征找 Top-8 正常 patch，
学习 Q/K/V 投影、候选权重、关系空间编码、像素和图像头。四层共享参数、输出 logits 等权平均。
支持可单独重新训练的 `--matching top1/uniform/learned`，默认 learned；
三种匹配应分别用相同预算训练，不能直接把 learned 权重切换模式当成公平消融。

源域使用 MVTec。原 Dataset 的 meta.train 提供正常参考，meta.test 提供有正常/异常标注的源域查询。
新增 RMA 将这些查询按类别、正常/异常分层固定划分80%训练、20%验证（小样本至少各一），seed=10。
保存所有身份到 `source_split.json`；验证查询不参与新增 RMA 的梯度训练，正常支持与查询禁止交叠。
**旧三个适配器此前已经用源域查询训练；这里的验证是对新增 RMA 留出的源域验证，
不是整套旧模型从未见过的数据，也不等同于目标域独立泛化结果。**
源域训练每批轮换1/2/4张正常支持，从源域正常训练池独立采样并记录。
验证使用固定嵌套正常支持，每类前1/2/4张，同一验证查询，每轮不重抽。
六个目标数据集图像/标签不参与训练、选轮或调参；流水线前置读取它们的历史配置和身份仅用于审计。

默认15轮、batch=2、梯度累积4、AdamW lr=1e-4；实际峰值显存与耗时尚未实测。
CLIP 冻结但仍须为查询和参考执行编码；不缓存整个训练集特征。
验证参考特征缓存到 CPU，评估仅保留一个类别的预测图，仍会占用一定主存。

损失：RMA 图像交叉熵 + RMA 像素 Focal/Dice（正常和异常） + 完整融合图 Focal/Dice。
最后一项默认权重1，可通过 `--fusion_weight` 在源域控制实验中修改，不能用目标测试集筛选。
融合监督针对高斯平滑前的最终概率图，避免通过不支持梯度的 SciPy 平滑反传；
选优和正式测试均执行原始平滑及图像打分。
旧适配器每批参与前向，冻结部分不重复优化常数损失。
新日志的损失组成与原训练不同，不能直接用两次损失绝对值比较算法好坏。

## 完整融合和最好权重

新局部图：VA/TA/PQA/RMA四项等权平均，再与原 ALIGN 按原谐波公式融合。
新图像分数：四个原生图像异常概率平均，再与平滑最终图最大值按原谐波公式融合。
对照分数/图直接走原 Baseline 路径。`test_rma.py --disable_rma` 让候选直接恢复原三分支结果，
不保留四分支分母。该开关需仍指定兼容 RMA 权重。

每轮用完整“旧三适配器+RMA”的源域验证六项指标选优：各shot内类别等权宏平均，
六指标等权平均，再对1/2/4-shot等权平均。严格更好才更新 `best.pth`，平分保留更早轮次。
不按最低训练损失、RMA单支路指标或目标数据集结果选轮。
`last.pth` 保存最后 RMA/优化器及部分随机状态，供后续恢复开发；**当前没有续训入口**。
`--save_every_epoch` 可另外保存各轮 RMA。
这些文件保存新增模块和原模型依赖信息，不重复存整份冻结 CLIP/旧三适配器；
部署需保留原 CLIP 权重及旧三适配器权重。测试校验二者内容哈希和模型协议，哈希不是 Git 提交号。

## 输出与核对

- `results/train_rma/<独立会话>/`：`best.pth`、`last.pth`、`source_split.json`、
  `training_history.csv`（每轮均值）、`training_steps.csv`、`training_episodes.csv`、
  `validation_metrics.csv`、每轮/每shot/每类别验证指标、日志和损失/验证曲线。
- `results/test_rma/<独立会话>/<数据集>/<shot>shot_seed10/`：完整与原 Baseline 的六项指标、
  每图分数、固定正常参考清单、查询清单、历史 Baseline 指标/分数/固定样本矩阵一致性核对、日志、运行状态。
  默认保存每类固定正常/异常各一张对照图；只有 `--save_maps` 才额外保存这些样本的矩阵。
- 测试会话根：`metrics_summary.csv`、`metrics_by_category.csv`、`deltas_vs_original.csv`、
  `retention_by_shot.csv`。每个shot单独按六数据集六指标净差规则判断；数据集不齐不标为可保留。
- `results/rma_pipeline/<独立会话>/`：训练/测试实际路径、执行计划和状态。

沿用历史 branch_diagnostics 和 dual_output 的精确参考、协议及原始 Evaluator；
AP 仍为项目梯形 PR 面积，非另换 sklearn AP。原始 Baseline 不一致则报错，保留失败文件供排查。
结果路径均基于项目根目录；也支持从 models_test 直接运行。未修改系统临时目录或 Windows 分页文件位置。

## 开发验证状态

新增 `tests/test_rma_contracts.py` 供后续显式执行：梯度隔离、参考依赖、冻结 buffer 防护、源域选优口径、
源域划分/固定支持可重复及冻结权重兼容性。
开发阶段只做静态检查，不自动运行这些测试、不加载模型、不启动训练或18轮推理。
实际训练收敛、显存、指标收益和历史一致性需用户运行后确认。

本次静态检查：六个新增 Python 文件均通过 AST、内存编译及 Ruff F/E9 检查。
只读核对当前 MVTec 元数据为15类、3629张源域正常支持、1725张带标注查询；
默认划分计划为1380张 RMA 训练查询与345张源域验证查询。未执行实际数据加载/模型前向验证。
原训练15轮损失整理见 `results/training_loss_review/20261030/`。
