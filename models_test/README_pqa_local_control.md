# 实验 M：PQA／冻结 CRA 的最小公平局部适配对照

2026年11月12日实际结果：外部准备与实验M串行流程均已完整结束。两臂各完成15轮、10350批、20700条查询/参考episode，初始化相同且仅训练130参数；源域像素三指标选优均为第15轮。六数据集18条件15408次查询完成，固定K图像分数、原H/K定位历史指标及四条件图像三指标核对差均为0。

相对当前K双输出，无CRA适配臂1/2/4-shot六集六指标净和分别为-1.232554/-0.500900/-0.042471，CRA适配臂为-6.304203/-4.937840/-3.088255，两组均0/3-shot通过保留规则；CRA臂相对同预算无CRA臂也0/3通过。两臂相对较早原A虽均3/3通过，但包含已有H/K收益，不能据此声称本次适配有效。保留本次否定性结果，不替换K双输出，不把“旧分类层不适配”写成已经证实的主要原因。

独立分析位于 `results/test_pqa_local_control/M_seed10/analysis_20261112/`。本次仅核对保存结果；没有重新训练/推理或执行pytest。官方SubspaceAD/DictAS仍未复现，外部准备只导出固定参考/查询及六指标协议。

本项与“外部评估准备”是两件独立工作。SubspaceAD、DictAS的官方复现由后续另行安排，不是本项训练的前置条件。实际 GPU 任务串行，两个工作流分别运行。

## 直接运行

在项目根目录运行：

```powershell
python .\models_test\prepare_paper_evaluation.py
python .\models_test\pqa_local_control_pipeline.py
```

第一条只导出外部对比所需的18个条件、正常参考和查询清单、指标协议、原A/H/K双输出历史六指标和空白外部表；不训练、不下载、不执行外部代码。输出至 `results/prepare_paper_evaluation/six_datasets_seed10/`，重复执行带运行编号。后续可用 `--external_csv <填好的表>` 收集数据，但导入表一律标为“同协议尚未验证”，不能把导入数值当成已核验复现。

第二条依次运行两次训练，然后完整测试六数据集。训练和测试代码分开：

```powershell
python .\models_test\train_pqa_local_control.py --arm plain
python .\models_test\train_pqa_local_control.py --arm cra
python .\models_test\test_pqa_local_control.py --plain_run .\results\train_pqa_local_control\plain_seed10 --cra_run .\results\train_pqa_local_control\cra_seed10
```

重复运行会创建编号目录，手工分步时应使用实际目录；串行入口通过回执自动传递目录。所有入口也支持在 `models_test` 内用脚本文件名运行；相对参数以项目根目录解析。

## 固定设置

| 项目 | plain臂 | cra臂 |
|---|---|---|
| 底座 | H／新D第14轮 | 相同 |
| CRA | 定位特征不经过CRA | 经过已训练的第9轮CRA，冻结 |
| 可训练参数 | 原共享64→2的1×1卷积副本，130参数 | 完全相同初始化的副本，130参数 |
| 其他模块 | CLIP、VA、TA、PQA其余层、RMA及所有BN冻结/eval | 相同，CRA也冻结 |
| 数据/参考 | MVTec原1380训练/345验证划分、15轮旧episode | 相同 |
| 预算 | 15轮、每轮690批、batch2、累积4、seed10 | 相同 |
| 优化器 | AdamW，lr=1e-4，betas=(.9,.999)，wd=1e-4，梯度裁剪1 | 相同 |
| 损失 | 最终12.5%混合、ALIGN后、Gaussian前的Focal＋异常Dice＋正常Dice | 相同 |
| 选优 | 源域验证像素三指标按类别宏平均，再按1/2/4-shot平均；含epoch0，平分留早轮 | 相同 |

没有新增PQA独立局部损失、全局损失或CRA损失。12.5%是混合图中RMA的占比，不是损失权重。训练不用Gaussian，验证/测试保持sigma=4。源域验证用于选轮，不作为独立泛化排名。

新分类层单独保存，不替换或写回旧PQA。四个PQA层共享同一分类层，所以是130个独立参数，不能按四份计为520。两臂都保留epoch0，可能选中0轮；这种情况报告为“没有超过不适配起点”，不强制选训练后的权重。

## 图像分数和定位图

图像分数真正固定：始终由H14＋冻结CRA9＋**原PQA分类层**计算K的完整最终图像分数，即 `.75*SE25+.25*G0`；新的分类层不会参与SE25的异常图最大值计算。

测试一次编码，共同参考，比较四种定位图，四种条件的图像分数使用同一张量：

| 输出条件 | 定位图 | 用途 |
|---|---|---|
| K_dual | 无CRA、原分类层，即H定位图 | 当前主候选直接基线 |
| K_original_local | CRA9、原分类层，即原K定位图 | 未适配CRA的起点 |
| M_plain | 无CRA、源域选出的新分类层 | 同训练预算对照 |
| M_cra | CRA9、源域选出的新分类层 | 唯一结构候选 |

定位全部采用12.5%RMA混合→ALIGN→sigma4。六数据集依次为 Visa→BTAD→MPDD→SDD→DTD→part_glass，每集依次1/2/4-shot。每个dataset/shot独立进程，进程退出后再启动下一轮，释放本轮模型、CPU张量和CUDA上下文；每类别评价后清理指标缓存。所有结果留在E盘项目results内，不修改SDD划分。

运行时核对：两臂初始化相同且优化器只有130参数；epoch0完整恢复各自原定位图；冻结参数和BN状态不变；首批分支聚合与原推理函数一致；目标逐样本K图像分数与历史一致(1e-6)，H/K定位历史指标一致(0.001pp)，四条件图像三指标相同。历史没有保存完整定位数组时只做历史指标核对，不声称比较了全部历史像素数组。

## 结果目录及中断

- `results/train_pqa_local_control/plain_seed10/` 和 `cra_seed10/`：`best.pth`、`last.pth`、每轮权重/损失、`training_history.csv`、源域三shot验证指标、进度和冻结检查。
- `results/test_pqa_local_control/M_seed10/`：18条件每类别六指标、图像分数、历史一致性核对、差值、六集汇总与逐shot保留判定；默认不保存大规模异常图/固定样本矩阵。
- `results/pqa_local_control_pipeline/M_seed10/`：总进度与实际子任务目录回执。

```powershell
python .\models_test\pqa_local_control_pipeline.py --status .\results\pqa_local_control_pipeline\M_seed10
python .\models_test\pqa_local_control_pipeline.py --resume_from .\results\pqa_local_control_pipeline\M_seed10
```

`--status`只读最后记录，不启动训练或判断进程是否还在运行。失败/中断停止后续并打印训练epoch/batch或测试dataset/shot。恢复时核验已完成阶段；训练恢复到最近完整epoch之后，未提交的半轮重新计算；恢复优化器和随机状态，不承诺CUDA跨进程逐比特一致。测试复用核验成功的条件，未完成条件在新编号目录重跑，不删除旧结果。不要让两个进程同时恢复同一个会话。

## 怎么解释结果

M_cra优于未适配K，只支持“局部适配可恢复部分定位能力”；还需要优于同预算M_plain，才有冻结CRA特征提供额外价值的证据。两臂都变好而相近，收益可能主要来自再次训练小分类层。负结果不能证明整个PQA/CRA都无效；130参数头的能力有限。

按每个shot分别计算六指标百分点差值，各数据集净和严格为正视为上升；至少3/6上升且六集总净和严格为正通过；至少两个shot通过才满足保留规则。完整保留下降项，同时报告相对K双输出、原K和原A的变化，CRA臂额外报告相对M_plain的变化。

本对照已实际完成，结果以本文件开头的2026年11月12日补充和独立分析为准。CPU回归测试位于 `tests/test_pqa_local_control_contracts.py`，用于逻辑和梯度连接验证，不用于判断算法提升；本轮未执行pytest，也未进行故意断电或文件锁故障注入。
