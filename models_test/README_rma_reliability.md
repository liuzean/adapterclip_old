# RMA 局部图可靠性约束融合（待运行的候选）

在项目根目录运行：

```powershell
python .\models_test\rma_reliability.py
```

默认先做 **MVTec 验证集 1/2/4-shot 的三轮校准推理**，再锁定一个统一参数，
依次运行 Visa → BTAD → MPDD → SDD → DTD → part_glass，每数据集 1 → 2 → 4-shot。
全部固定 seed=10。每轮一个子进程，等待退出并清理空闲缓存，再启动下一轮。
校准不是训练：没有优化器、反向传播或新权重；CLIP、四个适配器全部冻结。
也支持在 models_test 内执行 `python .\rma_reliability.py`。

## 只改哪一步

保留新 D 同一份源域选出的第14轮 best.pth：
`results/train_rma_scratch_new/mvtec_seed10/best.pth`。
由已完成的新 D campaign 精确绑定路径、格式、epoch、文件哈希，不猜测最新目录。

| 输出 | 图像原生全局项 | 局部融合 |
| --- | --- | --- |
| baseline（原 A） | 原 A 三适配器均值 | 原 A 三分支图＋ALIGN＋平滑 |
| G0_L1（直接对照） | 新 D 三适配器均值 | 新 D 四分支等权，RMA固定25%＋ALIGN＋平滑 |
| G1_L1（完整新 D） | 新 D 四分支均值 | 与 G0_L1 完全相同的图 |
| G0_L1_reliable（唯一新候选） | 与 G0_L1 相同 | 根据参考匹配一致性，将RMA比例从25%向下调整＋ALIGN＋平滑 |

每个条件的最终图像分数都按原规则，由自己的原生全局项与自己的平滑后图最大值调和融合。
因此新候选的图像指标也可能变化；没有复用原 A 最终图像分数。
不改变 RMA 内部、PQA 层权重、ALIGN、平滑、训练损失或图像评分公式。

## 一个统一的可靠性规则

用原有四层投影后768维 CLIP 特征，在与 RMA 相同的余弦 Top-8 匹配集合上，
检查八个**正常参考 patch 特征是否彼此接近**。不是检查异常图值大小，也不使用查询标签。
八个单位向量 r 的分散度：

`d = K/(K-1) * (1 - ||mean(r)||²)`，K固定为8。

这是不同匹配特征间 `1-cos` 的均值。分散度越大，表示候选匹配越不一致。
四层分散度等权平均后，按RMA相同的双线性插值方式放大到518×518。

`RMA比例 alpha = 0.25 * exp(-beta * d)`

`融合图 = (1-alpha) * 三分支平均图 + alpha * RMA局部概率图`

之后沿用原 ALIGN 调和融合、sigma=4 高斯平滑。其他三个分支始终等比例。
beta=0 直接复用 G0_L1 原计算张量，避免舍入误差；最大比例始终25%。
1-shot 时也有一张参考图的多个 patch，可以计算 Top-8，一样不增加正常参考图。
这是**匹配候选的一致性代理**，不是概率校准后的置信度，也不是匹配一定正确的保证。
它不能自动解决所有参考覆盖不足；重复参考和相似背景也可能使一致性偏高，实际收益待验证。

为不侵入旧 RMA，额外做一次分块的四层余弦 Top-8 检索；查询 CLIP 只编码一次。
没有额外增广、骨干、目标图像、模块训练或数据集专用开关。

## 参数只在源域确定

1. 复用新 D 保存的 MVTec `source_split.json`：1380训练查询不参与此次校准，只评价原345验证查询。
2. 沿用原60张正常验证参考，参考编码/验证查询 batch=2、参考顺序及1/2/4-shot前缀切片全部不变。
3. 预先固定 beta 候选 `[0,4,16,64]`。同一查询推理计算四种强度，不重复编码四次。
4. 使用原 `tools.Evaluator` 的六项指标、类别宏平均，再对三种 shot 和六指标等权平均选最高值。
   这是预先约定的源域选择准则，并非只优化像素指标；目标结果不参与选择。
   1e-9百分点内视为数值持平，优先较小beta；完全持平首先选0。
5. 保存 `calibration/locked_config.json`，同一个beta用于全部目标数据集、类别和shot。
   若选0，就如实报告“源域没有支持降权”，目标候选恢复G0_L1；仍执行全部目标评价。

校准的完整新 D 输出还要与原第14轮源域验证的逐类别及均值六指标回归核对。
权重、源域metadata/split、实现文件或已锁定校准指标有变化，就拒绝复用校准。
同一验证集此前也用于选epoch14，所以此次源域选择并非独立的无偏效果估计；最终效果看完整目标评价。
目标 metadata/历史指标在源域选择锁定后才进入目标审计；不会按目标结果改beta。

只校准、不测试目标集：

```powershell
python .\models_test\rma_reliability.py --calibrate_only
```

复用同一版本已经完成的校准（例如校准完成后目标测试中断）：

```powershell
python .\models_test\rma_reliability.py --calibration .\results\rma_reliability\six_datasets_seed10\calibration
```

这会新建结果目录，重新做18轮目标评价，不是从中断shot续跑。原结果不覆盖。

## 输出和核对

全部写入项目E盘 `results/rma_reliability/six_datasets_seed10/`，重复运行自动编号。
不指定C盘结果/矩阵目录。操作系统分页文件或CUDA/Python自身缓存仍由环境管理，不能据此保证C盘用量不变。

- `calibration/locked_config.json`、`source_selection.csv`：选定的规则、强度、源域审计及选择成绩。
- `calibration/<shot>shot_seed10/`：四强度和完整新D的源域六指标、查询与参考记录、端点回归核对。
- `metrics_summary.csv`、`metrics_by_category.csv`：四条件的全部六指标。
- `deltas_vs_G0_L1.csv`：新候选减直接对照；同时输出 `deltas_vs_original_A.csv` 和 `deltas_vs_new_D.csv`。
- `retention_by_shot.csv`、`retention_overall.json`：三个对照分别报告，主要判定预先固定为 G0_L1。
  每shot至少3/6数据集六指标差值之和严格为正，且六数据集总净变化为正；至少两个shot通过才具备整体保留价值。
  不完整的六数据集/三个shot不会判为整体可保留；不混合不同对照的通过条件。
- 各 `数据集/<shot>shot_seed10/` 内保留逐图分数、逐图RMA比例均值/最小/最大值、分散度均值、
  六指标、固定参考和查询清单、历史分数/指标/原A选定样本矩阵核对、日志及完成状态。
- 默认保存历史选定样本的四条件对比图，不默认新增矩阵。加 `--save_maps` 才保存其异常图及可靠性/比例矩阵。

逐图核对原 A、G0_L1、完整新 D 的分数；核对它们的每类别及均值六指标。
参考身份、顺序、重复次数、查询身份、权重和原协议不一致时停止。候选自身结果不同不作为回归错误。
每类别评估后清理缓存，最多缓存三套目标像素图；全局项不同但图相同的G0_L1/G1_L1共享像素指标。
目标标签仅用于指标/结果记录，可靠性函数只接收查询特征、原正常参考特征。
中断/失败停止后续任务，终端和根 `run_status.json` 显示阶段、数据集、shot及已完成条件。

原 train.py、test.py、adaptcliplib 及旧实验入口都不修改。开发仅静态检查；
`tests/test_rma_reliability.py` 是公式、隔离、选择和保留规则的回归测试，不是算法效果实验，尚未自动执行。
