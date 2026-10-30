# 特征接口与邻域匹配四条件实验

独立实验，不修改原 `test.py`、编码器源码、Adapter 或权重；不需要本地 DictAS 包/字典权重。借鉴 DictAS 的空间平均思想，不是完整 DictAS 复现。

## 运行

项目根目录，已有 Adaptclip 环境：

```powershell
python .\models_test\context_alignment.py
```

默认 Visa→BTAD→MPDD→SDD→DTD→part_glass，每个数据集依次1→2→4-shot、seed=10，共18组，每组完整测试集。同一次编码器/适配器前向生成四条件，不是四次编码。每个shot独立进程，退出后再开始下一组。

也可以显式写出全部1/2/4-shot，效果与默认命令相同：

```powershell
python .\models_test\context_alignment.py --shots 1 2 4
```

可限定首次人工运行范围，但子集不能当作六数据集验证：

```powershell
python .\models_test\context_alignment.py --datasets Visa --shots 1
```

关闭所有候选改动，B/C/D的最终图及图像分数直接共享A，用于恢复检查：

```powershell
python .\models_test\context_alignment.py --datasets Visa --shots 1 --disable_context_alignment
```

`--dry_run` 只审计输入、列出计划，不加载模型/生成输出。以上命令均未由开发者执行。也支持在 models_test 内直接 `python .\context_alignment.py`。

## 四条件含义

| 条件 | 最终融合使用的ALIGN | 其余分支 |
|---|---|---|
| baseline（A） | 原768维单点匹配，完全复用原计算 | 原VA/TA/PQA及原全局分数 |
| B_768_context | 每层原768单点距离与768邻域距离等权平均 | 同A |
| C_1024_point | 1024维投影前特征的单点匹配 | 同A |
| D_1024_context | 每层1024单点距离与1024邻域距离等权平均 | 同A |

768来自原ln_post＋proj；1024来自同一DPAM Transformer输出，在ln_post/proj之前只读捕获。接口差异包含LayerNorm/投影，并非仅“维度大小”。所有条件的VA/TA/PQA仍只使用原768特征，PQA内部原匹配不变。

邻域处理固定为3×3空间平均，显式零填充、完整9元素分母。逐图去CLS、池化后逐token L2归一化；每张支持图先分别处理再展平，不跨图池化。原单点距离与邻域距离在每层等权混合，四层用原谐波平均，最后与原三分支平均图谐波融合。仍用原sigma=4、原图像评分公式和六指标。没有新增目标图、训练、PCA、层加权或按数据集换规则。

因此B/C/D的图像指标可以变化：虽然原生全局分数不变，最终异常图最大值会变化。每个条件分别评价，不能把新图只用于定位而偷偷继续引用A的图像指标。

## 输入、输出和保护

- VisA源：`results/branch_diagnostics/visa_<shot>shot_seed10`；控制：`results/dual_output/visa_seed10/visa_<shot>shot_seed10`。可用 `--visa_source_root`、`--visa_control_root` 指定其他已完成会话。
- 新五集源/控制：`results/branch_diagnostics/multi_dataset_seed10/<dataset>/<shot>shot_seed10` 和对应dual_output目录。可用 `--history_session` 指定同名历史会话。
- 严格读取已记录正常TRAIN参考、重复次数、顺序和原编码批次。审计源/控制CSV与metadata摘要，不记录Git提交哈希。不得拿相同seed代替相同参考名单。
- 输出：`results/context_alignment/six_datasets_seed10/<dataset>/<shot>shot_seed10/`；重复运行使用 `_run02` 等。会话名固定，实际运行范围看campaign_config/execution_plan，不自动续跑或覆盖。
- 父目录：metrics_summary.csv、metrics_by_category.csv、factorial_effects.csv、interaction_effects.csv、retention_summary.csv、执行计划和状态。
- 子目录：完整六指标、image_scores.csv、baseline_reference_comparison.csv、baseline_score_checks.csv、baseline_map_checks.csv、routing_checks.csv、参考/测试清单、固定样本可视化、配置和日志。
- B−A为768下邻域效果，C−A为单点接口效果，D−C为1024下邻域效果，D−B为邻域条件下接口效果；交互项为(D−C)−(B−A)。D−A是组合改动相对原模型的总变化。
- 保留标准只用B/C/D分别相对A的六指标净变化，shot分别判断。只有完整六数据集时才输出Pass判断，子集显示N/A；交互项不是候选整体提升。
- 默认每类别正常/异常各一张PNG，不保存新NPY。加 `--save_maps` 保存这些固定样本的四条件原始/平滑图及ALIGN图，不缓存全测试集特征到磁盘。
- 旧A指标容差0.001百分点、逐图分数容差atol=1e-6/rtol=1e-5、固定原FINAL矩阵容差atol=1e-6/rtol=1e-5。缺失必要旧矩阵、基线不符或NaN/Inf均停止，不继续后续组。

## 资源与验证

临时hook仅在实验进程中使用，返回None，不更改原输出。正常支持只编码一次，临时包装原build_prompt_memory观察其四层输出，不更换采样或原forward。查询一次前向同时获得两接口，每批结束清除激活；支持原1024缓存于CPU，候选支持描述子按当前类别缓存。`--match_chunk 256` 只对查询token分块，保留全部参考位置，不做近似检索。

四个条件需要比原来更多的当前类别评价图缓存；原评价器仍可能占用较多内存/显存。每类别结束回收，子进程结束释放资源，不保证Windows页文件立即缩小。

主要逻辑和维度由静态源码审查；不能据此宣称已经跑通、A数值一致或指标提升。`tests/test_context_alignment.py` 包含池化隔离、分块精确匹配、hook输出隔离、类别路由等人工测试，本轮未执行。实际四条件推理、资源占用、关闭恢复、结果保存及基线一致性等待用户运行确认。
