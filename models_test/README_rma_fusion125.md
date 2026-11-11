# 实验L：12.5%训练融合监督与12.5%定位部署

2026年11月11日完成记录：15轮训练及六集18条件测试已完成，源域best选第12轮，89.592239073；原新D同口径best14为89.698437209。同第14轮＋12.5%定位的主条件相对H三个shot均未通过；best12＋12.5%仅4-shot通过，不能保留为本次主改进。辅助best12＋25%定位在2/4-shot通过，可保留为实验性结果，但相对K双输出仅4-shot通过，暂不替换主候选。20700条episode与原新D一致，冻结/梯度和同权重图像评分不变运行检查通过。完整分析在results/rma_fusion125_pipeline/L_seed10/analysis_20261111/report.md，已写入E盘Word；pytest与故意中断恢复故障测试未执行。下文静态开发说明描述初始交付范围，当前结果以本段及完成记录为准。

本实验回答：相比新D训练25%、H部署定位12.5%，从训练开始把融合监督中的RMA比例改为12.5%是否有效。CLIP冻结，VA/TA/PQA/RMA从原新D相同初始化共同训练15轮。不加载H旧适配器做额外训练，不创建CRA或叠加其他模块。

本次只新增独立文件。原始train.py、test.py、rma_scratch_common.py、rma_joint_utils.py、H及CRA代码不修改，也没有进程内monkey-patch。正在运行cra_dual_output.py的已记录代码指纹及历史输入保持不变。应等其GPU任务结束后再启动本实验；这次开发不自动运行任何模型。

## 三个关键点

1. 从与新D相同的预训练CLIP和新建四适配器起点重训，不把H继续训练当作同预算对照。复用初始化函数、同seed10、调用顺序、源域80/20划分、1380训练/345验证查询、正常参考池及逐批1/2/4参考轮换。每条查询与参考episode必须和原新D20700条记录相同，不同即在该批前向前停止。
2. 只改融合监督图：`0.875*mean(VA,TA,PQA)+0.125*RMA`，之后用原ALIGN和可微调和融合计算Focal/Dice融合损失。各分支独立分类/局部损失、融合损失总系数1均保持不变；12.5%不是RMA独立损失权重。训练仍不加Gaussian平滑，与旧训练一致。
3. 保存全部15轮、epoch14和源域best。主分析用新epoch14与H的旧epoch14；补充报告新源域best与旧源域best14，明确新best不一定为14。新best严格由源域验证最高分选出，同分保留早期轮次，目标测试标签不参与选轮。

源域选优**刻意沿用原新D完整四分支25%输出口径**：六指标类别宏平均，再对1/2/4-shot平均。没有同时把选优改为H/12.5%，以保证训练监督比例是唯一训练实验变量。这里“训练12.5%”指梯度监督图，“部署12.5%”指目标定位图，并非整个训练/验证/图像评分所有位置均为12.5%。若以后要改源域选优口径，应另作两组统一口径的比较。

## 不变的训练设置

- MVTec源域，15轮，seed10；batch2、梯度累积4（尾组按实际批数归一化）。
- AdamW，lr1e-4，betas0.9/0.999，weight_decay1e-4，梯度裁剪1，workers0。
- 原四层[6,12,18,24]，图像518，CLIP投影后768维，PQA context开启，原VA/TA/PQA开启。
- RMA维度128、Top8、learned匹配，与原新D相同。训练入口不提供这些参数的随意覆盖，防止偏离最小对照。

## 根目录运行

训练和测试分开：

```powershell
python .\models_test\train_rma_fusion125.py
python .\models_test\test_rma_fusion125.py --training_run .\results\train_rma_fusion125\mvtec_seed10
```

一键先训练再测试：

```powershell
python .\models_test\rma_fusion125_pipeline.py
```

也支持在models_test目录直接运行相应脚本。重复运行自动编号，不覆盖历史目录；单独测试时用训练终端实际打印的目录，自动管线会精确传递本次训练目录，避免误用旧运行。

## 测试内容与指标

Visa→BTAD→MPDD→SDD→DTD→part_glass；每集1→2→4-shot，seed10。自动核对H及旧25%消融的正常TRAIN参考、查询身份/顺序、类别和评价口径。每dataset/shot独立子进程，等待退出、清理后继续；类别缓冲评价后也清理。保留完整六指标。

每个条件分别评估新第14轮、源域best；同权重时复用已计算结果并明确记录。每份新权重只编码一次当前批查询，输出两种定位图：

- `epoch14_L125`、`best_L125`：12.5%定位＋ALIGN＋sigma4；分别为主、补充候选。
- `epoch14_L250`、`best_L250`：辅助25%定位＋相同ALIGN/sigma4，不新增训练。

两定位条件共用该新权重自己的完整H图像分数：`0.75*SE25+0.25*G0`。SE25仍是G0_L1的25%完整评分，不用12.5%图最大值重算图像分数，也不直接复制旧H分数。因此，同权重两个定位条件图像指标必须完全相同；新旧权重之间图像指标允许变化。

对照同时报告历史H和原A；`old_H_L250`是历史H图像指标与同权重G0_L1像素指标的组合，用于旧训练25%×定位25%的辅助表格，它是历史指标组合，不声称本实验重新推理了旧模型完整25%图。历史没有全量像素数组，不声称历史逐像素一致。

至少3/6数据集六指标净和严格为正且六集总净和正，该shot通过；至少两个shot通过才考虑保留。每个候选/比较对象/shot分别判定，不混合epoch14和best选择有利数据集。

## 结果目录

- 训练：`results/train_rma_fusion125/mvtec_seed10/`，包括逐轮权重、best、last、损失、源域验证指标、查询/参考episode、进度及输出指纹。
- 测试：`results/test_rma_fusion125/L_seed10/`，包括metrics_summary.csv、各类别指标、deltas_vs_H.csv、deltas_vs_baseline.csv、retention_by_shot.csv及各轮明细/逐图评分。
- 总记录：`results/rma_fusion125_pipeline/L_seed10/`，包括训练/测试精确目录、两个选定权重、阶段状态及停止摘要。

默认不写图片或npy矩阵，不把结果写到C盘；系统自身分页文件与第三方库系统缓存不由该结果路径控制。所有实验结果路径基于项目根目录。

## 状态和中断

```powershell
python .\models_test\rma_fusion125_pipeline.py --status .\results\rma_fusion125_pipeline\L_seed10
python .\models_test\rma_fusion125_pipeline.py --resume_from .\results\rma_fusion125_pipeline\L_seed10
python .\models_test\test_rma_fusion125.py --resume_from .\results\test_rma_fusion125\L_seed10
```

status只读，不启动模型、不证明进程存活。训练停止显示epoch/batch；测试停止显示数据集/shot，后续停止。只有训练完整完成后，管线resume才跳过训练并继续测试。未完成测试条件在新尝试目录重新运行；没有实现epoch断点续训，不假装恢复优化器/RNG。训练中断后重新启动新任务从头训练，旧输出保留，不自动删除。

## 文件与开发验证

训练、测试、管线与辅助模块分别为train_rma_fusion125.py、test_rma_fusion125.py、rma_fusion125_pipeline.py、rma_fusion125_utils.py。CPU契约在tests/test_rma_fusion125_contracts.py，覆盖融合比例、独立损失不变、梯度及分支输入不被原地修改；同权重图像分数不变由实际测试入口运行时检查。测试文件不负责退回代码。

开发阶段仅静态语法/接口、已有输入与采样记录检查，不自动训练、推理或执行pytest。新训练权重加载、数值/梯度、18条件测试及中断恢复需要实际运行验证。以终端和结果完成记录为准，不预先声称新方法有收益。
