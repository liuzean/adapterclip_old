# 新D同权重关闭RMA的原test.py测试入口

```powershell
python .\models_test\new_d_without_rma.py
```

在项目根目录执行，默认Visa→BTAD→MPDD→SDD→DTD→part_glass，每个数据集依次1/2/4-shot、seed10。
也支持在models_test目录直接运行。--datasets和--shots可显式限制范围，默认完整18条件。

直接调用原始test.py的test函数，不复制或修改其模型加载、推理循环、ALIGN、高斯平滑和最终图像评分。
固定使用完整新D测试中相同的第14轮best.pth；原test.py只读取其中三个适配器状态，忽略RMA。
不会实例化RMA，也不重新训练。与train_new_d_three.py产生新权重的训练对照不同。

子进程内临时包装原模块的三个接口，结束时恢复：PromptDataset构造器、Dataset构造器、Evaluator。
构造器仍返回原始数据集实例，保持Windows DataLoader多进程可序列化；不替换数据集取图、变换或采样规则。
原参考采样器实际选出的路径与历史参考逐项核对类别、路径、顺序和重复次数，任何不一致立即停止。
查询初始化及最终送入评价器的路径、顺序、类别、标签均核对历史全测试集名单。
评价包装器调用原Evaluator计算六指标，同时保存高精度CSV和逐图分数，不改变预测值。

每个dataset/shot用独立子进程运行，主调度器不加载torch/模型。
结束时执行垃圾回收和CUDA空闲缓存释放，等进程退出再启动下一条件；进程退出释放该轮模型及张量。
这是应用内资源释放，不保证操作系统文件缓存或分页文件体积立即缩小。
原test.py内部仍采用原来的全测试集结果缓冲方式，未改成新的分批评价协议。

输出位于results/new_d_without_rma/six_datasets_seed10，重复运行自动创建_run02等目录。
每条件保存原测试日志、实际参考、reference_identity_checks.csv、query_identity_checks.csv、
image_scores.csv、完整六指标、输入配置/哈希、查询名单和完成状态。
不默认新增矩阵或可视化。

汇总：metrics_summary.csv、metrics_by_category.csv、deltas_vs_original.csv、retention_by_shot.csv。
其中baseline来自历史原A高精度结果，没有重新运行原A，不声称本次通过原A预测回归。
deltas_vs_new_D.csv为“关闭RMA减完整新D”，负数表示该指标完整新D更好。
不把原A图像分数替换到候选结果，候选图像分数是原test.py重新计算的三分支分数。

开发只做静态检查、既有文件审计；未自动启动推理或pytest。原test.py未修改。
