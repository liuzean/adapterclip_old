# 新D三适配器训练对照

本实验从一开始不创建RMA，冻结CLIP，新建并训练VA、TA、PQA。
这与加载新D第14轮四适配器权重、只在推理时忽略RMA不同。

## 训练和公平对比

默认匹配已完成的results/train_rma_scratch_new/mvtec_seed10：
相同CLIP、源域元数据与1380/345划分，相同正常参考池及逐查询参考采样，seed10。
训练每小批轮换1/2/4参考，实际batch2、累积4，AdamW lr1e-4、betas(.9,.999)、decay1e-4，
15轮、梯度裁剪1。移除RMA及自身损失；三分支融合后仍与ALIGN融合，并保留融合图监督。
不加载原A、旧B或新D适配器作为初始化。三个新适配器的构造顺序及随机起点沿用新D；
新D创建RMA后恢复了随机状态，本对照直接跳过RMA创建。
源域验证自行选best.pth，不固定第14轮，不使用目标测试指标选权重。

默认读取匹配新D的训练参数、输入和training_episodes.csv，逐查询核对。
若任何设置或采样不同即停止，避免误称同协议对照；不会自动修改或覆盖历史新D。
要匹配其他完整新D训练，显式传--matched_new_d_run并保持对应设置。
这是一次新训练，尚未运行；不能据此声称效果提升。

## 从项目根目录运行

```powershell
python .\models_test\train_new_d_three.py
```

结束后使用终端显示的准确best.pth路径。第一次默认输出对应：

```powershell
python .\models_test\test_new_d_three.py --checkpoint .\results\train_new_d_three\mvtec_seed10\best.pth
```

重复训练创建_run02等目录，不能继续使用上一条示例路径代替实际输出。
可从models_test目录直接运行入口，路径均按项目根目录解析。
无需改动原train.py、test.py、adaptcliplib。测试拒绝四适配器检查点冒充本对照。

测试默认Visa→BTAD→MPDD→SDD→DTD→part_glass，每数据集依次1/2/4-shot、seed10。
每条件独立子进程，等待退出后再启动下一条件；训练每epoch释放空闲缓存，保留模型和优化器状态。
失败停止后续条件。测试历史正常参考和查询固定；保留原三分支＋ALIGN＋平滑及最终图像评分。

## 输出

- results/train_new_d_three/mvtec_seed10：每轮权重、best.pth、last.pth、损失曲线、训练/验证指标、配置、采样及匹配审计。
- results/test_new_d_three/six_datasets_seed10：原A及三适配器对照的六项指标、逐类别指标、逐图分数、固定少量可视化、基线一致性检查。
- deltas_vs_original.csv与retention_by_shot.csv：本对照减原A，按各shot分别判断。
- deltas_vs_new_D.csv：本对照减历史完整新D，负值表示该指标新D更好；不是新D减本对照。

--new_d_run默认匹配results/test_rma_scratch_new/six_datasets_seed10，检查其检查点来自同一训练及同一目标输入。
矩阵默认不新增，显式--save_maps才保存选定样本，不能代替全测试集指标。
只执行静态开发检查，未运行模型、训练、推理或pytest。

## 原test.py能否直接忽略新D的RMA

可以。原test.py只读取textual_learner、visual_learner、pq_learner三个字段，额外rma字段不参与推理。
必须显式启用--visual_learner --textual_learner --pq_learner --pq_context，并匹配原测试配置、参考图及指标。
这类推理消融使用新D自己的三个适配器，不等于加载原A，也不等于本次重新训练的三适配器。
另指定新save_path，避免原test.py向旧结果目录追加日志。原test.py按单数据集单shot运行，
不会自动核对历史参考身份，也不会自动顺序调度六数据集18条件。
