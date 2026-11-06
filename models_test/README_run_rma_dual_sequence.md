# 两项双输出顺序验证

在项目根目录执行：

```powershell
python .\models_test\run_rma_dual_sequence.py
```

支持在models_test内直接运行。顺序固定：

1. rma_fixed_dual_output.py：完整六数据集1/2/4-shot固定12.5%双输出。
2. rma_beta4_dual_output.py：完整六数据集1/2/4-shot β4双输出。

两者保持seed10、同一第14轮权重和各自历史输入/离线核对；不训练。
每个dataset/shot由原子任务入口使用独立子进程，完成后退出并清理，再开始下一shot。
只有第一项18轮及双输出/离线一致性全部通过，且整个阶段进程退出后，才启动第二项。
总计36轮dataset/shot。父进程不加载模型或导入torch；阶段间也调用cleanup。
退出释放本轮活动张量和CUDA上下文，空闲缓存清理不等于强制缩小操作系统分页文件。

总记录位于：
`E:\projects\FSAD\AdaptCLIP_old\results\run_rma_dual_sequence\two_duals_seed10\`。
重复运行新建编号目录，不覆盖旧结果。两个子实验结果仍保存在其各自results目录，
总记录保存准确目录回执、命令、完成阶段和失败信息，不猜测“最新目录”。

若失败或Ctrl+C中断，停止后续任务，终端和stop_summary.txt显示阶段/dataset/shot及已完成条件。
重启后只读查看最后保存的进度：

```powershell
python .\models_test\run_rma_dual_sequence.py --status .\results\run_rma_dual_sequence\two_duals_seed10
```

--status不会启动或续跑，也不证明进程当前是否仍在运行。普通运行会从第一项开始，
本入口没有自动断点续跑。复用既有调度器的只读进度显示，不修改旧调度器或双输出算法。
开发仅静态检查，未自动运行训练/推理/pytest。
