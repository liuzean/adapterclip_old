# H然后E顺序测试

2026-11-07已完成：H和E各18条件，共36轮；实际目录为results/test_h/H_seed10和E_seed10。30816项历史逐图分数和1476项六指标核对最大差0，H/E定位指标相同。精简入口回归通过，不另计新的算法提升。当前固定H、E回退；结果及证据边界见README_H_mainline.md。

从项目根目录运行：

```powershell
python .\models_test\run_test_h_sequence.py
```

在models_test内也可以执行`python .\run_test_h_sequence.py`。

依次执行：

1. `python .\models_test\test_h.py`，默认H，六数据集各1/2/4-shot。
2. H全部18条件完成，历史分数/六指标、冻结检查及结果文件指纹通过后，才执行`python .\models_test\test_h.py --mode E`，同样18条件。

保持test_h默认seed10、第14轮权重、固定参数和参考图，不重新训练。各数据集/shot由test_h独立子进程运行并清理；整个H阶段进程退出后才运行E。总共36个条件。

启动器只增加进度回执参数，不修改预测方法。test_h新增隐藏_progress_file回执，直接记录本次实际结果目录，不猜测“最新”目录。失败或Ctrl+C停止后续模式并输出当前模式、数据集/shot、最近查询数量、已完成条件和结果目录。启动前检查失败、尚未创建结果目录时会明确提示没有回执。

总记录在`results/run_test_h_sequence/H_then_E_seed10/`，重复运行自动编号；H/E各自结果仍在`results/test_h/H_seed10/`和`results/test_h/E_seed10/`或对应编号目录。总记录的stage_results指向实际本次目录。

只查看进度，不运行或恢复：

```powershell
python .\models_test\run_test_h_sequence.py --status .\results\run_test_h_sequence\H_then_E_seed10
```

总启动器不自动续跑；若中断，可以根据总记录中的实际test_h目录，使用test_h.py自己的`--resume_from`完成当前模式，再单独运行尚未启动的模式。重新正常启动总启动器会建立新记录，重新运行H和E，不自动复用以前任务。

新增契约测试在tests/test_run_test_h_sequence.py。开发按静态验证约定不自动运行pytest或GPU推理；此脚本不是新算法实验，只负责顺序运行两种已有模式。
