# 新D两项对照的顺序调度

在项目根目录运行：

```powershell
python .\models_test\run_new_d_controls.py
```

顺序固定：

1. new_d_without_rma.py：新D同一第14轮权重，只在推理时关闭RMA；六数据集各1/2/4-shot。
2. train_new_d_three.py：新流程三适配器从头训练，默认15轮，源域验证选择自己的best.pth。
3. test_new_d_three.py：使用第2步实际生成的best.pth；六数据集各1/2/4-shot。

两个测试均为seed10，每个数据集依次1、2、4张参考。每阶段和每个测试条件都在独立子进程中运行，
等待退出后再继续；训练每epoch清理空闲缓存，继续训练所需模型和优化器保留。
父调度器不导入torch、不载入模型。第2步重复训练出现_run02等编号目录时，第3步自动跟随该次训练回执，
不使用硬编码的mvtec_seed10/best.pth，不猜测最新目录，不会误用旧权重。

总记录：results/run_new_d_controls/new_D_controls_seed10；重复运行创建_run02等新目录。
子任务的训练、测试结果仍分别写入对应入口同名结果根目录。
总记录包括执行计划、各阶段命令、子任务准确目录回执、选中权重路径/哈希、run_status.json。

失败或Ctrl+C时停止后续阶段，并输出阶段名、已完成阶段、子任务目录；训练显示最近完整记录epoch，
测试显示已完成条件数量和最后记录的数据集/shot。stop_summary.txt和run_status.json保存相同信息。
子任务在初始化前失败，可能尚无结果目录或epoch记录，此时明确报告未知进度，不猜测。

突然断电或强制结束进程无法当场打印；可在重启后只读查看最后保存进度：

```powershell
python .\models_test\run_new_d_controls.py --status .\results\run_new_d_controls\new_D_controls_seed10
```

--status不启动、不恢复训练或测试，只显示已保存记录；不能依据旧running状态判断进程当前是否存活。
默认重运行总入口会创建新会话并重新执行三个阶段，本入口不自动断点续训或跳过已完成实验。

开发只做AST、Ruff静态检查及回执接口核对，未自动启动长时间任务或pytest。
原train.py、test.py、adaptcliplib未修改。
