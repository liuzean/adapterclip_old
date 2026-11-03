"""方案一测试：原训练流程从头训练的四适配器，对比已有原A。

默认六数据集、1/2/4-shot、seed10；使用该训练会话训练损失最低的best.pth。
不重训三适配器；不接受旧B/C或新流程权重。
支持复用中断测试中完整的数据集；从首个不完整数据集重跑全部shot，保留旧文件。
"""
from pathlib import Path
from rma_scratch_test_common import testing_main

if __name__ == '__main__':
    testing_main('original', Path(__file__).stem, Path(__file__).resolve())
