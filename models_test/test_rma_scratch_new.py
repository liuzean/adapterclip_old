"""方案二测试：新训练流程从头训练的四适配器，对比已有原A。

默认六数据集、1/2/4-shot、seed10；使用源域验证选出的best.pth。
暂未安排同样新流程的三适配器，先看结果再决定；结果包含RMA和训练流程影响。
"""
from pathlib import Path
from rma_scratch_test_common import testing_main

if __name__ == '__main__':
    testing_main('new', Path(__file__).stem, Path(__file__).resolve())
