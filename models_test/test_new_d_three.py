"""新D三适配器重训对照测试；拒绝用新D四适配器权重冒充此对照。

默认六数据集，各1/2/4-shot，seed10；每条件独立子进程，退出清理后继续。
使用本次训练源域验证选出的best.pth；保留ALIGN及原三分支最终评分。
输出原A和三适配器对照完整六指标，另与历史新D比较，不重新训练或运行新D。
"""
from pathlib import Path
from rma_scratch_test_common import testing_main

if __name__ == '__main__':
    testing_main('new', Path(__file__).stem, Path(__file__).resolve(), three_adapter_control=True)
