"""新D的三适配器训练对照：从训练开始不创建RMA。

冻结CLIP，全新VA/TA/PQA，沿用新D划分、参考采样、AdamW、融合监督及验证选优。
与test.py加载新D后忽略RMA的推理消融不同；本实验会产生新的三适配器权重。
默认严格核对已完成新D训练的设置、输入及逐查询参考采样；不使用目标测试选轮。
"""
from pathlib import Path
from rma_scratch_common import training_main

if __name__ == '__main__':
    training_main('new', Path(__file__).stem, three_adapter_control=True)
