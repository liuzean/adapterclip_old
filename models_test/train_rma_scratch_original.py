"""方案一：CLIP预训练＋四个全新适配器；沿用原AdaptCLIP训练流程。

全部源域查询、随机1参考、Adam/batch8/15轮、分支监督，无额外融合损失。
CLIP冻结，VA/TA/PQA/RMA从第一轮共同训练；不加载epoch_15初始化。
主对照为已有原始三适配器；选每轮global_loss均值＋local_loss均值最低的best.pth。
不使用验证集或目标测试选轮，不额外重训三适配器；损失相同时保留更早轮次。
"""
from pathlib import Path
from rma_scratch_common import training_main

if __name__ == '__main__':
    training_main('original', Path(__file__).stem)
