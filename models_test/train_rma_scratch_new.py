"""方案二：CLIP预训练＋四个全新适配器；采用新的源域训练/验证流程。

80/20源域划分、训练参考轮换1/2/4、AdamW/batch2/累积4、融合监督、验证选优。
四适配器从第一轮共同学习，不加载原epoch15或B/C权重；统一初始lr=1e-4。
暂未安排同样新流程的三适配器对照，先看结果；不能单独归因训练流程影响。
"""
from pathlib import Path
from rma_scratch_common import training_main

if __name__ == '__main__':
    training_main('new', Path(__file__).stem)
