"""实验I训练：同Top8候选，仅余弦softmax加权；四适配器从头共同学习。

严格匹配已完成新D的设置、划分、查询和参考episode；冻结CLIP及不用的K。
按同一源域验证规则独立选best，不加载H继续训练，不强制选第14轮。
"""
from pathlib import Path
from rma_scratch_common import training_main

if __name__ == '__main__':
    training_main('new', Path(__file__).stem, matching_control=True)
