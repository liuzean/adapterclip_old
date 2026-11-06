"""β4双输出：G0_L1完整最终图像分数＋β4融合/ALIGN/平滑最终图。

不训练、不用β4图最大值重新计算dual图像分数；六数据集1/2/4-shot、seed10。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rma_new_d_dual_common import main

if __name__ == '__main__':
    main('beta4', Path(__file__).resolve())
