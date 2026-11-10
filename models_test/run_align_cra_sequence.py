r"""依次执行实验J ALIGN定位对照，再执行实验K CRA训练和完整测试。

python .\models_test\run_align_cra_sequence.py
每阶段/测试shot独立进程；训练每epoch清理；失败报告阶段/epoch/batch或数据集/shot。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from cra_workflow import pipeline_main

if __name__ == '__main__': pipeline_main(Path(__file__).stem)
