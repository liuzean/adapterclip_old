r"""实验K：先训练CRA，再自动用本次源域best测试六集1/2/4-shot。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from cra_workflow import pipeline_main

if __name__ == '__main__': pipeline_main(Path(__file__).stem)
