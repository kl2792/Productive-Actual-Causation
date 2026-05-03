import sys
import os

_src = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, _src)
sys.path.insert(0, os.path.join(_src, "baselines"))
