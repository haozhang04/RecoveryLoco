# Copyright (c) 2026-2027 zh
"""
内容：
    为依赖 Isaac Gym 的工具脚本预加载 Conda 环境中的 libpython 动态库。
"""

import ctypes
import os
import sys
from pathlib import Path


def prepend_conda_lib_path() -> None:
    """将当前 Conda 环境的库目录前置，并以全局模式加载 libpython。"""
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return

    conda_lib = Path(conda_prefix) / "lib"
    ld_library_path = os.environ.get("LD_LIBRARY_PATH")
    os.environ["LD_LIBRARY_PATH"] = f"{conda_lib}:{ld_library_path}" if ld_library_path else str(conda_lib)

    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    candidates = sorted(conda_lib.glob(f"libpython{python_version}*.so*"))
    if candidates:
        ctypes.CDLL(str(candidates[0]), mode=ctypes.RTLD_GLOBAL)
