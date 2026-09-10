# -*- coding: utf-8 -*-
"""``python -m xpu_converter`` 入口(等价于 ``xpu-converter`` 命令)。"""
import sys

from xpu_converter.cli.main import main

if __name__ == "__main__":
    sys.exit(main())
