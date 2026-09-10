# -*- coding: utf-8 -*-
"""xpu-model-converter: 模型转换平台。

流水线::

    .pt / Paddle
        -> FRONTEND (Model Adapter)
        -> IR / ONNX
        -> OPTIMIZER (shape/constant/simplify/fusion)
        -> REWRITE (算子等价改写)
        -> XPU COMPILE (Kunlun Backend)
        -> model.xpu
        -> VALIDATION (精度 / 性能)
        -> EXPORT (Docker 交付包)
"""
from xpu_converter.version import CONVERTER_VERSION, __version__

__all__ = ["__version__", "CONVERTER_VERSION"]
