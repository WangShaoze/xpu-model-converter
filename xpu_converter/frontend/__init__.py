# -*- coding: utf-8 -*-
"""Frontend 层: 把不同框架的原生模型统一为 IR/ONNX。"""
from xpu_converter.frontend.base import BaseModelAdapter, FrontendModel

__all__ = ["BaseModelAdapter", "FrontendModel"]
