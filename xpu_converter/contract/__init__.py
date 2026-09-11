# -*- coding: utf-8 -*-
"""Contract 机制: 把模型输入/输出语义显式化, 让 Runtime 不再靠 shape 猜。

当前实现输出侧契约 (:mod:`xpu_converter.contract.output`); 输入/预处理契约
沿用 :class:`xpu_converter.config.ModelConfig` 的 ``input_*`` / ``preprocess``。
"""
from xpu_converter.contract.output import (
    FORMAT_CXCYWH,
    FORMAT_XYXY,
    LAYOUT_BCN,
    LAYOUT_BNC,
    LAYOUT_BNC6,
    LAYOUT_PAIR,
    ModelOutputContract,
    OutputContract,
)

__all__ = [
    "OutputContract",
    "ModelOutputContract",
    "LAYOUT_BCN",
    "LAYOUT_BNC",
    "LAYOUT_BNC6",
    "LAYOUT_PAIR",
    "FORMAT_XYXY",
    "FORMAT_CXCYWH",
]
