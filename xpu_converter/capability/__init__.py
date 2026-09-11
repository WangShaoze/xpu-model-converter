# -*- coding: utf-8 -*-
"""Capability 机制: 算子能力 + 硬件能力 + 编译前门禁。

ChatGPT 修改意见 §33 要求建立 Capability Matrix, 本包即其实现:
``operators`` 描述"支持哪些算子/属性/dtype", ``hardware`` 描述"这台机器的
XPU 能力与版本指纹", ``matrix`` 把两者与最终图合成一次编译前判定。
"""
from xpu_converter.capability.hardware import HardwareCapability, probe_kunlun
from xpu_converter.capability.matrix import CapabilityReport, OperatorDecision, build_report
from xpu_converter.capability.operators import (
    OperatorCapability,
    OperatorCapabilitySet,
)

__all__ = [
    "HardwareCapability",
    "probe_kunlun",
    "CapabilityReport",
    "OperatorDecision",
    "build_report",
    "OperatorCapability",
    "OperatorCapabilitySet",
]
