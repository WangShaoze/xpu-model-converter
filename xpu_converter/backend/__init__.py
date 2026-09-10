# -*- coding: utf-8 -*-
"""硬件后端层。

- :mod:`xpu_converter.backend.base`        后端抽象契约
- :mod:`xpu_converter.backend.kunlun`      昆仑芯 XPU 后端(SDK 解耦)
"""
from xpu_converter.backend.base import (
    BackendArtifact,
    BaseBackend,
    BaseRuntimeSession,
    OperatorAnalysis,
    OperatorAnalysisResult,
)

__all__ = [
    "BackendArtifact",
    "BaseBackend",
    "BaseRuntimeSession",
    "OperatorAnalysis",
    "OperatorAnalysisResult",
]
