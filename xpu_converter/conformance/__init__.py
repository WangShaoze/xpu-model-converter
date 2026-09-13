# -*- coding: utf-8 -*-
"""模型级 Conformance(逐后端一致性)框架(ChatGPT 修改意见 P0-8)。

把"模型真支持还是假支持"从人工声明变成可自动判定的数值证据链: 对同一份
golden 输入, 逐级后端输出均保存指纹(shape/dtype/sha256)并与基准数值比对。
"""
from xpu_converter.conformance.model_conformance import (
    ConformanceRunner,
    ConformanceStage,
    ModelConformanceReport,
    StageResult,
    tensor_fingerprint,
)

__all__ = [
    "ConformanceRunner",
    "ConformanceStage",
    "StageResult",
    "ModelConformanceReport",
    "tensor_fingerprint",
]