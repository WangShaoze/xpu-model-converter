# -*- coding: utf-8 -*-
"""统一异常体系。

所有异常都继承 :class:`XpuConverterError`，便于 CLI 顶层统一捕获并输出可读信息。
"""


class XpuConverterError(Exception):
    """转换平台基础异常。"""


class ConfigError(XpuConverterError):
    """配置缺失或格式错误。"""


class ModelLoadError(XpuConverterError):
    """模型加载失败。"""


class NotSupportedError(XpuConverterError):
    """当前版本不支持的能力（例如未落地的 Paddle Adapter）。"""


class ExportError(XpuConverterError):
    """前端导出 ONNX/IR 失败。"""


class IrError(XpuConverterError):
    """IR 加载/保存失败。"""


class BackendNotAvailableError(XpuConverterError):
    """目标后端在本机不可用（缺少 SDK / 依赖）。"""


class CompileError(XpuConverterError):
    """后端编译失败。"""


class ValidationError(XpuConverterError):
    """精度/性能校验失败。"""


class PackageError(XpuConverterError):
    """交付包生成失败。"""
