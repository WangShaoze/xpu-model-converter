# -*- coding: utf-8 -*-
"""昆仑芯 XPU 后端。

对外只暴露 :class:`KunlunBackend` 与配置对象; SDK 细节全部封装在
:mod:`xpu_converter.backend.kunlun.compiler` 的适配器里。
"""
from xpu_converter.backend.kunlun.compiler import (
    ARTIFACT_FILENAME,
    METADATA_FILENAME,
    KunlunBackend,
    KunlunSdkAdapter,
    PaddleXpuSdkAdapter,
    StubSdkAdapter,
    XpuToolkitSdkAdapter,
)
from xpu_converter.backend.kunlun.config import SDK_ADAPTERS, KunlunConfig
from xpu_converter.backend.kunlun.graph_builder import XpuGraph, XpuGraphBuilder, write_compile_report
from xpu_converter.backend.kunlun.operator_registry import (
    NATIVE_OPS,
    REWRITE_OPS,
    KunlunOperatorRegistry,
    default_registry,
)

__all__ = [
    "ARTIFACT_FILENAME",
    "METADATA_FILENAME",
    "SDK_ADAPTERS",
    "KunlunBackend",
    "KunlunConfig",
    "KunlunOperatorRegistry",
    "KunlunSdkAdapter",
    "NATIVE_OPS",
    "PaddleXpuSdkAdapter",
    "REWRITE_OPS",
    "StubSdkAdapter",
    "XpuGraph",
    "XpuGraphBuilder",
    "XpuToolkitSdkAdapter",
    "default_registry",
    "write_compile_report",
]
