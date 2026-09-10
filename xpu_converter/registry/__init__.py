# -*- coding: utf-8 -*-
"""注册表层: 模型适配器与硬件后端的统一解析入口。"""
from xpu_converter.registry.backend_registry import (
    available_backends,
    create_backend,
    get_backend_class,
    kunlun_config_from,
    register_backend,
    resolve_hardware_config,
)
from xpu_converter.registry.model_registry import (
    available_model_types,
    create_adapter,
    detect_model_type,
    get_adapter_class,
    register_adapter,
    resolve_model_config,
)

__all__ = [
    "available_backends",
    "available_model_types",
    "create_adapter",
    "create_backend",
    "detect_model_type",
    "get_adapter_class",
    "get_backend_class",
    "kunlun_config_from",
    "register_adapter",
    "register_backend",
    "resolve_hardware_config",
    "resolve_model_config",
]
