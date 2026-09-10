# -*- coding: utf-8 -*-
"""模型适配器注册表。

用于把 ``--model-type yolov10`` 这类字符串解析为具体的
:class:`xpu_converter.frontend.base.BaseModelAdapter` 实现, 并负责装配该模型的
配置文件 (``configs/models/<model_type>.yaml``)。
"""
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from xpu_converter.config import ModelConfig, deep_merge, load_yaml
from xpu_converter.errors import ConfigError, NotSupportedError
from xpu_converter.frontend.base import BaseModelAdapter
from xpu_converter.paths import model_config_path

# 文件名中的模型关键字 -> model_type(用于 --model-type 缺省时的自动识别)
FILENAME_HINTS: List[tuple] = [
    ("yolov5", "yolov5"),
    ("yolov8", "yolov8"),
    ("yolov9", "yolov9"),
    ("yolov10", "yolov10"),
    ("yolo11", "yolov11"),
    ("yolo12", "yolov12"),
    ("yolo26", "yolov26"),
    ("ppocr", "ppocr"),
    ("paddledetection", "paddledetection"),
    ("ppdet", "paddledetection"),
]

_ADAPTERS: Dict[str, Type[BaseModelAdapter]] = {}
_LOADED = False


def register_adapter(adapter_cls: Type[BaseModelAdapter]) -> Type[BaseModelAdapter]:
    """注册适配器(可用于扩展自定义模型)。"""
    _ADAPTERS[adapter_cls.model_type] = adapter_cls
    return adapter_cls


def _ensure_loaded() -> None:
    """惰性导入内置适配器, 避免在未安装 torch/paddle 的环境下导入失败。"""
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    from xpu_converter.frontend.pytorch import (
        yolo12, yolo26, yolov5, yolov8, yolov9, yolov10, yolov11,
    )
    from xpu_converter.frontend.paddle import ppocr, paddledetection

    for module in (yolov5, yolov8, yolov9, yolov10, yolov11, yolo12, yolo26,
                   ppocr, paddledetection):
        for name in dir(module):
            candidate = getattr(module, name)
            if isinstance(candidate, type) and issubclass(candidate, BaseModelAdapter) \
                    and candidate is not BaseModelAdapter and candidate.model_type != "base":
                register_adapter(candidate)


def available_model_types() -> List[str]:
    _ensure_loaded()
    return sorted(_ADAPTERS)


def get_adapter_class(model_type: str) -> Type[BaseModelAdapter]:
    _ensure_loaded()
    key = str(model_type or "").strip().lower()
    if key not in _ADAPTERS:
        raise NotSupportedError(
            "不支持的模型类型: {!r}, 当前支持: {}".format(model_type, ", ".join(available_model_types()))
        )
    return _ADAPTERS[key]


def resolve_model_config(model_type: str, overrides: Optional[Dict[str, Any]] = None) -> ModelConfig:
    """读取模型配置文件并与调用方覆盖项合并。

    配置文件缺失时退化为适配器默认值, 保证新增模型类型也能跑通。
    """
    adapter_cls = get_adapter_class(model_type)
    data: Dict[str, Any] = {
        "model_type": adapter_cls.model_type,
        "framework": adapter_cls.framework,
        "task": adapter_cls.task,
    }
    path = model_config_path(model_type)
    if Path(path).is_file():
        data = deep_merge(data, load_yaml(path))
    data = deep_merge(data, overrides or {})
    config = ModelConfig.from_dict(data)
    if not config.class_names:
        config.class_names = list(adapter_cls().default_class_names())
    if not config.num_classes:
        config.num_classes = len(config.class_names) or adapter_cls().default_num_classes()
    return config


def create_adapter(model_type: str, overrides: Optional[Dict[str, Any]] = None) -> BaseModelAdapter:
    """按模型类型创建适配器实例。"""
    adapter_cls = get_adapter_class(model_type)
    return adapter_cls(resolve_model_config(model_type, overrides))


def detect_model_type(model_path: str, explicit: Optional[str] = None) -> str:
    """推断模型类型: 显式指定优先, 其次按文件名关键字匹配。"""
    if explicit:
        get_adapter_class(explicit)
        return str(explicit).lower()
    name = os.path.basename(str(model_path or "")).lower()
    for keyword, model_type in FILENAME_HINTS:
        if keyword in name:
            return model_type
    raise ConfigError(
        "无法从文件名 {} 推断模型类型, 请显式指定 --model-type (可选: {})".format(
            os.path.basename(str(model_path or "")), ", ".join(available_model_types())
        )
    )
