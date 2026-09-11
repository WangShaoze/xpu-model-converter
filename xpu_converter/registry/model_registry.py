# -*- coding: utf-8 -*-
"""模型适配器注册表。

用于把 ``--model-type yolov10`` 这类字符串解析为具体的
:class:`xpu_converter.frontend.base.BaseModelAdapter` 实现, 并负责装配该模型的
配置文件 (``configs/models/<model_type>.yaml``)。
"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from xpu_converter.config import ModelConfig, deep_merge, load_yaml
from xpu_converter.errors import ConfigError, NotSupportedError
from xpu_converter.frontend.base import BaseModelAdapter
from xpu_converter.paths import model_config_path

# ---------------------------------------------------------------- 生命周期
#: 已随 V1 交付验证、可对外宣称支持
STATUS_STABLE = "stable"
#: 代码可跑但未经真机/真数据验证, 不承诺
STATUS_EXPERIMENTAL = "experimental"
#: 保留兼容, 不再演进
STATUS_DEPRECATED = "deprecated"
#: 规划中, 当前不支持(调用会明确报错, 不做静默降级)
STATUS_PLANNED = "planned"

STATUS_ORDER = (STATUS_STABLE, STATUS_EXPERIMENTAL, STATUS_DEPRECATED, STATUS_PLANNED)

#: V1 锁定范围(ChatGPT 修改意见 §51): 只有 YOLOv10 是 stable, 其余一律不算承诺
V1_STABLE_MODELS = ("yolov10",)


@dataclass(frozen=True)
class ModelSupport:
    """模型的生命周期状态(ChatGPT 修改意见 §50)。

    "代码里有 adapter" 不等于 "声称支持": 只有 ``status == stable`` 才是可对外
    交付的模型, 其余需显式标注 experimental / planned, 避免维护失控(§49/§51)。
    """

    model_type: str
    status: str = STATUS_EXPERIMENTAL
    framework: str = "pytorch"
    task: str = "detection"
    min_framework_version: str = ""
    notes: str = ""

    @property
    def is_deliverable(self) -> bool:
        return self.status == STATUS_STABLE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_type,
            "status": self.status,
            "framework": self.framework,
            "task": self.task,
            "min_framework_version": self.min_framework_version,
            "notes": self.notes,
        }


#: 各模型的生命周期声明。新增模型必须在此登记状态, 否则默认 experimental。
SUPPORT_TABLE: List[ModelSupport] = [
    ModelSupport("yolov10", STATUS_STABLE, "pytorch", "detection", "ultralytics==8.1.*",
                 "V1 唯一承诺交付的模型"),
    ModelSupport("yolov8", STATUS_EXPERIMENTAL, "pytorch", "detection", "ultralytics>=8.0"),
    ModelSupport("yolov9", STATUS_EXPERIMENTAL, "pytorch", "detection", ""),
    ModelSupport("yolov7", STATUS_EXPERIMENTAL, "pytorch", "detection", ""),
    ModelSupport("yolov6", STATUS_EXPERIMENTAL, "pytorch", "detection", "",
                 "美团视觉智能部, 原生 pt→onnx→paddle"),
    ModelSupport("yolov11", STATUS_EXPERIMENTAL, "pytorch", "detection", "ultralytics>=8.3"),
    ModelSupport("yolov5", STATUS_EXPERIMENTAL, "pytorch", "detection", ""),
    ModelSupport("yolov12", STATUS_EXPERIMENTAL, "pytorch", "detection", ""),
    ModelSupport("yolov26", STATUS_EXPERIMENTAL, "pytorch", "detection", ""),
    ModelSupport("ppocr", STATUS_PLANNED, "paddle", "ocr", "",
                 "Paddle/OCR 属于 P2, 当前不支持"),
    ModelSupport("paddledetection", STATUS_PLANNED, "paddle", "detection", "",
                 "Paddle 前端属于 P2, 当前不支持"),
]


def model_support(model_type: str) -> ModelSupport:
    """查询模型生命周期声明; 未登记的模型按 experimental 处理。"""
    key = str(model_type or "").strip().lower()
    for item in SUPPORT_TABLE:
        if item.model_type == key:
            return item
    return ModelSupport(key or "unknown")


def supported_models(status: Optional[str] = None) -> List[ModelSupport]:
    """按状态筛选模型; ``status`` 为空时返回全部。"""
    if not status:
        return list(SUPPORT_TABLE)
    wanted = str(status).strip().lower()
    return [item for item in SUPPORT_TABLE if item.status == wanted]


# 文件名中的模型关键字 -> model_type(用于 --model-type 缺省时的自动识别)
FILENAME_HINTS: List[tuple] = [
    ("yolov5", "yolov5"),
    ("yolov6", "yolov6"),
    ("yolov7", "yolov7"),
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
        yolo12, yolo26, yolov5, yolov6, yolov7, yolov8, yolov9, yolov10, yolov11,
    )
    from xpu_converter.frontend.paddle import ppocr, paddledetection

    for module in (yolov5, yolov6, yolov7, yolov8, yolov9, yolov10, yolov11, yolo12, yolo26,
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
    support = model_support(key)
    if support.status == STATUS_PLANNED:
        # 规划中的模型即便残留了适配器代码也明确拒绝, 不做"能跑就算支持"的静默承诺(§49/§51)
        raise NotSupportedError(
            "模型 {} 当前状态为 {}: {} (见 xpu-converter list-models)".format(
                key, support.status, support.notes or "尚未实现"
            )
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
