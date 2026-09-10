# -*- coding: utf-8 -*-
"""Frontend 公共接口。

每个 :class:`BaseModelAdapter` 子类对应一种「框架 + 模型结构」(如 PyTorch + YOLOv10),
负责:

1. ``inspect(model_path)``      : 识别模型并给出元信息
2. ``export_onnx(...)``         : 导出静态 shape 的 ONNX
3. 声明默认输入 shape / 类别数 / 类别名 / 是否为端到端 (含 NMS) 模型
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from xpu_converter.classes import coco_classes
from xpu_converter.config import ModelConfig
from xpu_converter.errors import ExportError, ModelLoadError, NotSupportedError


@dataclass
class FrontendModel:
    """模型识别结果。"""

    framework: str
    model_type: str
    task: str
    source_path: str = ""
    input_shape: List[int] = field(default_factory=lambda: [1, 3, 640, 640])
    input_dtype: str = "float32"
    num_classes: int = 80
    class_names: List[str] = field(default_factory=list)
    end2end: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "framework": self.framework,
            "model_type": self.model_type,
            "task": self.task,
            "source_path": self.source_path,
            "input_shape": list(self.input_shape),
            "input_dtype": self.input_dtype,
            "num_classes": self.num_classes,
            "class_names": list(self.class_names),
            "end2end": self.end2end,
            "extra": dict(self.extra),
        }


class BaseModelAdapter(ABC):
    """模型适配器基类。"""

    framework: str = "base"
    model_type: str = "base"
    task: str = "detection"
    # 端到端(带 NMS)模型标记, 决定后处理是否需要在 CPU 上做 NMS
    end2end_default: bool = False

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig(model_type=self.model_type, framework=self.framework)

    # ------------------------------------------------------------- 默认值
    def default_input_shape(self) -> List[int]:
        return [1, 3, 640, 640]

    def default_num_classes(self) -> int:
        return 80

    def default_class_names(self) -> List[str]:
        return coco_classes()

    def opset(self) -> int:
        return int(self.config.opset or 13)

    def input_shape(self) -> List[int]:
        return list(self.config.input_shape or self.default_input_shape())

    # ------------------------------------------------------------- 生命周期
    @abstractmethod
    def load_model(self, model_path: str) -> Any:
        """加载原生模型对象; 失败抛 :class:`ModelLoadError`。"""

    @abstractmethod
    def export_onnx(
        self,
        model_path: str,
        output_path: str,
        input_shape: Optional[List[int]] = None,
        opset: Optional[int] = None,
        dynamic: Optional[bool] = None,
    ) -> str:
        """导出 ONNX, 返回 ONNX 文件路径。"""

    def inspect(self, model_path: str) -> FrontendModel:
        """识别模型并返回元信息 (默认实现基于适配器默认值)。"""
        return FrontendModel(
            framework=self.framework,
            model_type=self.model_type,
            task=self.task,
            source_path=str(model_path or ""),
            input_shape=self.input_shape(),
            input_dtype=self.config.input_dtype,
            num_classes=self.config.num_classes or self.default_num_classes(),
            class_names=list(self.config.class_names or self.default_class_names()),
            end2end=bool(self.config.end2end) or self.end2end_default,
        )

    # ------------------------------------------------------------- 工具
    @staticmethod
    def _require_file(model_path: str) -> str:
        import os

        path = str(model_path or "")
        if not os.path.isfile(path):
            raise ModelLoadError("模型文件不存在: {}".format(path or "<empty>"))
        return path
