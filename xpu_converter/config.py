# -*- coding: utf-8 -*-
"""配置对象与 YAML 加载。

- :class:`ModelConfig`    : 模型侧配置 (``configs/models/<model_type>.yaml``)
- :class:`HardwareConfig` : 硬件侧配置 (``configs/hardware/<hardware>.yaml``)
- :class:`BuildManifest`  : 一次转换任务描述 (建设目标 §14 的 model.yaml)
"""
import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:  # PyYAML 为必装依赖, 此处仍做防御
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from xpu_converter.classes import coco_classes
from xpu_converter.errors import ConfigError
from xpu_converter.version import RUNTIME_API_VERSION


def load_yaml(path) -> Dict[str, Any]:
    """读取 YAML 文件为 dict。"""
    path = Path(path)
    if not path.is_file():
        raise ConfigError("配置文件不存在: {}".format(path))
    if yaml is None:
        raise ConfigError("缺少 PyYAML 依赖, 无法解析: {}".format(path))
    with open(path, "r", encoding="utf-8") as fr:
        data = yaml.safe_load(fr) or {}
    if not isinstance(data, dict):
        raise ConfigError("配置文件顶层必须是字典: {}".format(path))
    return data


def dump_yaml(data: Dict[str, Any], path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n": 交付包内的 yaml 需要 LF 换行, 避免 Windows 下写成 CRLF
    with open(path, "w", encoding="utf-8", newline="\n") as fw:
        if yaml is not None:
            yaml.safe_dump(data, fw, allow_unicode=True, sort_keys=False)
        else:  # pragma: no cover
            json.dump(data, fw, ensure_ascii=False, indent=2)


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并 dict, override 优先。"""
    result = copy.deepcopy(base or {})
    for key, value in (override or {}).items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def normalize_shape(shape: Any) -> List[int]:
    """把 ``"1,3,640,640"`` / ``[1,3,640,640]`` 统一为 int 列表。"""
    if shape is None:
        raise ConfigError("input_shape 不能为空")
    if isinstance(shape, str):
        parts = [p.strip() for p in shape.replace("x", ",").split(",") if p.strip()]
        values = []
        for part in parts:
            try:
                values.append(int(part))
            except ValueError:
                raise ConfigError("input_shape 含非法维度: {}".format(shape))
        return values
    if isinstance(shape, (list, tuple)):
        return [int(v) for v in shape]
    raise ConfigError("无法解析 input_shape: {!r}".format(shape))


@dataclass
class ModelConfig:
    """模型侧配置。"""

    model_type: str = "yolov10"
    framework: str = "pytorch"
    task: str = "detection"
    input_shape: List[int] = field(default_factory=lambda: [1, 3, 640, 640])
    input_dtype: str = "float32"
    input_layout: str = "NCHW"
    num_classes: int = 80
    class_names: List[str] = field(default_factory=coco_classes)
    opset: int = 13
    end2end: bool = False
    output_layout: str = "auto"
    dynamic: bool = False
    # 预处理: RGB 顺序 / 归一化 / letterbox
    preprocess: Dict[str, Any] = field(
        default_factory=lambda: {"color": "RGB", "scale": 1.0 / 255.0, "letterbox": True, "pad_value": 114}
    )
    # 后处理: CPU NMS 参数
    postprocess: Dict[str, Any] = field(
        default_factory=lambda: {"nms": "cpu", "max_det": 300, "conf_thres": 0.25, "iou_thres": 0.45}
    )
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ModelConfig":
        data = dict(data or {})
        known = {f for f in cls.__dataclass_fields__ if f != "raw"}
        kwargs = {k: v for k, v in data.items() if k in known}
        if "input_shape" in kwargs:
            kwargs["input_shape"] = normalize_shape(kwargs["input_shape"])
        cfg = cls(**kwargs)
        cfg.raw = data
        return cfg

    @classmethod
    def from_yaml(cls, path) -> "ModelConfig":
        return cls.from_dict(load_yaml(path))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_type,
            "framework": self.framework,
            "task": self.task,
            "input_shape": list(self.input_shape),
            "input_dtype": self.input_dtype,
            "input_layout": self.input_layout,
            "num_classes": self.num_classes,
            "class_names": list(self.class_names),
            "opset": self.opset,
            "end2end": self.end2end,
            "output_layout": self.output_layout,
            "dynamic": self.dynamic,
            "preprocess": copy.deepcopy(self.preprocess),
            "postprocess": copy.deepcopy(self.postprocess),
        }

    def merged(self, override: Optional[Dict[str, Any]]) -> "ModelConfig":
        return ModelConfig.from_dict(deep_merge(self.to_dict(), override or {}))


@dataclass
class HardwareConfig:
    """硬件/后端侧配置。"""

    name: str = "kunlun"
    base_image: str = ""
    sdk_adapter: str = "auto"          # auto | paddle | xpuctl
    sdk_module: str = ""               # sdk_adapter=xpuctl 时使用的模块名
    target_chip: str = "auto"          # 昆仑芯型号 (R200/R300/...)
    precision: str = "fp16"            # fp32 | fp16
    device: str = "auto"               # auto | xpu | cpu
    optimization_level: int = 2
    # 交付容器默认参数
    runtime: Dict[str, Any] = field(
        default_factory=lambda: {
            "api_version": RUNTIME_API_VERSION,
            "type": "detection",
            "port": 58025,
            "workers": 1,
            "conf_thres": 0.25,
            "iou_thres": 0.45,
        }
    )
    # 交付层参数
    docker: Dict[str, Any] = field(
        default_factory=lambda: {
            "template": "kunlun",
            "include_packages": True,
            "include_paddle": True,
        }
    )
    extra: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "HardwareConfig":
        data = dict(data or {})
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in data.items() if k in known and k != "raw"}
        cfg = cls(**kwargs)
        cfg.raw = data
        return cfg

    @classmethod
    def from_yaml(cls, path) -> "HardwareConfig":
        return cls.from_dict(load_yaml(path))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "base_image": self.base_image,
            "sdk_adapter": self.sdk_adapter,
            "sdk_module": self.sdk_module,
            "target_chip": self.target_chip,
            "precision": self.precision,
            "device": self.device,
            "optimization_level": self.optimization_level,
            "runtime": copy.deepcopy(self.runtime),
            "docker": copy.deepcopy(self.docker),
            "extra": copy.deepcopy(self.extra),
        }


@dataclass
class BuildManifest:
    """一次转换任务的完整描述 (建设目标 §14)。"""

    name: str = "model"
    version: str = "v1.0"
    source: Dict[str, Any] = field(default_factory=dict)      # framework/model_type/file
    input: Dict[str, Any] = field(default_factory=dict)       # shape/dtype
    target: Dict[str, Any] = field(default_factory=dict)      # hardware/precision/batch_size
    runtime: Dict[str, Any] = field(default_factory=dict)     # type/port
    validation: Dict[str, Any] = field(default_factory=dict)  # enabled/dataset
    package: Dict[str, Any] = field(default_factory=dict)     # docker/name/version
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BuildManifest":
        data = dict(data or {})
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in data.items() if k in known and k != "raw"}
        manifest = cls(**kwargs)
        manifest.raw = data
        return manifest

    @classmethod
    def from_yaml(cls, path) -> "BuildManifest":
        return cls.from_dict(load_yaml(path))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "task": self.task,
            "source": copy.deepcopy(self.source),
            "input": copy.deepcopy(self.input),
            "target": copy.deepcopy(self.target),
            "runtime": copy.deepcopy(self.runtime),
            "validation": copy.deepcopy(self.validation),
            "package": copy.deepcopy(self.package),
        }

    # ---- 便捷访问 ----
    @property
    def model_type(self) -> str:
        return str(self.source.get("model_type") or self.name)

    @property
    def framework(self) -> str:
        return str(self.source.get("framework") or "pytorch")

    @property
    def model_file(self) -> str:
        return str(self.source.get("file") or "")

    @property
    def hardware(self) -> str:
        return str(self.target.get("hardware") or "kunlun")

    @property
    def precision(self) -> str:
        return str(self.target.get("precision") or "fp16")

    @property
    def input_shape(self) -> List[int]:
        return normalize_shape(self.input.get("shape") or [1, 3, 640, 640])

    @property
    def package_name(self) -> str:
        return str(self.package.get("name") or "{}_dockerimg_{}".format(self.name, self.version))

    def to_model_override(self) -> Dict[str, Any]:
        """转换为 ModelConfig 的覆盖字段。"""
        return {
            "model_type": self.model_type,
            "framework": self.framework,
            "task": self.task,
            "input_shape": self.input_shape,
            "input_dtype": self.input.get("dtype", "float32"),
        }

    @property
    def task(self) -> str:
        return str(self.raw.get("task") or self.source.get("task") or "detection")
