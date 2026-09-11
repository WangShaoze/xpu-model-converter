# -*- coding: utf-8 -*-
"""昆仑芯后端配置对象。

与 :class:`xpu_converter.config.HardwareConfig` 的差别:
``HardwareConfig`` 描述"硬件这一层的完整交付参数", 本类只保留编译/推理真正
需要的字段, 便于后续替换 SDK 时保持接口稳定。
"""
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from xpu_converter.errors import ConfigError

# 可选 SDK 适配器: auto 按优先级自动探测, stub 为无 SDK 时的离线占位
SDK_ADAPTERS = ("auto", "paddle", "xpuctl", "stub")
PRECISIONS = ("fp32", "fp16")


@dataclass
class KunlunConfig:
    """昆仑芯编译/运行配置。"""

    sdk_adapter: str = "auto"
    sdk_module: str = ""
    target_chip: str = "auto"
    precision: str = "fp16"
    device: str = "auto"
    optimization_level: int = 2
    num_classes: int = 80
    output_layout: str = "auto"
    model_name: str = "model"
    # SDK 缺失时是否允许生成占位产物。默认 False: 生产链路在缺少真实后端时直接失败,
    # 只有显式打开(CLI --allow-degraded)才允许生成仅供联调的占位件。
    allow_degraded: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.sdk_adapter = str(self.sdk_adapter or "auto").lower()
        self.precision = str(self.precision or "fp16").lower()
        if self.sdk_adapter not in SDK_ADAPTERS:
            raise ConfigError(
                "不支持的 sdk_adapter: {!r}, 可选 {}".format(self.sdk_adapter, ", ".join(SDK_ADAPTERS))
            )
        if self.precision not in PRECISIONS:
            raise ConfigError(
                "不支持的 precision: {!r}, 可选 {}".format(self.precision, ", ".join(PRECISIONS))
            )
        try:
            self.optimization_level = int(self.optimization_level)
        except (TypeError, ValueError):
            raise ConfigError("optimization_level 必须是整数: {!r}".format(self.optimization_level))
        if self.optimization_level < 0:
            raise ConfigError("optimization_level 不能为负数")

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "KunlunConfig":
        data = dict(data or {})
        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        kwargs = {k: v for k, v in data.items() if k in known}
        cfg = cls(**kwargs)
        cfg.extra = {k: copy.deepcopy(v) for k, v in data.items() if k not in known}
        return cfg

    @classmethod
    def from_yaml(cls, path) -> "KunlunConfig":
        from xpu_converter.config import load_yaml

        return cls.from_dict(load_yaml(path))

    @classmethod
    def from_hardware_config(cls, hardware, **overrides) -> "KunlunConfig":
        """由 ``HardwareConfig`` 派生, ``overrides`` 优先级最高。"""
        data = {
            "sdk_adapter": getattr(hardware, "sdk_adapter", "auto"),
            "sdk_module": getattr(hardware, "sdk_module", ""),
            "target_chip": getattr(hardware, "target_chip", "auto"),
            "precision": getattr(hardware, "precision", "fp16"),
            "device": getattr(hardware, "device", "auto"),
            "optimization_level": getattr(hardware, "optimization_level", 2),
            "allow_degraded": getattr(hardware, "allow_degraded", False),
        }
        data.update({k: v for k, v in (overrides or {}).items() if v is not None})
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sdk_adapter": self.sdk_adapter,
            "sdk_module": self.sdk_module,
            "target_chip": self.target_chip,
            "precision": self.precision,
            "device": self.device,
            "optimization_level": self.optimization_level,
            "num_classes": self.num_classes,
            "output_layout": self.output_layout,
            "model_name": self.model_name,
            "allow_degraded": self.allow_degraded,
            "extra": copy.deepcopy(self.extra),
        }
