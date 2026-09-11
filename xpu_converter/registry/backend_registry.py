# -*- coding: utf-8 -*-
"""硬件后端注册表。

把 ``--hardware kunlun`` 解析为具体的 :class:`xpu_converter.backend.base.BaseBackend`
实现, 并装配 ``configs/hardware/<hardware>.yaml``。

新增后端(如昇腾/寒武纪)只需实现 ``BaseBackend`` 并在此注册, 上层流水线与 CLI 不变。
"""
from typing import Any, Dict, List, Optional, Type

from xpu_converter.backend.base import BaseBackend
from xpu_converter.backend.kunlun.compiler import KunlunBackend
from xpu_converter.backend.kunlun.config import KunlunConfig
from xpu_converter.config import HardwareConfig, deep_merge, load_yaml
from xpu_converter.errors import ConfigError, NotSupportedError
from xpu_converter.paths import deployment_config_path, hardware_config_path

_BACKENDS: Dict[str, Type[BaseBackend]] = {}


def register_backend(name: str, backend_cls: Type[BaseBackend]) -> None:
    _BACKENDS[str(name).lower()] = backend_cls


def available_backends() -> List[str]:
    return sorted(_BACKENDS)


def get_backend_class(hardware: str) -> Type[BaseBackend]:
    key = str(hardware or "").strip().lower()
    if key not in _BACKENDS:
        raise NotSupportedError(
            "不支持的硬件后端: {!r}, 当前支持: {}".format(hardware, ", ".join(available_backends()))
        )
    return _BACKENDS[key]


def resolve_hardware_config(hardware: str, overrides: Optional[Dict[str, Any]] = None) -> HardwareConfig:
    """读取硬件配置并应用覆盖项(如 ``--precision fp16``)。

    配置分两处, 职责分离(ChatGPT 修改意见 §26/§27):
    - ``configs/hardware/<hardware>.yaml``   硬件能力(sdk/chip/precision/device)
    - ``configs/deployment/<hardware>_docker.yaml`` 部署环境(base_image/docker/log)
    两者合并为同一个 :class:`HardwareConfig`, 上层无需感知拆分。
    """
    get_backend_class(hardware)
    data: Dict[str, Any] = {}
    hw_path = hardware_config_path(hardware)
    if hw_path.is_file():
        data = deep_merge(data, load_yaml(hw_path))
    deploy_path = deployment_config_path(hardware)
    if deploy_path.is_file():
        data = deep_merge(data, load_yaml(deploy_path))
    config = HardwareConfig.from_dict(data) if data else HardwareConfig(name=hardware)
    for key, value in (overrides or {}).items():
        if value is None:
            continue
        if key in config.__dataclass_fields__:
            setattr(config, key, value)
        else:
            config.extra[key] = value
    return config


def create_backend(hardware: str, config: Optional[Any] = None,
                   overrides: Optional[Dict[str, Any]] = None) -> BaseBackend:
    """创建后端实例。"""
    backend_cls = get_backend_class(hardware)
    hardware_config = config if isinstance(config, HardwareConfig) else \
        resolve_hardware_config(hardware, overrides)
    if backend_cls is KunlunBackend:
        return KunlunBackend(KunlunConfig.from_hardware_config(hardware_config, **(overrides or {})))
    return backend_cls(hardware_config)


def kunlun_config_from(hardware_config: HardwareConfig, **overrides) -> KunlunConfig:
    """由硬件配置派生昆仑后端配置(供 CLI 直接使用)。"""
    if not isinstance(hardware_config, HardwareConfig):
        raise ConfigError("hardware_config 类型错误: {!r}".format(type(hardware_config)))
    return KunlunConfig.from_hardware_config(hardware_config, **overrides)


register_backend(KunlunBackend.name, KunlunBackend)
