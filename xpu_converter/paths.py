# -*- coding: utf-8 -*-
"""资源路径解析。

``templates/`` / ``configs/`` / ``runtime/`` 默认位于仓库根目录（与包同级），
安装为 wheel 后也可通过环境变量 ``XPU_CONVERTER_HOME`` 指定。
"""
import os
from pathlib import Path
from typing import List, Optional

from xpu_converter.errors import ConfigError

PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = PACKAGE_DIR.parent


def _first_dir(candidates: List[Optional[Path]]) -> Optional[Path]:
    for cand in candidates:
        if cand and Path(cand).is_dir():
            return Path(cand).resolve()
    return None


def home_dir() -> Path:
    """仓库/安装根目录。"""
    env_home = os.environ.get("XPU_CONVERTER_HOME")
    found = _first_dir([Path(env_home) if env_home else None, _REPO_ROOT, PACKAGE_DIR])
    if found is None:
        raise ConfigError(
            "无法定位 xpu-model-converter 资源目录，请设置环境变量 XPU_CONVERTER_HOME 指向项目根目录"
        )
    return found


def _resource(name: str) -> Path:
    env_home = os.environ.get("XPU_CONVERTER_HOME")
    found = _first_dir(
        [
            Path(env_home) / name if env_home else None,
            _REPO_ROOT / name,
            PACKAGE_DIR / name,
        ]
    )
    if found is None:
        raise ConfigError("缺少资源目录: {}（可通过 XPU_CONVERTER_HOME 指定项目根目录）".format(name))
    return found


def templates_dir() -> Path:
    return _resource("templates")


def configs_dir() -> Path:
    return _resource("configs")


def runtime_dir() -> Path:
    return _resource("runtime")


def models_config_dir() -> Path:
    return configs_dir() / "models"


def hardware_config_dir() -> Path:
    return configs_dir() / "hardware"


def deployment_config_dir() -> Path:
    """交付/部署环境配置目录 (``configs/deployment``)。

    与 Hardware Capability 分离: base_image / packages / minio / kafka 属于
    **部署环境**, 不应混进昆仑 XPU 硬件能力配置(ChatGPT 修改意见 §26/§27)。
    """
    return configs_dir() / "deployment"


def docker_template_dir(hardware: str = "kunlun") -> Path:
    return templates_dir() / "docker" / hardware


def model_config_path(model_type: str) -> Path:
    return models_config_dir() / "{}.yaml".format(model_type)


def hardware_config_path(hardware: str) -> Path:
    return hardware_config_dir() / "{}.yaml".format(hardware)


def hardware_capabilities_path(hardware: str) -> Path:
    """算子能力表路径 (``configs/hardware/<hardware>_capabilities.yaml``)。"""
    return hardware_config_dir() / "{}_capabilities.yaml".format(hardware)


def deployment_config_path(hardware: str) -> Path:
    """交付/部署配置路径 (``configs/deployment/<hardware>_docker.yaml``)。"""
    return deployment_config_dir() / "{}_docker.yaml".format(hardware)
