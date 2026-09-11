# -*- coding: utf-8 -*-
"""转换环境指纹(ChatGPT 修改意见 §13)。

一份交付产物只有连同"它是在什么框架/库/解释器版本下、从哪个 checkpoint 转出来的"
一起记录, 现场问题才可复现。本模块把这些信息收敛成一个可序列化 dict。
"""
import hashlib
import importlib
import importlib.util
import platform
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from xpu_converter.version import CONVERTER_VERSION

# metadata.json 需要平铺的来源指纹字段名(ChatGPT 修改意见 §13 / §31)
SOURCE_FINGERPRINT_KEYS = (
    "source_framework",
    "source_framework_version",
    "source_library",
    "source_library_version",
    "python_version",
    "exporter_version",
    "checkpoint",
    "model_sha256",
)


def file_sha256(path: Optional[str]) -> str:
    """计算文件 SHA256; 路径为空或文件不存在时返回空串(ChatGPT 修改意见 §31)。"""
    if not path:
        return ""
    target = Path(path)
    if not target.is_file():
        return ""
    digest = hashlib.sha256()
    with open(target, "rb") as fr:
        for chunk in iter(lambda: fr.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# 框架 -> 顶层解释器模块名(取 __version__ 用)
FRAMEWORK_MODULES = {"pytorch": "torch", "paddle": "paddle"}
# 框架 -> 可能的上游模型库候选(按优先级)
FRAMEWORK_LIBRARIES = {
    "pytorch": ("ultralytics", "torchvision"),
    "paddle": ("paddledet", "paddleocr"),
}


def module_version(name: str) -> str:
    """返回模块版本号; 模块不存在或读取失败时返回空串。"""
    if not name:
        return ""
    try:
        if importlib.util.find_spec(name) is None:
            return ""
    except (ImportError, ValueError):
        return ""
    try:
        return str(getattr(importlib.import_module(name), "__version__", "") or "")
    except Exception:
        return ""


def first_installed(names) -> Tuple[str, str]:
    for name in names or ():
        version = module_version(name)
        if version:
            return name, version
    return "", ""


def source_fingerprint(
    framework: str = "pytorch",
    checkpoint: Optional[str] = None,
    library: Optional[str] = None,
    model_path: Optional[str] = None,
) -> Dict[str, Any]:
    """构造 metadata.json 要求的来源指纹字段。

    字段名对齐 ChatGPT 修改意见 §13:
    ``source_framework`` / ``source_framework_version`` / ``source_library`` /
    ``source_library_version`` / ``python_version`` / ``exporter_version``。

    传入 ``model_path`` 时额外计算源模型 SHA256(§31), 用于事后核对"交付的模型
    到底是不是客户给的那个 checkpoint"。
    """
    framework = str(framework or "pytorch").lower()
    if library:
        library_name, library_version = library, module_version(library)
    else:
        library_name, library_version = first_installed(FRAMEWORK_LIBRARIES.get(framework, ()))
    payload: Dict[str, Any] = {
        "source_framework": framework,
        "source_framework_version": module_version(FRAMEWORK_MODULES.get(framework, framework)),
        "source_library": library_name,
        "source_library_version": library_version,
        "python_version": platform.python_version(),
        "exporter_version": CONVERTER_VERSION,
    }
    if checkpoint:
        payload["checkpoint"] = str(checkpoint)
    digest = file_sha256(model_path)
    if digest:
        payload["model_sha256"] = digest
    return payload
