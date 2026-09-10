# -*- coding: utf-8 -*-
"""运行时配置加载。

公共 Runtime 通过固定相对路径读取交付包内配置, 因此**同一套 Runtime 代码**
可以服务所有模型(建设目标 §5/§10):

    <RUNTIME_HOME>/config/runtime.yaml
    <RUNTIME_HOME>/config/confidence.json
    <RUNTIME_HOME>/model/model.xpu
    <RUNTIME_HOME>/model/metadata.json
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, List

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

RUNTIME_YAML = "config/runtime.yaml"
CONFIDENCE_JSON = "config/confidence.json"
MODEL_DIR = "model"
DEFAULT_MODEL_FILENAME = "model.xpu"
METADATA_FILENAME = "metadata.json"

DEFAULT_CONFIG: Dict[str, Any] = {
    "name": "model",
    "version": "v1.0",
    "task": "detection",
    "framework": "pytorch",
    "algorithm": {
        "name": "",
        "version": "v1.0",
        "type": "detection",
        "app_type": "",
        "component_code": "",
        "template_version": "v1.0",
    },
    "model": {"file": DEFAULT_MODEL_FILENAME},
    "input": {"shape": [1, 3, 640, 640], "dtype": "float32", "layout": "NCHW"},
    "preprocess": {"color": "RGB", "scale": 0.00392156862745098, "letterbox": True, "pad_value": 114},
    "postprocess": {"nms": "cpu", "conf_thres": 0.25, "iou_thres": 0.45, "max_det": 300,
                    "output_layout": "auto"},
    "num_classes": 80,
    "class_names": [],
    "runtime": {"api_version": "v1", "port": 58025, "workers": 1, "device": "auto",
                "host_ip": "127.0.0.1", "result_dir": "/tmp/nwai_result"},
    "logging": {"process_log": "/tmp/nwai_log/process.log"},
}


def runtime_home() -> Path:
    """定位 Runtime 根目录(交付包内 ``config/`` 与 ``model/`` 的父目录)。"""
    env_home = os.environ.get("RUNTIME_HOME")
    if env_home and Path(env_home).is_dir():
        return Path(env_home).resolve()
    here = Path(__file__).resolve().parent
    for candidate in (here, here.parent):
        if (candidate / RUNTIME_YAML).is_file():
            return candidate
    return here


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in (override or {}).items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_runtime_config(home: str = None) -> Dict[str, Any]:
    """读取 ``config/runtime.yaml``, 与默认值合并。"""
    root = Path(home) if home else runtime_home()
    path = root / RUNTIME_YAML
    data: Dict[str, Any] = {}
    if path.is_file() and yaml is not None:
        with open(path, "r", encoding="utf-8") as fr:
            data = yaml.safe_load(fr) or {}
    config = _merge(DEFAULT_CONFIG, data)
    config["_home"] = str(root)
    return config


def model_path(config: Dict[str, Any] = None) -> str:
    """编译模型路径。"""
    config = config or load_runtime_config()
    root = Path(config.get("_home") or runtime_home())
    filename = str((config.get("model") or {}).get("file") or DEFAULT_MODEL_FILENAME)
    return str(root / MODEL_DIR / filename)


def metadata_path(config: Dict[str, Any] = None) -> str:
    config = config or load_runtime_config()
    root = Path(config.get("_home") or runtime_home())
    return str(root / MODEL_DIR / METADATA_FILENAME)


def confidence_path(config: Dict[str, Any] = None) -> str:
    config = config or load_runtime_config()
    root = Path(config.get("_home") or runtime_home())
    return str(root / CONFIDENCE_JSON)


def load_model_metadata(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """读取编译产物侧 ``model/metadata.json``, 用于判断后端类型。"""
    path = Path(metadata_path(config))
    if not path.is_file():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fr:
            return json.load(fr) or {}
    except Exception:
        return {}


def class_names(config: Dict[str, Any] = None) -> List[str]:
    config = config or load_runtime_config()
    names = config.get("class_names") or []
    return [str(item) for item in names]


def algorithm_info(config: Dict[str, Any] = None) -> Dict[str, str]:
    config = config or load_runtime_config()
    algorithm = config.get("algorithm") or {}
    return {
        "algorithm": str(algorithm.get("name") or config.get("name") or "model"),
        "algorithm_name": str(algorithm.get("name") or config.get("name") or "model"),
        "algorithm_version": str(algorithm.get("version") or config.get("version") or "v1.0"),
        "algorithm_type": str(algorithm.get("type") or config.get("task") or "detection"),
        "algorithm_app_type": str(algorithm.get("app_type") or ""),
        "component_code": str(algorithm.get("component_code") or ""),
        "template_version": str(algorithm.get("template_version") or "v1.0"),
    }
