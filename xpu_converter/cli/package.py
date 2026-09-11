# -*- coding: utf-8 -*-
"""``xpu-converter package``(建设目标 §13): 由已有 XPU 产物生成 Docker 交付包。"""
from pathlib import Path

from xpu_converter.cli.common import (
    artifact_from_model,
    hardware_config_from,
    input_shape_from,
    load_sidecar_metadata,
    print_kv,
)
from xpu_converter.config import BuildManifest
from xpu_converter.errors import ConfigError
from xpu_converter.exporter.docker_exporter import DockerExporter
from xpu_converter.registry.model_registry import resolve_model_config


def cmd_package(args) -> int:
    """``package --model model.xpu --model-type yolov10 --output ./package``。"""
    model_path = Path(args.model)
    if not model_path.is_file():
        raise ConfigError("待打包产物不存在: {}".format(model_path))

    config = hardware_config_from(args)
    model_type = args.model_type or _infer_model_type_from_metadata(model_path) or "yolov10"
    shape = input_shape_from(getattr(args, "input_shape", None))
    overrides = {"input_shape": shape} if shape else None
    model_config = resolve_model_config(model_type, overrides)

    artifact = artifact_from_model(str(model_path), config.precision, config.target_chip)
    name = getattr(args, "package_name", None) or model_type
    version = getattr(args, "version", None) or "v1.0"
    manifest = BuildManifest.from_dict({
        "name": name,
        "version": version,
        "task": model_config.task,
        "source": {"framework": model_config.framework, "model_type": model_type,
                   "file": model_path.name},
        "input": {"shape": list(model_config.input_shape), "dtype": model_config.input_dtype},
        "target": {"hardware": config.name, "precision": config.precision, "batch_size": 1},
        "runtime": {"type": getattr(args, "runtime", None) or model_config.task,
                    "port": int(config.runtime.get("port", 58025))},
        "validation": {"enabled": False, "dataset": ""},
        "package": {"docker": True, "name": "{}-dockerimg_{}".format(name, version)},
    })

    exporter = DockerExporter(
        manifest=manifest,
        model_config=model_config,
        hardware_config=config,
        runtime=getattr(args, "runtime", None) or model_config.task,
        assets_dir=getattr(args, "assets_dir", None),
        allow_degraded=bool(getattr(args, "dev_package", False)),
    )
    output_dir = getattr(args, "output", None) or "./package"
    zip_path = exporter.export(
        model_path=str(model_path),
        output_dir=output_dir,
        artifact=artifact,
        onnx_path=str(model_path) if model_path.suffix.lower() == ".onnx" else None,
        image_tar=getattr(args, "image_tar", None),
    )

    print_kv("交付包完成", {
        "产物": str(model_path),
        "包名": manifest.package_name,
        "输出目录": str(Path(output_dir) / manifest.package_name),
        "交付 ZIP": zip_path,
        "SDK 适配器": artifact.sdk_adapter,
        "是否降级": artifact.degraded,
        "镜像名": "{}:{}".format(name, version),
        "Web 端口": config.runtime.get("port", 58025),
    })
    for warning in exporter.warnings:
        print("  ! {}".format(warning))
    for note in artifact.notes:
        print("  ! {}".format(note))
    if artifact.degraded:
        print("")
        print("注意: 当前交付包内为占位产物(degraded), 需在拿到昆仑 SDK 后重新编译再交付。")
    return 0


def _infer_model_type_from_metadata(model_path: Path):
    """从 ``metadata.json`` / ``model.yaml`` 反查模型类型。"""
    metadata = load_sidecar_metadata(str(model_path))
    nested = metadata.get("metadata") if isinstance(metadata.get("metadata"), dict) else {}
    for source in (metadata, nested):
        value = source.get("model_type") if isinstance(source, dict) else None
        if value:
            return str(value)
    model_yaml = model_path.parent / "model.yaml"
    if model_yaml.is_file():
        try:
            import yaml

            with open(model_yaml, "r", encoding="utf-8") as fr:
                data = yaml.safe_load(fr) or {}
            value = (data.get("source") or {}).get("model_type")
            if value:
                return str(value)
        except Exception:
            return None
    return None
