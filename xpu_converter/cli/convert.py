# -*- coding: utf-8 -*-
"""``xpu-converter convert`` / ``build`` / ``export-onnx``(建设目标 §2 / §13 / §14)。"""
from pathlib import Path
from typing import Optional

from xpu_converter.cli.common import (
    DEFAULT_HARDWARE,
    adapter_for,
    input_shape_from,
    print_kv,
    save_json,
)
from xpu_converter.config import BuildManifest
from xpu_converter.errors import ConfigError
from xpu_converter.ir import onnx as onnx_ir
from xpu_converter.pipeline import ConversionPipeline
from xpu_converter.version import CONVERTER_VERSION


def _from_manifest(args) -> Optional[BuildManifest]:
    path = getattr(args, "manifest", None)
    if not path:
        return None
    if not Path(path).is_file():
        raise ConfigError("manifest 文件不存在: {}".format(path))
    return BuildManifest.from_yaml(path)


def _run_pipeline(args, default_output: str) -> int:
    """合并 ``--manifest``(建设目标 §14)与命令行参数, 命令行优先。"""
    manifest = _from_manifest(args)
    base_dir = Path(args.manifest).resolve().parent if manifest else Path.cwd()

    model_path = getattr(args, "model", None)
    if not model_path and manifest and manifest.model_file:
        model_path = str(base_dir / manifest.model_file)
    if not model_path:
        raise ConfigError("请通过 --model 指定模型, 或通过 --manifest 提供 source.file")

    shape = input_shape_from(getattr(args, "input_shape", None))
    if not shape and manifest:
        shape = manifest.input_shape
    version = getattr(args, "version", None) or (manifest.version if manifest else "v1.0")
    package_name = getattr(args, "package_name", None) or (manifest.name if manifest else None)
    hardware = getattr(args, "hardware", None) or (manifest.hardware if manifest else DEFAULT_HARDWARE)
    precision = getattr(args, "precision", None) or (manifest.precision if manifest else None)
    runtime = getattr(args, "runtime", None) or (
        (manifest.runtime or {}).get("type") if manifest else None) or "detection"
    dataset = getattr(args, "validation_dataset", None) or (
        (manifest.validation or {}).get("dataset") if manifest else None) or None
    validation_enabled = not getattr(args, "no_validate", False)
    if manifest:
        validation_enabled = validation_enabled and bool((manifest.validation or {}).get("enabled", True))

    pipeline = ConversionPipeline(
        model_path=model_path,
        output_dir=getattr(args, "output", None) or default_output,
        model_type=getattr(args, "model_type", None) or (manifest.model_type if manifest else None),
        hardware=hardware,
        precision=precision,
        input_shape=shape,
        runtime=runtime,
        version=version,
        package_name=package_name,
        optimization_level=getattr(args, "optimization_level", None),
        device=getattr(args, "exec_device", None),
        sdk_adapter=getattr(args, "sdk_adapter", None),
        validation_dataset=dataset,
        validation_enabled=validation_enabled,
        benchmark_iterations=getattr(args, "benchmark_iterations", 50) or 50,
        export_docker=not getattr(args, "no_docker", False),
        allow_degraded=bool(getattr(args, "allow_degraded", False)),
        allow_degraded_package=bool(getattr(args, "dev_package", False)),
    )
    if manifest:
        print("已加载 manifest: {} (name={}, version={})".format(
            args.manifest, manifest.name, manifest.version))
    result = pipeline.run()
    _report(result, args)
    return 0


def _report(result, args) -> None:
    print("")
    print_kv("转换完成", {
        "模型": "{} / {}".format(result.framework, result.model_type),
        "任务": result.task,
        "目标硬件": result.hardware,
        "精度": result.precision,
        "输入形状": result.input_shape,
        "中间模型(ONNX)": result.onnx_path,
        "优化后 ONNX": result.optimized_onnx_path,
        "编译产物": result.artifact.model_path if result.artifact else "-",
        "SDK 适配器": result.artifact.sdk_adapter if result.artifact else "-",
        "精度校验": result.accuracy.summary() if result.accuracy else "-",
        "性能基准": result.benchmark.summary() if result.benchmark else "-",
        "交付包": result.package_zip or "-",
        "转换器版本": CONVERTER_VERSION,
    })
    if result.degraded:
        print("")
        print("注意: 未检测到昆仑 SDK, 当前产物为占位件(degraded), 仅供链路联调, 不可对外交付。")
    if result.warnings:
        print("")
        print("提示:")
        for note in result.warnings:
            print("  - {}".format(note))
    if getattr(args, "json", None):
        save_json(result.to_dict(), args.json)
        print("已写出: {}".format(args.json))


def cmd_convert(args) -> int:
    """``convert``: 一条命令跑完 10 步并产出交付包。"""
    return _run_pipeline(args, "./output")


def cmd_build(args) -> int:
    """``build``: 与 convert 相同的一键入口, 默认输出到 ``dist/``。"""
    return _run_pipeline(args, "./dist")


def cmd_export_onnx(args) -> int:
    """``export-onnx``: 只导出 ONNX 中间模型(建设目标 §11 的 Intermediate Model)。"""
    output = getattr(args, "output", None)
    shape = input_shape_from(getattr(args, "input_shape", None))
    if output and Path(output).suffix.lower() == ".onnx":
        target = Path(output)
    else:
        target = Path(output or "./output/onnx") / "{}_raw.onnx".format(args.model_type or "model")
    target.parent.mkdir(parents=True, exist_ok=True)

    adapter = adapter_for(args.model, args.model_type, {"input_shape": shape} if shape else None)
    exported = adapter.export_onnx(
        args.model,
        str(target),
        input_shape=shape or adapter.input_shape(),
        opset=getattr(args, "opset", None) or adapter.opset(),
        dynamic=bool(getattr(args, "dynamic", False)),
    )

    graph = onnx_ir.load(exported)
    graph.infer_shapes()
    print_kv("ONNX 导出完成", {
        "文件": exported,
        "大小": "{:.2f} MB".format(Path(exported).stat().st_size / (1024.0 * 1024.0)),
        "opset": graph.opset,
        "输入": ["{} {}".format(t.name, t.shape) for t in graph.inputs],
        "输出": ["{} {}".format(t.name, t.shape) for t in graph.outputs],
        "节点数": len(graph.nodes),
        "动态 batch": bool(getattr(args, "dynamic", False)),
    })
    return 0
