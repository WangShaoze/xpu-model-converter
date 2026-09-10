# -*- coding: utf-8 -*-
"""``xpu-converter compile``(建设目标 §13): ONNX → XPU 编译产物。"""
from pathlib import Path

from xpu_converter.backend.kunlun.graph_builder import XpuGraphBuilder
from xpu_converter.cli.common import (
    backend_for,
    hardware_config_from,
    input_shape_from,
    print_kv,
    save_json,
)
from xpu_converter.errors import ConfigError
from xpu_converter.ir import onnx as onnx_ir


def cmd_compile(args) -> int:
    """``compile --model model.onnx --device kunlun --precision fp16``。"""
    source = Path(args.model)
    if not source.is_file():
        raise ConfigError("待编译模型不存在: {}".format(source))
    if source.suffix.lower() != ".onnx":
        raise ConfigError(
            "compile 只接受 ONNX 中间模型; 请先执行 export-onnx, 或直接使用 convert 一键完成"
        )
    if not onnx_ir.available():
        raise ConfigError("缺少 onnx 依赖, 请执行: pip install onnx")

    config = hardware_config_from(args)
    backend = backend_for(args, config)

    output_dir = Path(getattr(args, "output", None) or "./output/xpu")
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "model.xpu"

    graph = onnx_ir.load(str(source))
    graph.infer_shapes()
    shape = input_shape_from(getattr(args, "input_shape", None))
    input_shapes = {}
    if shape and graph.inputs:
        input_shapes[graph.inputs[0].name] = list(shape)

    xpu_graph = XpuGraphBuilder(config.name).build(
        graph, precision=config.precision, workdir=str(output_dir), input_shapes=input_shapes
    )
    for note in xpu_graph.notes:
        print("  - {}".format(note))

    artifact = backend.compile(xpu_graph, str(artifact_path))

    print_kv("编译完成", {
        "输入": str(source),
        "产物": artifact.model_path,
        "精度": artifact.precision,
        "SDK 适配器": artifact.sdk_adapter,
        "昆仑芯型号": artifact.target_chip,
        "产物格式": artifact.artifact_format,
        "是否降级": artifact.degraded,
        "算子数": len(xpu_graph.ops),
        "输入形状": xpu_graph.input_shapes,
        "输出": xpu_graph.output_names,
    })
    for note in artifact.notes:
        print("  ! {}".format(note))
    if artifact.degraded:
        print("")
        print("注意: 未检测到昆仑 SDK(kunlun), 产物为占位件, 不能作为最终交付物。")
    if getattr(args, "json", None):
        save_json({"artifact": artifact.to_dict(), "graph": xpu_graph.to_dict()}, args.json)
        print("已写出: {}".format(args.json))
    return 0
