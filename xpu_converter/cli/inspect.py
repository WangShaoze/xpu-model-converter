# -*- coding: utf-8 -*-
"""``xpu-converter inspect`` / ``xpu-converter analyze``(建设目标 §13)。"""
from typing import Any, Dict

from xpu_converter.cli.common import (
    adapter_for,
    backend_for,
    hardware_config_from,
    input_shape_from,
    print_kv,
    save_json,
)
from xpu_converter.errors import ConfigError
from xpu_converter.ir import onnx as onnx_ir
from xpu_converter.paths import model_config_path


def cmd_inspect(args) -> int:
    """``inspect best.pt``: 识别模型并打印元信息。"""
    overrides: Dict[str, Any] = {}
    shape = input_shape_from(getattr(args, "input_shape", None))
    if shape:
        overrides["input_shape"] = shape
    adapter = adapter_for(args.model, args.model_type, overrides)
    frontend = adapter.inspect(args.model)

    print_kv("模型信息", {
        "文件": args.model,
        "框架": frontend.framework,
        "模型类型": frontend.model_type,
        "任务": frontend.task,
        "输入": "{} ({})".format(frontend.input_shape, frontend.input_dtype),
        "类别数": frontend.num_classes,
        "端到端(含 NMS)": frontend.end2end,
        "配置文件": str(model_config_path(frontend.model_type)) if args.model_type else "自动推断",
        "ONNX opset": adapter.opset(),
    })
    print_kv("类别名", {
        "前 10 个": frontend.class_names[:10],
        "总数": len(frontend.class_names),
    })
    if frontend.extra:
        print_kv("其它", frontend.extra)

    if getattr(args, "hardware", None):
        config = hardware_config_from(args)
        backend = backend_for(args, config)
        detail = backend.describe()
        print_kv("目标后端", {
            "后端": detail.get("backend"),
            "SDK 适配器": detail.get("sdk_adapter"),
            "是否降级": detail.get("degraded"),
            "精度": detail.get("precision"),
            "昆仑芯型号": detail.get("target_chip"),
        })
    if getattr(args, "json", None):
        save_json(frontend.to_dict(), args.json)
        print("已写出: {}".format(args.json))
    return 0


def cmd_analyze(args) -> int:
    """``analyze model.onnx --device kunlun``: 分析算子在目标后端上的支持情况。"""
    from xpu_converter.registry.model_registry import resolve_model_config

    path = str(args.model)
    if not onnx_ir.available():
        raise ConfigError("缺少 onnx 依赖, 请执行: pip install onnx")
    graph = onnx_ir.load(path)
    graph.infer_shapes()

    model_type = args.model_type or "yolov10"
    model_config = resolve_model_config(model_type)
    backend = backend_for(args)
    analysis = backend.analyze(graph)

    print(graph.summary())
    print("")
    print_kv("算子分析", {
        "汇总": analysis.summary(),
        "总计": analysis.total,
        "支持": analysis.supported,
        "需改写": analysis.rewritten,
        "不支持": analysis.unsupported,
        "结论": "OK" if analysis.ok else "存在不支持算子",
    })
    if analysis.rewrite_ops:
        print_kv("改写算子", analysis.rewrite_ops)
    if analysis.unsupported_ops:
        print_kv("不支持算子", {"列表": analysis.unsupported_ops})
    if analysis.domain_ops:
        print_kv("自定义域", {"列表": analysis.domain_ops})

    config = hardware_config_from(args)
    print_kv("目标后端", {
        "后端": backend.name,
        "精度": config.precision,
        "SDK 适配器": getattr(backend, "config", config).sdk_adapter,
        "输入 layout": model_config.input_layout,
    })
    if getattr(args, "json", None):
        save_json({
            "graph": graph.to_dict(),
            "analysis": analysis.to_dict(),
            "backend": backend.describe(),
        }, args.json)
        print("已写出: {}".format(args.json))
    return 0 if analysis.ok else 1
