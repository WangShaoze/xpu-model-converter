# -*- coding: utf-8 -*-
"""``xpu-converter inspect`` / ``analyze`` / ``list-models`` / ``list-capabilities``。"""
from pathlib import Path
from typing import Any, Dict, List

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
from xpu_converter.paths import hardware_capabilities_path, model_config_path


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


def cmd_list_models(args) -> int:
    """``list-models``: 列出模型的生命周期状态(ChatGPT 修改意见 §50/P0-7)。

    "代码里有 adapter" 不等于 "承诺支持"。除生命周期声明外, 这里自动附加结构就绪
    (``ready``)口径 —— Adapter/YAML/OutputContract/Runtime 是否齐全(P0-7), 把
    声明状态与自动检测结果并列, 一眼看出"哪一句 stable 可能是假完成"。
    """
    from xpu_converter.registry.model_registry import available_model_types, supported_models

    try:
        loaded = set(available_model_types())
    except Exception:  # pragma: no cover - 依赖缺失时仍应能列出生命周期表
        loaded = set()

    rows: List[Dict[str, Any]] = []
    seen = set()
    for item in supported_models():
        seen.add(item.model_type)
        ready, gap = _readiness_summary(item.model_type, loaded)
        rows.append({
            "model_type": item.model_type,
            "framework": item.framework,
            "status": item.status.upper(),
            "adapter": "yes" if item.model_type in loaded else "no",
            "ready": ready,
            "gap": gap,
            "notes": item.notes,
        })
    for model_type in sorted(loaded - seen):
        ready, gap = _readiness_summary(model_type, loaded)
        rows.append({"model_type": model_type, "framework": "-", "status": "EXPERIMENTAL",
                     "adapter": "yes", "ready": ready, "gap": gap,
                     "notes": "未在生命周期表登记"})

    wanted = getattr(args, "status", None)
    if wanted:
        rows = [row for row in rows if row["status"].lower() == str(wanted).lower()]

    _print_table(rows, ("model_type", "framework", "status", "adapter", "ready", "gap", "notes"))
    if getattr(args, "json", None):
        save_json({"models": rows}, args.json)
        print("已写出: {}".format(args.json))
    return 0


def _readiness_summary(model_type: str, loaded: set) -> tuple:
    """自动就绪检测摘要(ready/缺口), 失败降级为 unknown。"""
    from xpu_converter.registry.readiness import check_model_readiness

    try:
        readiness = check_model_readiness(model_type, loaded_adapters=loaded)
    except Exception:  # pragma: no cover - 属核心链路的防御
        return "?", "detect-error"
    if readiness.structural_ok:
        return "OK", ""
    return "MISSING", ",".join(readiness.missing())


def cmd_list_capabilities(args) -> int:
    """``list-capabilities --hardware kunlun``: 打印算子能力表与硬件能力指纹。"""
    from xpu_converter.capability import OperatorCapabilitySet, probe_kunlun

    config = hardware_config_from(args)
    cap_path = hardware_capabilities_path(config.name)
    capability_set = (
        OperatorCapabilitySet.from_yaml(cap_path)
        if Path(cap_path).is_file() else OperatorCapabilitySet()
    )
    hardware = probe_kunlun(getattr(config, "target_chip", "auto"), getattr(config, "device", "auto"))

    print_kv("算子能力表", {
        "文件": str(cap_path) if Path(cap_path).is_file() else "(缺失, 使用空能力表)",
        "版本": capability_set.version,
        "目标芯片": capability_set.target_chip,
        "算子数": len(capability_set.operators),
        "需改写算子": capability_set.rewrite_ops,
    })
    print_kv("硬件能力", {
        "后端": hardware.name,
        "芯片": hardware.chip,
        "SDK 版本": hardware.sdk_version,
        "设备数": hardware.device_count,
        "设备可用": hardware.device_available,
        "支持精度": sorted(hardware.supported_precisions),
        "支持 layout": sorted(hardware.layouts),
        "动态 shape": hardware.dynamic_shape,
        "paddle": hardware.paddle_version,
    })
    if hardware.notes:
        print_kv("备注", {"提示": hardware.notes})
    if getattr(args, "json", None):
        save_json({
            "operator_capability": capability_set.to_dict(),
            "hardware": hardware.to_dict(),
        }, args.json)
        print("已写出: {}".format(args.json))
    return 0


def _print_table(rows: List[Dict[str, Any]], columns) -> None:
    """等宽列打印(避免额外依赖)。"""
    widths = {key: len(key) for key in columns}
    for row in rows:
        for key in columns:
            widths[key] = max(widths[key], len(str(row.get(key, ""))))
    header = "  ".join("{:<{w}}".format(key, w=widths[key]) for key in columns)
    print(header)
    print("-" * len(header))
    for row in rows:
        print("  ".join("{:<{w}}".format(str(row.get(key, "")), w=widths[key]) for key in columns))
