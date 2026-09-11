# -*- coding: utf-8 -*-
"""后端无关的图构建。

:class:`XpuGraph` 是"交给编译器"的统一输入: 已固化的静态输入规格、拓扑序算子
清单、以及落盘后的 ONNX 文件路径。任何后端 SDK 都只依赖这一层, 因此后续替换
昆仑 SDK 不会波及 Optimizer / Rewrite 层。
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from xpu_converter.errors import IrError
from xpu_converter.ir import onnx as onnx_ir

# 交付包内约定的模型文件名(建设目标 §6)
DEFAULT_MODEL_FILENAME = "model.xpu"
# 交给编译器的规范化 ONNX 文件名。刻意与最终产物(model.xpu / model.onnx /
# model.pdmodel)区分, 否则 stub 适配器复制时会与源文件同名(Windows/Linux 均报错)。
DEFAULT_ONNX_FILENAME = "compile_graph.onnx"


@dataclass
class XpuGraph:
    """待编译图描述。"""

    graph: Any
    precision: str = "fp16"
    input_names: List[str] = field(default_factory=list)
    input_shapes: List[List[int]] = field(default_factory=list)
    input_dtype: str = "float32"
    output_names: List[str] = field(default_factory=list)
    ops: List[str] = field(default_factory=list)
    onnx_path: str = ""
    notes: List[str] = field(default_factory=list)

    @property
    def op_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for op_type in self.ops:
            counts[op_type] = counts.get(op_type, 0) + 1
        return counts

    def summary(self) -> str:
        return "precision={} nodes={} inputs={} outputs={}".format(
            self.precision,
            len(self.ops),
            ",".join("x".join(str(d) for d in shape) for shape in self.input_shapes) or "-",
            ",".join(self.output_names) or "-",
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "precision": self.precision,
            "input_names": list(self.input_names),
            "input_shapes": [list(s) for s in self.input_shapes],
            "input_dtype": self.input_dtype,
            "output_names": list(self.output_names),
            "op_counts": self.op_counts,
            "node_count": len(self.ops),
            "onnx_path": self.onnx_path,
            "notes": list(self.notes),
        }


class XpuGraphBuilder:
    """把优化后的 IR 整理成后端可直接消费的 :class:`XpuGraph`。"""

    def __init__(self, backend_name: str = "kunlun") -> None:
        self.backend_name = backend_name

    def build(
        self,
        graph,
        precision: str = "fp16",
        workdir: Optional[str] = None,
        input_shapes: Optional[Dict[str, List[int]]] = None,
        model_filename: str = DEFAULT_ONNX_FILENAME,
    ) -> XpuGraph:
        """构建待编译图。

        - ``input_shapes``: 覆盖模型输入形状(通常来自 ``--input-shape``), 用于把
          V1 约定的静态 shape / batch=1 固化进图;
        - ``workdir``: 不为空时把规范化后的 ONNX 落盘, 供 SDK 读取。
        """
        if graph is None:
            raise IrError("图对象为空, 无法构建待编译图")
        if not getattr(graph, "has_raw", False):
            raise IrError("构建待编译图需要图带有原始 ONNX ModelProto (graph.raw 为空)")

        notes: List[str] = []
        overrides = {k: list(v) for k, v in (input_shapes or {}).items()}
        self._freeze_inputs(graph, overrides, notes)

        xpu_graph = XpuGraph(
            graph=graph,
            precision=str(precision or "fp32").lower(),
            input_names=[t.name for t in graph.inputs],
            input_shapes=[list(t.shape) for t in graph.inputs],
            input_dtype=graph.inputs[0].dtype if graph.inputs else "float32",
            output_names=[t.name for t in graph.outputs],
            ops=[node.op_type for node in graph.topological_order()],
            notes=notes,
        )

        if workdir:
            target = Path(workdir) / model_filename
            target.parent.mkdir(parents=True, exist_ok=True)
            onnx_ir.save_model(graph.raw, str(target))
            xpu_graph.onnx_path = str(target)
        return xpu_graph

    # ------------------------------------------------------------------ 内部
    @staticmethod
    def _freeze_inputs(graph, overrides: Dict[str, List[int]], notes: List[str]) -> None:
        """把动态维度固化为静态值, 并应用调用方给定的输入形状。"""
        changed = False
        for tensor in graph.inputs:
            override = overrides.get(tensor.name)
            if override:
                if list(tensor.shape) != list(override):
                    XpuGraphBuilder._set_input_shape(graph, tensor.name, override)
                    notes.append("输入 {} 形状覆盖为 {}".format(tensor.name, override))
                    changed = True
                continue
            if tensor.has_dynamic_dim():
                static_shape = [1 if (d is None or int(d) < 0) else int(d) for d in tensor.shape]
                if static_shape and static_shape[0] != 1:
                    static_shape[0] = 1
                XpuGraphBuilder._set_input_shape(graph, tensor.name, static_shape)
                notes.append("输入 {} 动态维度固化为 {}".format(tensor.name, static_shape))
                changed = True
        if changed:
            graph.refresh()

    @staticmethod
    def _set_input_shape(graph, name: str, shape: List[int]) -> None:
        """就地改写 ``raw`` 中该输入的静态形状 (protobuf repeated field 原地替换)。"""
        for value_info in graph.raw.graph.input:
            if value_info.name != name:
                continue
            dimensions = value_info.type.tensor_type.shape.dim
            del dimensions[:]
            for dim in shape:
                dimensions.add().dim_value = int(dim)
        for tensor in graph.inputs:
            if tensor.name == name:
                tensor.shape = list(shape)


def write_compile_report(xpu_graph: XpuGraph, output_dir: str, extra: Optional[Dict[str, Any]] = None) -> str:
    """把待编译图信息写成 ``compile_report.json``, 便于排查 SDK 侧问题。"""
    path = Path(output_dir) / "compile_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = xpu_graph.to_dict()
    if extra:
        payload.update(extra)
    with open(path, "w", encoding="utf-8") as fw:
        json.dump(payload, fw, ensure_ascii=False, indent=2)
    return str(path)
