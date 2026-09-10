# -*- coding: utf-8 -*-
"""常量折叠 Pass。

把所有输入都是常量 (initializer 或已折叠结果) 的算子直接用 numpy 求值, 结果固化为
新的 initializer。YOLO 图中大量 ``Shape / Gather / Concat / Reshape`` 子图会被彻底
消除, 显著减小下发到后端的图规模。
"""
from typing import Any, Dict, List, Optional

import numpy as np

from xpu_converter.ir import onnx as onnx_ir
from xpu_converter.ir.graph import Graph
from xpu_converter.optimizer.base import GraphPass, PassResult

_BINARY_OPS = {
    "Add": np.add,
    "Sub": np.subtract,
    "Mul": np.multiply,
    "Div": np.divide,
    "Pow": np.power,
    "Max": np.maximum,
    "Min": np.minimum,
}

_UNARY_OPS = {
    "Neg": np.negative,
    "Abs": np.abs,
    "Sqrt": np.sqrt,
    "Exp": np.exp,
    "Log": np.log,
    "Floor": np.floor,
    "Ceil": np.ceil,
    "Round": np.round,
    "Reciprocal": lambda x: 1.0 / x,
    "Relu": lambda x: np.maximum(x, 0),
    "Sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
    "Tanh": np.tanh,
}


class ConstantFoldPass(GraphPass):
    name = "constant_fold"

    def applicable(self, graph: Graph) -> bool:
        return graph.has_raw and onnx_ir.available()

    # ------------------------------------------------------------- 工具
    @staticmethod
    def _attr(onnx, node, name: str, default: Any = None) -> Any:
        for attr in node.attribute:
            if attr.name == name:
                return onnx.helper.get_attribute_value(attr)
        return default

    def _axes(self, onnx, node, consts: Dict[str, np.ndarray]) -> List[int]:
        """取 Unsqueeze/Squeeze 的 axes (opset>=13 为输入, 早期版本为属性)。"""
        if len(node.input) > 1 and node.input[1] in consts:
            return [int(v) for v in np.asarray(consts[node.input[1]]).reshape(-1)]
        axes = self._attr(onnx, node, "axes") or []
        return [int(v) for v in axes]

    def _eval_node(
        self,
        onnx,
        node,
        consts: Dict[str, np.ndarray],
        static_shapes: Dict[str, List[int]],
    ) -> Optional[np.ndarray]:
        """尝试对单输出节点求值, 无法求值返回 None。"""
        op = node.op_type
        inputs = list(node.input)

        if op in _BINARY_OPS and len(inputs) == 2 and all(i in consts for i in inputs):
            return _BINARY_OPS[op](consts[inputs[0]], consts[inputs[1]])
        if op in _UNARY_OPS and len(inputs) >= 1 and inputs[0] in consts:
            return _UNARY_OPS[op](consts[inputs[0]])
        if op == "Cast" and inputs[0] in consts:
            dtype = onnx.helper.tensor_dtype_to_np_dtype(int(self._attr(onnx, node, "to")))
            return consts[inputs[0]].astype(dtype)
        if op == "Transpose" and inputs[0] in consts:
            perm = self._attr(onnx, node, "perm")
            return np.transpose(consts[inputs[0]], perm) if perm else np.transpose(consts[inputs[0]])
        if op == "Concat" and inputs and all(i in consts for i in inputs):
            axis = int(self._attr(onnx, node, "axis", 0))
            return np.concatenate([consts[i] for i in inputs], axis=axis)
        if op == "Unsqueeze" and inputs[0] in consts:
            value = consts[inputs[0]]
            for axis in sorted(self._axes(onnx, node, consts)):
                value = np.expand_dims(value, axis=int(axis))
            return value
        if op == "Squeeze" and inputs[0] in consts:
            value = consts[inputs[0]]
            axes = self._axes(onnx, node, consts)
            return np.squeeze(value, axis=tuple(axes)) if axes else np.squeeze(value)
        if op == "Reshape" and len(inputs) == 2 and inputs[0] in consts and inputs[1] in consts:
            return np.reshape(consts[inputs[0]], consts[inputs[1]].astype(np.int64))
        if op == "Gather" and len(inputs) == 2 and inputs[0] in consts and inputs[1] in consts:
            axis = int(self._attr(onnx, node, "axis", 0))
            return np.take(consts[inputs[0]], consts[inputs[1]].astype(np.int64), axis=axis)
        if op == "Clip" and inputs[0] in consts:
            low = None
            high = None
            if len(inputs) > 1 and inputs[1] in consts:
                low = consts[inputs[1]]
            elif self._attr(onnx, node, "min") is not None:
                low = self._attr(onnx, node, "min")
            if len(inputs) > 2 and inputs[2] in consts:
                high = consts[inputs[2]]
            elif self._attr(onnx, node, "max") is not None:
                high = self._attr(onnx, node, "max")
            value = consts[inputs[0]]
            if low is not None:
                value = np.maximum(value, low)
            if high is not None:
                value = np.minimum(value, high)
            return value
        if op == "Shape" and len(inputs) == 1:
            name = inputs[0]
            if name in static_shapes:
                return np.array(static_shapes[name], dtype=np.int64)
            if name in consts:
                return np.array(consts[name].shape, dtype=np.int64)
        return None

    # ------------------------------------------------------------- 主流程
    def run(self, graph: Graph) -> PassResult:
        if not self.applicable(graph):
            return PassResult(self.name, skipped="需要 onnx 依赖与原始 ModelProto")
        onnx = onnx_ir.require_onnx()
        model = graph.raw

        consts: Dict[str, np.ndarray] = {
            init.name: onnx.numpy_helper.to_array(init) for init in model.graph.initializer
        }
        static_shapes: Dict[str, List[int]] = {}
        for vi in list(model.graph.input) + list(model.graph.value_info):
            dims = vi.type.tensor_type.shape.dim
            shape: List[int] = []
            complete = True
            for dim in dims:
                if dim.HasField("dim_value") and int(dim.dim_value) > 0:
                    shape.append(int(dim.dim_value))
                else:
                    complete = False
                    break
            if complete and shape:
                static_shapes[vi.name] = shape

        graph_outputs = {out.name for out in model.graph.output}
        kept_nodes = []
        folded_names: List[str] = []
        new_initializers = []
        for node in model.graph.node:
            # 多输出节点与图输出节点不折叠, 避免改变对外接口
            if len(node.output) != 1 or node.output[0] in graph_outputs:
                kept_nodes.append(node)
                continue
            try:
                value = self._eval_node(onnx, node, consts, static_shapes)
            except Exception:
                value = None
            if value is None:
                kept_nodes.append(node)
                continue
            out_name = node.output[0]
            value = np.asarray(value)
            consts[out_name] = value
            new_initializers.append(onnx.numpy_helper.from_array(value, name=out_name))
            folded_names.append(out_name)

        if not folded_names:
            return PassResult(self.name, changed=False, details={"folded": 0})

        del model.graph.node[:]
        model.graph.node.extend(kept_nodes)
        for name in folded_names:
            onnx_ir.remove_initializer(model, name)
        model.graph.initializer.extend(new_initializers)

        graph.refresh()
        for name in folded_names:
            graph.value_info.pop(name, None)
        return PassResult(self.name, changed=True, details={"folded": len(folded_names)})
