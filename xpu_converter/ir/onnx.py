# -*- coding: utf-8 -*-
"""ONNX 与 IR 之间的转换; 同时提供张量取值互转工具。

前端导出 ONNX 后, ``Graph.raw`` 指向原始 ``ModelProto``; 需要真实取值的 Pass
(常量折叠 / 算子融合) 通过 :func:`tensor_to_numpy` / :func:`numpy_to_tensor` 直接
操作 ``raw``。
"""
import os
from typing import Any, Dict, List, Optional

from xpu_converter.errors import IrError
from xpu_converter.ir.graph import Graph
from xpu_converter.ir.node import Node
from xpu_converter.ir.tensor import NAME_TO_ONNX_DTYPE, ONNX_DTYPE_TO_NAME, Tensor


def available() -> bool:
    """本机是否可用 onnx。"""
    try:
        import onnx  # noqa: F401
        return True
    except ImportError:
        return False


def require_onnx():
    try:
        import onnx
        return onnx
    except ImportError:
        raise IrError("缺少 onnx 依赖, 请执行: pip install onnx")


# --------------------------------------------------------------------- 属性
def _attr_to_python(onnx, attr) -> Any:
    """ONNX AttributeProto -> Python 值 (字节串解码; 张量/子图做结构化描述)。"""
    from onnx import AttributeProto, TensorProto

    if attr.type == AttributeProto.TENSOR:
        tensor = attr.t
        return {
            "__tensor__": True,
            "dtype": ONNX_DTYPE_TO_NAME.get(tensor.data_type, "float32"),
            "shape": [int(d) for d in tensor.dims],
        }
    if attr.type == AttributeProto.GRAPH:
        return {"__graph__": True, "nodes": len(attr.g.node)}
    value = onnx.helper.get_attribute_value(attr)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple)):
        return [v.decode("utf-8", errors="replace") if isinstance(v, bytes) else v for v in value]
    return value


def _tensor_from_value_info(vi, is_initializer: bool = False) -> Tensor:
    tensor_type = vi.type.tensor_type
    dtype = ONNX_DTYPE_TO_NAME.get(int(tensor_type.elem_type) or 1, "float32")
    shape: List[Optional[int]] = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            shape.append(int(dim.dim_value))
        else:
            shape.append(None)
    return Tensor(name=vi.name, shape=shape, dtype=dtype, is_initializer=is_initializer)


def _tensor_from_initializer(init) -> Tensor:
    return Tensor(
        name=init.name,
        shape=[int(d) for d in init.dims],
        dtype=ONNX_DTYPE_TO_NAME.get(int(init.data_type), "float32"),
        is_initializer=True,
    )


# --------------------------------------------------------------- Model <-> IR
def from_model(model, name: Optional[str] = None) -> Graph:
    """ONNX ModelProto -> IR Graph (``Graph.raw`` 指向 model)。"""
    onnx = require_onnx()
    graph_proto = model.graph
    initializer_names = {init.name for init in graph_proto.initializer}

    nodes = [
        Node(
            name=node.name or "node_{}".format(index),
            op_type=node.op_type,
            inputs=list(node.input),
            outputs=list(node.output),
            attributes={attr.name: _attr_to_python(onnx, attr) for attr in node.attribute},
            domain=node.domain or "",
        )
        for index, node in enumerate(graph_proto.node)
    ]
    inputs = [_tensor_from_value_info(vi) for vi in graph_proto.input if vi.name not in initializer_names]
    outputs = [_tensor_from_value_info(vi) for vi in graph_proto.output]
    initializers = {init.name: _tensor_from_initializer(init) for init in graph_proto.initializer}
    value_info = {vi.name: _tensor_from_value_info(vi) for vi in graph_proto.value_info}

    opset = 13
    for imp in model.opset_import:
        if imp.domain in ("", "ai.onnx"):
            opset = int(imp.version)

    graph = Graph(
        name=name or (graph_proto.name or "graph"),
        nodes=nodes,
        inputs=inputs,
        outputs=outputs,
        initializers=initializers,
        value_info=value_info,
        opset=opset,
    )
    graph.raw = model
    return graph


def _simple_attributes(attributes: Dict[str, Any]) -> Dict[str, Any]:
    """只保留 ONNX make_node 可直接接受的属性类型。"""
    result: Dict[str, Any] = {}
    for key, value in attributes.items():
        if isinstance(value, bool):
            result[key] = int(value)
        elif isinstance(value, (int, float, str)):
            result[key] = value
        elif isinstance(value, (list, tuple)) and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in value
        ):
            result[key] = list(value)
    return result


def to_model(graph: Graph):
    """IR Graph -> ONNX ModelProto (仅用于无 raw 的纯 IR 图)。"""
    onnx = require_onnx()

    nodes = [
        onnx.helper.make_node(
            item.op_type,
            list(item.inputs),
            list(item.outputs),
            name=item.name,
            domain=item.domain or "",
            **_simple_attributes(item.attributes),
        )
        for item in graph.nodes
    ]

    def _vi(tensor: Tensor):
        return onnx.helper.make_tensor_value_info(
            tensor.name, NAME_TO_ONNX_DTYPE.get(tensor.dtype, 1), list(tensor.shape) or None
        )

    graph_proto = onnx.helper.make_graph(
        nodes,
        graph.name or "graph",
        [_vi(t) for t in graph.inputs],
        [_vi(t) for t in graph.outputs],
        value_info=[_vi(t) for t in graph.value_info.values() if t.shape],
    )
    opset_imports = [onnx.helper.make_opsetid("", int(graph.opset or 13))]
    return onnx.helper.make_model(graph_proto, opset_imports=opset_imports)


def load_model(path):
    onnx = require_onnx()
    path = str(path)
    if not os.path.isfile(path):
        raise IrError("ONNX 模型不存在: {}".format(path))
    try:
        return onnx.load(path)
    except Exception as err:
        raise IrError("读取 ONNX 失败: {} ({})".format(path, err))


def save_model(model, path) -> str:
    onnx = require_onnx()
    path = str(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    onnx.save(model, path)
    return path


def load(path, name: Optional[str] = None) -> Graph:
    """读取 ONNX 文件为 IR Graph。"""
    return from_model(load_model(path), name=name)


def save(graph: Graph, path) -> str:
    """保存 IR Graph 为 ONNX 文件; 优先保存 raw。"""
    if graph.raw is not None:
        return save_model(graph.raw, path)
    return save_model(to_model(graph), path)


def infer_shapes(graph: Graph) -> bool:
    """对 raw 执行 shape inference 并刷新视图; 无 raw 或 onnx 缺失时返回 False。"""
    if graph.raw is None or not available():
        return False
    onnx = require_onnx()
    try:
        graph.raw = onnx.shape_inference.infer_shapes(graph.raw)
    except Exception:
        return False
    graph.refresh()
    return True


def check_model(model) -> Optional[str]:
    """用 ONNX checker 校验 ModelProto。

    合法返回 ``None``; 不合法或 onnx 缺失返回错误摘要字符串(ChatGPT 修改意见
    §35/§36: 每个 Pass/Rewrite 落地后都要过一遍图合法性检查, 失败则回滚)。
    """
    if model is None or not available():
        return None
    onnx = require_onnx()
    try:
        onnx.checker.check_model(model)
        return None
    except Exception as err:  # noqa: BLE001 - checker 异常类型不稳定, 一律视为不合法
        return "{}: {}".format(type(err).__name__, err)


# ------------------------------------------------------------ 张量取值工具
def tensor_to_numpy(tensor_proto):
    """ONNX TensorProto -> numpy 数组。"""
    onnx = require_onnx()
    return onnx.numpy_helper.to_array(tensor_proto)


def numpy_to_tensor(array, name: str):
    """numpy 数组 -> ONNX TensorProto。"""
    onnx = require_onnx()
    return onnx.numpy_helper.from_array(array, name=name)


def find_initializer(model, name: str):
    for init in model.graph.initializer:
        if init.name == name:
            return init
    return None


def remove_initializer(model, name: str) -> bool:
    graph_proto = model.graph
    for index, init in enumerate(graph_proto.initializer):
        if init.name == name:
            del graph_proto.initializer[index]
            return True
    return False


def set_graph_outputs(model, output_names: List[str], value_info_map: Dict[str, Any] = None) -> None:
    """重设模型图输出 (用于 NMS 剥离等改写)。"""
    onnx = require_onnx()
    del model.graph.output[:]
    for name in output_names:
        vi = None
        if value_info_map:
            vi = value_info_map.get(name)
        if vi is None:
            vi = onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None)
        model.graph.output.append(vi)
