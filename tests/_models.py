# -*- coding: utf-8 -*-
"""构造合成 ONNX 模型, 供单元测试使用 (不依赖 torch / ultralytics)。"""
from typing import Dict, List

import numpy as np

from xpu_converter.ir import onnx as onnx_ir


def _np(onnx, array, name):
    return onnx.numpy_helper.from_array(np.asarray(array), name=name)


def _finalize(onnx, graph):
    """构造模型并固定 IR 版本 (兼容 onnxruntime 支持范围)。"""
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)])
    model.ir_version = min(int(model.ir_version), 10)
    return model


def build_conv_bn_relu(batch: int = 1, channels: int = 4, height: int = 8, width: int = 8):
    """input -> Conv -> BatchNormalization -> Relu -> output (用于融合测试)。"""
    onnx = onnx_ir.require_onnx()
    rng = np.random.RandomState(0)
    weight = rng.randn(channels, 3, 3, 3).astype(np.float32)
    bias = rng.randn(channels).astype(np.float32)
    gamma = (rng.rand(channels).astype(np.float32) + 0.5)
    beta = rng.randn(channels).astype(np.float32)
    mean = rng.randn(channels).astype(np.float32)
    var = rng.rand(channels).astype(np.float32) + 0.5

    nodes = [
        onnx.helper.make_node("Conv", ["images", "conv_w", "conv_b"], ["conv_out"], name="conv",
                              pads=[1, 1, 1, 1]),
        onnx.helper.make_node(
            "BatchNormalization", ["conv_out", "bn_s", "bn_b", "bn_m", "bn_v"], ["bn_out"],
            name="bn", epsilon=1e-5,
        ),
        onnx.helper.make_node("Relu", ["bn_out"], ["output"], name="act"),
    ]
    graph = onnx.helper.make_graph(
        nodes,
        "conv_bn_relu",
        [onnx.helper.make_tensor_value_info("images", onnx.TensorProto.FLOAT, [batch, 3, height, width])],
        [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [batch, channels, height, width])],
        initializer=[
            _np(onnx, weight, "conv_w"),
            _np(onnx, bias, "conv_b"),
            _np(onnx, gamma, "bn_s"),
            _np(onnx, beta, "bn_b"),
            _np(onnx, mean, "bn_m"),
            _np(onnx, var, "bn_v"),
        ],
    )
    return _finalize(onnx, graph)


def build_silu_graph(batch: int = 1, channels: int = 3, height: int = 4, width: int = 4):
    """input -> Silu -> output (用于 Silu 改写测试)。"""
    onnx = onnx_ir.require_onnx()
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Silu", ["images"], ["output"], name="silu")],
        "silu_graph",
        [onnx.helper.make_tensor_value_info("images", onnx.TensorProto.FLOAT, [batch, channels, height, width])],
        [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [batch, channels, height, width])],
    )
    return _finalize(onnx, graph)


def build_shape_fold(batch: int = 1, channels: int = 3, height: int = 8, width: int = 16):
    """Shape -> Gather -> Cast 常量折叠链。"""
    onnx = onnx_ir.require_onnx()
    nodes = [
        onnx.helper.make_node("Shape", ["images"], ["shape_out"], name="shape"),
        onnx.helper.make_node("Gather", ["shape_out", "hw_idx"], ["hw"], name="gather", axis=0),
        onnx.helper.make_node("Cast", ["hw"], ["spatial"], name="cast", to=onnx.TensorProto.FLOAT),
    ]
    graph = onnx.helper.make_graph(
        nodes,
        "shape_fold",
        [onnx.helper.make_tensor_value_info("images", onnx.TensorProto.FLOAT, [batch, channels, height, width])],
        [onnx.helper.make_tensor_value_info("spatial", onnx.TensorProto.FLOAT, [2])],
        initializer=[_np(onnx, np.array([2, 3], dtype=np.int64), "hw_idx")],
    )
    return _finalize(onnx, graph)


def build_nms_graph(extra_head: bool = False):
    """boxes/scores -> NMS -> Gather -> Add -> output (用于 NMS 剥离测试)。

    模拟真实端到端导出图: NMS 的下游节点消费 ``selected`` 索引, 最终产出解码后的
    检测结果; 剥离 NMS 后图输出应退化为 ``[boxes, scores]``。

    ``extra_head=True`` 时额外挂一个与 NMS 无关的输出头, 用于验证多输出模型会被
    保守跳过 (否则会误删该输出)。
    """
    onnx = onnx_ir.require_onnx()
    nodes = [
        onnx.helper.make_node(
            "NonMaxSuppression", ["boxes", "scores", "max_out", "iou_thres", "score_thres"],
            ["selected"], name="nms",
        ),
        onnx.helper.make_node("Gather", ["boxes", "selected"], ["sel_boxes"], name="gather_boxes", axis=1),
        onnx.helper.make_node("Add", ["sel_boxes", "one"], ["detections"], name="add_one"),
    ]
    outputs = [onnx.helper.make_tensor_value_info("detections", onnx.TensorProto.FLOAT, None)]
    if extra_head:
        nodes.append(onnx.helper.make_node("Mul", ["boxes", "one"], ["extra_head"], name="extra"))
        outputs.append(
            onnx.helper.make_tensor_value_info("extra_head", onnx.TensorProto.FLOAT, [1, 100, 4])
        )
    graph = onnx.helper.make_graph(
        nodes,
        "nms_graph",
        [
            onnx.helper.make_tensor_value_info("boxes", onnx.TensorProto.FLOAT, [1, 100, 4]),
            onnx.helper.make_tensor_value_info("scores", onnx.TensorProto.FLOAT, [1, 80, 100]),
        ],
        outputs,
        initializer=[
            _np(onnx, np.array(300, dtype=np.int64), "max_out"),
            _np(onnx, np.array(0.45, dtype=np.float32), "iou_thres"),
            _np(onnx, np.array(0.25, dtype=np.float32), "score_thres"),
            _np(onnx, np.array(1.0, dtype=np.float32), "one"),
        ],
    )
    return _finalize(onnx, graph)


def run_onnx(model, feeds: Dict[str, np.ndarray]) -> List[np.ndarray]:
    """用 onnxruntime 执行模型 (测试用)。"""
    import onnxruntime as ort

    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    names = [out.name for out in session.get_outputs()]
    return session.run(names, feeds)
