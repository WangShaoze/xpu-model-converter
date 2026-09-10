# -*- coding: utf-8 -*-
"""Resize 归一化改写。

- ``Upsample`` (opset 9/10 旧算子) -> ``Resize`` (统一支持的新算子);
- ``Resize`` 若使用后端不支持的 ``coordinate_transformation_mode`` / ``mode``,
  只记录待人工确认, 不做会改变语义的静默改写。
"""
import numpy as np

from xpu_converter.ir.graph import Graph
from xpu_converter.rewrite.base import RewriteRule

# 后端明确不支持的组合
_UNSUPPORTED_MODES = ("cubic", "tf_crop_and_resize")
_UNSUPPORTED_COORD = ("tf_crop_and_resize",)


class ResizeRewrite(RewriteRule):
    name = "resize_normalize"
    op_types = ("Upsample", "Resize")
    priority = 30

    def matches(self, node) -> bool:
        return node.op_type in self.op_types

    @staticmethod
    def _attr(node, name: str, default=None):
        for attr in node.attribute:
            if attr.name == name:
                if attr.type == attr.INT:
                    return int(attr.i)
                if attr.type == attr.FLOAT:
                    return float(attr.f)
                if attr.type == attr.STRING:
                    return attr.s.decode("utf-8", errors="replace")
                if attr.type == attr.FLOATS:
                    return [float(v) for v in attr.floats]
                if attr.type == attr.INTS:
                    return [int(v) for v in attr.ints]
        return default

    def rewrite(self, graph: Graph, node) -> bool:
        if node.op_type == "Upsample":
            return self._upsample_to_resize(graph, node)
        return self._check_resize(node)

    def _upsample_to_resize(self, graph: Graph, node) -> bool:
        model = graph.raw
        onnx = self.onnx()
        if len(node.input) != 1 or len(node.output) != 1:
            return False
        scales = self._attr(node, "scales")
        if not scales:
            self.notes.append("Upsample 的 scales 非静态属性, 跳过: {}".format(node.name))
            return False
        mode = self._attr(node, "mode", "nearest")
        if mode not in ("nearest", "linear"):
            self.notes.append("Upsample mode={} 暂不支持自动改写: {}".format(mode, node.name))
            return False

        base = node.name or "upsample"
        roi_name = self.unique_name(model, "{}/roi".format(base))
        scales_name = self.unique_name(model, "{}/scales".format(base))
        model.graph.initializer.append(onnx.numpy_helper.from_array(np.array([], dtype=np.float32), name=roi_name))
        model.graph.initializer.append(
            onnx.numpy_helper.from_array(np.array(scales, dtype=np.float32), name=scales_name)
        )
        resize = self.make_node(
            "Resize",
            [node.input[0], roi_name, scales_name],
            [node.output[0]],
            name=self.unique_name(model, "{}/Resize".format(base)),
            mode=mode,
            coordinate_transformation_mode="asymmetric",
            nearest_mode="floor",
        )
        self.replace_node(model, node, [resize])
        self.notes.append("Upsample -> Resize: {}".format(node.output[0]))
        return True

    def _check_resize(self, node) -> bool:
        mode = self._attr(node, "mode", "nearest")
        coord = self._attr(node, "coordinate_transformation_mode", "half_pixel")
        if mode in _UNSUPPORTED_MODES or coord in _UNSUPPORTED_COORD:
            self.notes.append(
                "Resize 使用后端不支持的模式 (mode={}, coord={}), 需人工确认: {}".format(mode, coord, node.name)
            )
        return False
