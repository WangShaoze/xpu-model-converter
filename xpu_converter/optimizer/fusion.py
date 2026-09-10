# -*- coding: utf-8 -*-
"""算子融合 Pass。

当前实现 ``Conv + BatchNormalization -> Conv``: 把 BN 的 (scale/bias/mean/var)
折算进卷积权重与偏置, 消除推理阶段多余的逐元素算子。
"""
from typing import Optional

import numpy as np

from xpu_converter.ir import onnx as onnx_ir
from xpu_converter.ir.graph import Graph
from xpu_converter.optimizer.base import GraphPass, PassResult


class ConvBNFusionPass(GraphPass):
    name = "conv_bn_fusion"

    def applicable(self, graph: Graph) -> bool:
        return graph.has_raw and onnx_ir.available()

    @staticmethod
    def _attr(onnx, node, name: str, default=None):
        for attr in node.attribute:
            if attr.name == name:
                return onnx.helper.get_attribute_value(attr)
        return default

    def run(self, graph: Graph) -> PassResult:
        if not self.applicable(graph):
            return PassResult(self.name, skipped="需要 onnx 依赖与原始 ModelProto")

        onnx = onnx_ir.require_onnx()
        model = graph.raw
        init_map = {init.name: init for init in model.graph.initializer}
        consumers = {}
        for node in model.graph.node:
            for name in node.input:
                if name:
                    consumers.setdefault(name, []).append(node)
        graph_outputs = {out.name for out in model.graph.output}

        fused = 0
        for conv in [n for n in model.graph.node if n.op_type == "Conv"]:
            if len(conv.input) < 2 or len(conv.output) != 1:
                continue
            weight_name = conv.input[1]
            if weight_name not in init_map:
                continue
            if len(consumers.get(weight_name, [])) != 1:
                continue  # 权重被多个节点共享, 不融合
            conv_out = conv.output[0]
            conv_consumers = consumers.get(conv_out, [])
            if len(conv_consumers) != 1 or conv_consumers[0].op_type != "BatchNormalization":
                continue
            bn = conv_consumers[0]
            if len(bn.input) != 5 or len(bn.output) != 1 or not bn.input[0] == conv_out:
                continue
            if bn.output[0] in graph_outputs:
                continue  # BN 输出是对外输出, 保持原图
            scale_name, bias_name, mean_name, var_name = bn.input[1], bn.input[2], bn.input[3], bn.input[4]
            if not all(name in init_map for name in (scale_name, bias_name, mean_name, var_name)):
                continue

            # 解析卷积偏置 (可能存在 / 为空 / 缺失)
            conv_bias_name: Optional[str] = None
            if len(conv.input) > 2:
                if conv.input[2] == "":
                    conv_bias_name = "{}_bias".format(conv.name)
                    conv.input[2] = conv_bias_name
                elif conv.input[2] in init_map and len(consumers.get(conv.input[2], [])) == 1:
                    conv_bias_name = conv.input[2]
                else:
                    continue  # 偏置非独占常量, 保守跳过
            else:
                conv_bias_name = "{}_bias".format(conv.name)
                conv.input.append(conv_bias_name)

            weight = onnx.numpy_helper.to_array(init_map[weight_name]).astype(np.float32)
            if len(conv.input) > 2 and conv.input[2] in init_map:
                conv_bias = onnx.numpy_helper.to_array(init_map[conv.input[2]]).astype(np.float32)
            else:
                conv_bias = np.zeros((weight.shape[0],), dtype=np.float32)

            gamma = onnx.numpy_helper.to_array(init_map[scale_name]).astype(np.float32)
            beta = onnx.numpy_helper.to_array(init_map[bias_name]).astype(np.float32)
            mean = onnx.numpy_helper.to_array(init_map[mean_name]).astype(np.float32)
            var = onnx.numpy_helper.to_array(init_map[var_name]).astype(np.float32)
            eps = float(self._attr(onnx, bn, "epsilon", 1e-5))
            spatial = int(self._attr(onnx, bn, "spatial", 1))

            factor = gamma / np.sqrt(var + eps)
            if spatial == 0:
                reshape = [1, -1] + [1] * (weight.ndim - 2)
            else:
                reshape = [-1] + [1] * (weight.ndim - 1)
            new_weight = (weight * factor.reshape(reshape)).astype(np.float32)
            new_bias = ((conv_bias - mean) * factor + beta).astype(np.float32)

            onnx_ir.remove_initializer(model, weight_name)
            model.graph.initializer.append(onnx.numpy_helper.from_array(new_weight, name=weight_name))
            onnx_ir.remove_initializer(model, conv_bias_name)
            model.graph.initializer.append(onnx.numpy_helper.from_array(new_bias, name=conv_bias_name))

            # 把 BN 输出的消费者接到 Conv 输出上, 删除 BN
            for node in model.graph.node:
                for index, name in enumerate(node.input):
                    if name == bn.output[0]:
                        node.input[index] = conv_out
            model.graph.node.remove(bn)
            fused += 1

        if fused:
            graph.refresh()
        return PassResult(self.name, changed=fused > 0, details={"fused": fused})
