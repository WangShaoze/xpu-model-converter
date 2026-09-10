# -*- coding: utf-8 -*-
"""NMS 改写: 把 ``NonMaxSuppression`` 从计算图中剥离, 交给 Runtime 在 CPU 上完成。

建设目标 §17 明确 V1 的策略是 **NMS = CPU**。剥离后图的输出变为
``[boxes, scores]``, Runtime 侧按顺序消费这两个张量做解码 + NMS。
"""
from typing import List

from xpu_converter.ir.graph import Graph
from xpu_converter.rewrite.base import RewriteRule


class NMSRewrite(RewriteRule):
    name = "nms_to_cpu"
    op_types = ("NonMaxSuppression",)
    priority = 40

    def rewrite(self, graph: Graph, node) -> bool:
        model = graph.raw
        onnx = self.onnx()
        nms_nodes = [n for n in model.graph.node if n.op_type == "NonMaxSuppression"]
        if len(nms_nodes) != 1:
            self.notes.append("NonMaxSuppression 节点数为 {}, 无法安全剥离, 跳过".format(len(nms_nodes)))
            return False
        nms = nms_nodes[0]
        if len(nms.input) < 2:
            self.notes.append("NonMaxSuppression 输入不足, 跳过剥离")
            return False

        boxes_name, scores_name = nms.input[0], nms.input[1]
        downstream = self._collect_downstream(model, [name for name in nms.output if name])
        if not self._is_closed(model, downstream, nms):
            self.notes.append("NMS 下游存在集合外消费者, 保守跳过 (请人工检查)")
            return False
        if not self._all_outputs_in_chain(model, downstream, nms):
            self.notes.append("模型存在 NMS 之外的输出头, 剥离会误删该输出, 保守跳过 (请人工确认)")
            return False

        new_outputs = []
        for name in (boxes_name, scores_name):
            vi = self.find_value_info(model, name)
            if vi is None:
                vi = onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None)
            new_outputs.append(vi)

        remove_keys = {self.node_key(item) for item in downstream}
        remove_keys.add(self.node_key(nms))
        kept = [item for item in model.graph.node if self.node_key(item) not in remove_keys]
        del model.graph.node[:]
        model.graph.node.extend(kept)

        del model.graph.output[:]
        for vi in new_outputs:
            model.graph.output.append(vi)

        self.notes.append(
            "NonMaxSuppression -> CPU NMS, 图输出改为 [{}, {}]".format(boxes_name, scores_name)
        )
        return True

    @staticmethod
    def _collect_downstream(model, seeds: List[str]) -> List:
        consumers = {}
        for item in model.graph.node:
            for name in item.input:
                if name:
                    consumers.setdefault(name, []).append(item)
        collected = {}
        queue = list(seeds)
        while queue:
            name = queue.pop(0)
            for item in consumers.get(name, []):
                if NMSRewrite.node_key(item) in collected:
                    continue
                collected[NMSRewrite.node_key(item)] = item
                queue.extend(out for out in item.output if out)
        return list(collected.values())

    @classmethod
    def _all_outputs_in_chain(cls, model, downstream: List, nms) -> bool:
        """所有对外输出都必须由「NMS 链路内」的节点产出, 否则剥离会误删其他输出头。"""
        producers = {}
        for item in model.graph.node:
            for name in item.output:
                if name:
                    producers[name] = item
        inside_keys = {cls.node_key(item) for item in downstream}
        inside_keys.add(cls.node_key(nms))
        for out in model.graph.output:
            producer = producers.get(out.name)
            if producer is None or cls.node_key(producer) not in inside_keys:
                return False
        return True

    @classmethod
    def _is_closed(cls, model, downstream: List, nms) -> bool:
        """下游产出的张量不得被「下游集合之外」的节点消费。"""
        inside_keys = {cls.node_key(item) for item in downstream}
        inside_keys.add(cls.node_key(nms))
        inside_outputs = set()
        for item in list(downstream) + [nms]:
            inside_outputs.update(name for name in item.output if name)
        for item in model.graph.node:
            if cls.node_key(item) in inside_keys:
                continue
            for name in item.input:
                if name and name in inside_outputs:
                    return False
        return True
