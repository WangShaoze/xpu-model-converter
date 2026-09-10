# -*- coding: utf-8 -*-
"""图简化 Pass。

1. 若安装了 ``onnxsim`` 则优先使用 (最强简化);
2. 内置兜底简化: Identity 消除 / 死节点消除 / 无用 initializer 清理。
"""
from xpu_converter.ir import onnx as onnx_ir
from xpu_converter.ir.graph import Graph
from xpu_converter.logging_utils import get_logger
from xpu_converter.optimizer.base import GraphPass, PassResult

logger = get_logger(__name__)

# 这些算子有副作用或随机性, 不做死代码消除
_SIDE_EFFECT_PREFIXES = ("Random", "Multinomial")


def onnxsim_available() -> bool:
    try:
        import onnxsim  # noqa: F401
        return True
    except Exception:
        return False


class GraphSimplifyPass(GraphPass):
    name = "graph_simplify"

    def applicable(self, graph: Graph) -> bool:
        return graph.has_raw and onnx_ir.available()

    # ------------------------------------------------------------- 工具
    @staticmethod
    def _replace_consumers(model, old: str, new: str) -> None:
        """把消费者引用的 old 换成 new, 并同步图输出与 value_info。"""
        for node in model.graph.node:
            for index, name in enumerate(node.input):
                if name == old:
                    node.input[index] = new
        for out in model.graph.output:
            if out.name == old:
                out.name = new
        for vi in model.graph.value_info:
            if vi.name == old:
                vi.name = new

    def _remove_identity(self, model) -> int:
        removed = 0
        output_names = {out.name for out in model.graph.output}
        changed = True
        while changed:
            changed = False
            for node in list(model.graph.node):
                if node.op_type != "Identity" or len(node.input) != 1 or len(node.output) != 1:
                    continue
                src, dst = node.input[0], node.output[0]
                if not src or src == dst:
                    continue
                if dst in output_names and src in output_names:
                    continue  # 目标名已被其它输出占用
                self._replace_consumers(model, dst, src)
                if dst in output_names:
                    output_names.discard(dst)
                    output_names.add(src)
                model.graph.node.remove(node)
                removed += 1
                changed = True
        return removed

    @staticmethod
    def _remove_dead_nodes(model) -> int:
        removed = 0
        changed = True
        while changed:
            changed = False
            used = set()
            for node in model.graph.node:
                used.update(name for name in node.input if name)
            for out in model.graph.output:
                used.add(out.name)
            for node in list(model.graph.node):
                if node.op_type.startswith(_SIDE_EFFECT_PREFIXES):
                    continue
                if any(name for name in node.output):
                    if all((not name) or (name not in used) for name in node.output):
                        model.graph.node.remove(node)
                        removed += 1
                        changed = True
        return removed

    @staticmethod
    def _remove_unused_initializers(model) -> int:
        used = set()
        for node in model.graph.node:
            used.update(name for name in node.input if name)
        for out in model.graph.output:
            used.add(out.name)
        keep = [init for init in model.graph.initializer if init.name in used]
        removed = len(model.graph.initializer) - len(keep)
        if removed:
            del model.graph.initializer[:]
            model.graph.initializer.extend(keep)
        return removed

    # ------------------------------------------------------------- 主流程
    def run(self, graph: Graph) -> PassResult:
        if not self.applicable(graph):
            return PassResult(self.name, skipped="需要 onnx 依赖与原始 ModelProto")

        details = {}
        changed = False

        if onnxsim_available():
            try:
                import onnxsim

                simplified, check = onnxsim.simplify(graph.raw)
                if check:
                    graph.raw = simplified
                    changed = True
                    details["onnxsim"] = "ok"
                else:
                    details["onnxsim"] = "check_failed"
            except Exception as err:  # onnxsim 失败不影响后续内置简化
                details["onnxsim"] = "error: {}".format(err)
                logger.warning("onnxsim 简化失败, 使用内置简化: %s", err)

        identity_removed = self._remove_identity(graph.raw)
        dead_removed = self._remove_dead_nodes(graph.raw)
        unused_removed = self._remove_unused_initializers(graph.raw)
        details.update(
            {
                "identity_removed": identity_removed,
                "dead_nodes_removed": dead_removed,
                "unused_initializers_removed": unused_removed,
            }
        )
        changed = changed or bool(identity_removed or dead_removed or unused_removed)

        graph.refresh()
        return PassResult(self.name, changed=changed, details=details)
