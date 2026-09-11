# -*- coding: utf-8 -*-
"""YOLOv6 适配器 (PyTorch, 原生 pt→onnx→paddle)。

美团视觉智能部发布的 YOLOv6(仓库根目录 ``yolov6/``, v3 分支)。其 checkpoint ``.pt``
内含 ``model``/``ema`` 等完整的 ``nn.Module``, 反序列化依赖作者仓库的 ``yolov6.*``
包(class 如 ``SimConv``/``Detect`` 必须可解析), 因此需要把源码仓库前置到 sys.path。

**不能**走 ultralytics 导出: 官方权重来自独立仓库, 结构由 ``Model/Detect`` 定义,
ultralytics 无法重建。这里走原生导出:

* 原生 ``Model.forward`` 用 ``torch.onnx.is_in_onnx_export()`` 区分两个阶段, 导出时
  返回 ``x``(detect 输出), 推理探针时返回 ``[x, featmaps]``, 两阶段结构不一致会污染
  ONNX 输出。这里用实例级 ``forward`` 覆盖, 两阶段都只返回 ``detect(x)``。
* detect 头 ``export=False`` 则走其 dist2bbox 解码路径, 产出 **单个** ``[1, N, 5+nc]``
  (cxcywh + 恒为 1 的 objectness + 类分数)主张量, 供 Runtime 的 bnc 解码器直接消费。
  NMS 在 Runtime 侧 CPU 完成。
"""
import os
import types

from xpu_converter.frontend.pytorch.base import PyTorchAdapter


class YOLOv6Adapter(PyTorchAdapter):
    """YOLOv6: anchor-free raw detection 输出(单主张量), NMS 在 CPU 后处理完成。"""

    model_type = "yolov6"
    end2end_default = False
    # 需要作者仓库源码(yolov6.* 包)才能反序列化 checkpoint; 完整 import 无需 torchvision
    requires_torchvision_stub = False
    model_source_root = os.environ.get("XPU_MEITUAN_YOLOV6_REPO") or "/home/compose/develop/yolov6"

    def _prepare_for_export(self, model):
        # 统一 forward: 两阶段都只返回 detect 头的解码主张量, 避免探测/导出行为不一致
        def _forward(module, x):
            x = module.backbone(x)
            x = module.neck(x)
            return module.detect(x)

        model.forward = types.MethodType(_forward, model)
        # detect.export=False 才走 dist2bbox 解码路径, 返回单个 [B, N, 5+nc]
        detect = getattr(model, "detect", None)
        if detect is not None:
            detect.export = False

    def default_input_shape(self):
        return [1, 3, 640, 640]