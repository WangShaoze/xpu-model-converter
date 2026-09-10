# -*- coding: utf-8 -*-
"""YOLOv10 适配器 (PyTorch) —— V1 首个落地模型。"""
from xpu_converter.frontend.pytorch.base import PyTorchAdapter


class YOLOv10Adapter(PyTorchAdapter):
    """YOLOv10。

    YOLOv10 为 NMS-free 端到端结构, 但 ultralytics 导出的 ONNX 默认仍会带
    ``NonMaxSuppression`` 节点; 该节点由 :mod:`xpu_converter.rewrite.nms` 剥离,
    改由 Runtime 在 CPU 上完成 NMS (建设目标 §17: NMS = CPU)。
    """

    model_type = "yolov10"
    ultralytics_name = "yolov10"
    end2end_default = True

    def default_input_shape(self):
        return [1, 3, 640, 640]
