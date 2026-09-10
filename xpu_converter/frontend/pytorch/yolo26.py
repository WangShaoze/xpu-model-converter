# -*- coding: utf-8 -*-
"""YOLO26 适配器 (PyTorch)。"""
from xpu_converter.frontend.pytorch.base import PyTorchAdapter


class YOLO26Adapter(PyTorchAdapter):
    """YOLO26。

    权重文件形如 ``yolo26n.pt``; ultralytics 导出为 ``[1, 4+num_classes, 8400]``
    的 raw 检测结果(不含 NMS), 后处理与 NMS 由 Runtime 在 CPU 上完成(建设目标 §17)。
    """

    model_type = "yolov26"
    ultralytics_name = "yolo26"
    end2end_default = False

    def default_input_shape(self):
        return [1, 3, 640, 640]
