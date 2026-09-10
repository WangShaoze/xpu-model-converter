# -*- coding: utf-8 -*-
"""YOLOv8 适配器 (PyTorch)。"""
from xpu_converter.frontend.pytorch.base import PyTorchAdapter


class YOLOv8Adapter(PyTorchAdapter):
    """YOLOv8: 单输出 raw detection, NMS 在 CPU 后处理完成。"""

    model_type = "yolov8"
    ultralytics_name = "yolov8"
    end2end_default = False

    def default_input_shape(self):
        return [1, 3, 640, 640]
