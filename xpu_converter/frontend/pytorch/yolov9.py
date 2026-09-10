# -*- coding: utf-8 -*-
"""YOLOv9 适配器 (PyTorch)。"""
from xpu_converter.frontend.pytorch.base import PyTorchAdapter


class YOLOv9Adapter(PyTorchAdapter):
    """YOLOv9: 列式 raw detection 输出, NMS 在 CPU 后处理完成。"""

    model_type = "yolov9"
    ultralytics_name = "yolov9"
    end2end_default = False

    def default_input_shape(self):
        return [1, 3, 640, 640]
