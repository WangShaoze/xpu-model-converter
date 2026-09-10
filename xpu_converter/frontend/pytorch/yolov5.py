# -*- coding: utf-8 -*-
"""YOLOv5 适配器 (PyTorch) —— 对应 ultralytics 的 YOLOv5u (anchor-free) 权重。"""
from xpu_converter.frontend.pytorch.base import PyTorchAdapter


class YOLOv5Adapter(PyTorchAdapter):
    """YOLOv5u。

    权重文件形如 ``yolov5nu.pt``; ultralytics 导出的 ONNX 输出为
    ``[1, 4+num_classes, 8400]`` 的 raw 检测结果(不含 NMS),
    后处理与 NMS 由 Runtime 在 CPU 上完成(建设目标 §17)。
    """

    model_type = "yolov5"
    ultralytics_name = "yolov5u"
    end2end_default = False

    def default_input_shape(self):
        return [1, 3, 640, 640]
