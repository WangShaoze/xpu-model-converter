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


# ---- YOLOv8 多任务变体: 沿用 ultralytics ``YOLO(weight)`` 按权重自动推断任务导出 ----
class YOLOv8SegAdapter(YOLOv8Adapter):
    model_type = "yolov8-seg"
    task = "segment"


class YOLOv8PoseAdapter(YOLOv8Adapter):
    model_type = "yolov8-pose"
    task = "pose"


class YOLOv8ObbAdapter(YOLOv8Adapter):
    model_type = "yolov8-obb"
    task = "obb"


class YOLOv8ClsAdapter(YOLOv8Adapter):
    model_type = "yolov8-cls"
    task = "cls"
    default_topk = 5
