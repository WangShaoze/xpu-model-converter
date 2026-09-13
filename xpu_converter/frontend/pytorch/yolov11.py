# -*- coding: utf-8 -*-
"""YOLO11 适配器 (PyTorch)。"""
from xpu_converter.frontend.pytorch.base import PyTorchAdapter


class YOLOv11Adapter(PyTorchAdapter):
    """YOLO11: 与 YOLOv8 类似的 raw detection 输出。"""

    model_type = "yolov11"
    ultralytics_name = "yolo11"
    end2end_default = False

    def default_input_shape(self):
        return [1, 3, 640, 640]


# ---- YOLO11 多任务变体 ----
class YOLOv11SegAdapter(YOLOv11Adapter):
    model_type = "yolov11-seg"
    task = "segment"


class YOLOv11PoseAdapter(YOLOv11Adapter):
    model_type = "yolov11-pose"
    task = "pose"


class YOLOv11ObbAdapter(YOLOv11Adapter):
    model_type = "yolov11-obb"
    task = "obb"


class YOLOv11ClsAdapter(YOLOv11Adapter):
    model_type = "yolov11-cls"
    task = "cls"
    default_topk = 5
