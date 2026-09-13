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


# ---- YOLO26 多任务变体 ----
class YOLO26SegAdapter(YOLO26Adapter):
    model_type = "yolov26-seg"
    task = "segment"


class YOLO26PoseAdapter(YOLO26Adapter):
    model_type = "yolov26-pose"
    task = "pose"


class YOLO26ObbAdapter(YOLO26Adapter):
    model_type = "yolov26-obb"
    task = "obb"


class YOLO26ClsAdapter(YOLO26Adapter):
    model_type = "yolov26-cls"
    task = "cls"
    default_topk = 5


class YOLO26DepthAdapter(YOLO26Adapter):
    model_type = "yolov26-depth"
    task = "depth"


class YOLO26SemAdapter(YOLO26Adapter):
    model_type = "yolov26-sem"
    task = "sem"
