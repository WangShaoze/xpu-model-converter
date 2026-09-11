# -*- coding: utf-8 -*-
"""PyTorch 前端适配器。"""
from xpu_converter.frontend.pytorch.base import PyTorchAdapter
from xpu_converter.frontend.pytorch.yolov5 import YOLOv5Adapter
from xpu_converter.frontend.pytorch.yolov7 import YOLOv7Adapter
from xpu_converter.frontend.pytorch.yolov8 import YOLOv8Adapter
from xpu_converter.frontend.pytorch.yolov9 import YOLOv9Adapter
from xpu_converter.frontend.pytorch.yolov10 import YOLOv10Adapter
from xpu_converter.frontend.pytorch.yolov11 import YOLOv11Adapter
from xpu_converter.frontend.pytorch.yolo12 import YOLO12Adapter
from xpu_converter.frontend.pytorch.yolo26 import YOLO26Adapter

__all__ = [
    "PyTorchAdapter",
    "YOLOv5Adapter",
    "YOLOv7Adapter",
    "YOLOv8Adapter",
    "YOLOv9Adapter",
    "YOLOv10Adapter",
    "YOLOv11Adapter",
    "YOLO12Adapter",
    "YOLO26Adapter",
]
