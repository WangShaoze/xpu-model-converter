# -*- coding: utf-8 -*-
"""PyTorch 前端适配器。"""
from xpu_converter.frontend.pytorch.base import PyTorchAdapter
from xpu_converter.frontend.pytorch.yolov8 import YOLOv8Adapter
from xpu_converter.frontend.pytorch.yolov9 import YOLOv9Adapter
from xpu_converter.frontend.pytorch.yolov10 import YOLOv10Adapter
from xpu_converter.frontend.pytorch.yolov11 import YOLOv11Adapter

__all__ = ["PyTorchAdapter", "YOLOv8Adapter", "YOLOv9Adapter", "YOLOv10Adapter", "YOLOv11Adapter"]
