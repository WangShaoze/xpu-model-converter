# -*- coding: utf-8 -*-
"""YOLOv7 适配器 (PyTorch, 原生 pt→onnx→paddle)。

王建尧(WongKinYiu)团队的 YOLOv7。其 checkpoint 反序列化依赖作者仓库的
``models.*`` 包, 且 ``models/common.py -> utils.xxx`` 在 import 期会触发
``import torchvision``; 本环境 torch 与 torchvision ABI 不匹配, 需用 torchvision
占位模块隔离(见 :func:`xpu_converter.frontend.pytorch.base.install_torchvision_stub`)。

**不能**走 ultralytics 导出: thuyngch 的 YOLOv7 用 IDetect 头 + 隐式层(Implicit),
ultralytics 无法重建其结构。这里走原生导出: 置 Detect 头 ``export`` 标志, 由
forward 产出静态主张量(NMS 在 Runtime 侧完成)。
"""
import os

from xpu_converter.frontend.pytorch.base import PyTorchAdapter
from xpu_converter.frontend.pytorch.vendors.yolov7_seg_shim import patch_yolov7_seg


class YOLOv7Adapter(PyTorchAdapter):
    """YOLOv7: anchor-based raw detection 输出, NMS 在 CPU 后处理完成。"""

    model_type = "yolov7"
    end2end_default = False
    # thuyngch 系: 需要作者仓库源码 + torchvision 占位才能反序列化 checkpoint
    requires_torchvision_stub = True
    model_source_root = os.environ.get("XPU_THUYNGCH_YOLOV7_REPO") or "/home/compose/develop/yolov7"

    def _prepare_for_export(self, model):
        # 产出单一 raw detection 输出 [1, 8400, 85](cxcywh, 已解码, 三尺度拼接):
        # 置 Detect 头 export=False + concat=True 走推理分支并触发 onnx_export 解码,
        # forward 返回 torch.cat(z, 1) 单个张量, 令 Runtime 的 bnc 解码器可直接消费。
        detect = None
        for _m in model.modules():
            if hasattr(_m, "concat"):
                detect = _m
                break
        if detect is not None:
            detect.export = False
            detect.concat = True

    def default_input_shape(self):
        return [1, 3, 640, 640]


class YOLOv7PoseAdapter(YOLOv7Adapter):
    """YOLOv7 pose(关键点), 基于 IKeypoint 头。

    IKeypoint 推理分支返回 ``(torch.cat(z, 1), x)``: 前者是拼接后的单张量
    ``[1, N, 4+(1+nc)+3K]``(bnc, 带 objectness), 后者是多尺度 strided 旁支。
    走原生导出且只保留主张量, NMS/关键点后处理在 Runtime 侧完成。
    """

    model_type = "yolov7-pose"
    task = "pose"
    # forward 返回 (torch.cat(z,1), x); 只导出第 0 个主张量, 去掉 strided 旁支
    raw_output_index = 0

    def default_input_shape(self):
        return [1, 3, 1280, 1280]  # w6 系 pose 大尺寸输入


class YOLOv7SegAdapter(YOLOv7Adapter):
    """YOLOv7 实例分割, 基于 ISegment 头(mask 分支结构)。

    作者 ``mask`` 分支的 ``SegmentationModel``/``ISegment``/``Proto``/``ImplicitA/M``
    等类在离线环境无法拉取, 由 :mod:`xpu_converter.frontend.pytorch.vendors.yolov7_seg_shim`
    本地补全并注入 ``models.yolo``。

    反序列化后置 ISegment ``export=True``: forward 返回 ``(out[1,N,117], proto[1,32,160,160])``
    双输出, det(含 objectness, 前 85 列) + 掩膜系数(后 32 列) 与 ``segment`` 输出契约对齐。
    """

    model_type = "yolov7-seg"
    task = "segment"
    # 需要同时导出 det+coeff 与 proto 双输出, 关闭 raw_output_index 截断
    raw_output_index = None

    def _prepare_import_env(self):
        super()._prepare_import_env()
        # checkpoint 反序列化需要 models.yolo.SegmentationModel / ISegment / Proto 等类
        patch_yolov7_seg()

    def _prepare_for_export(self, model):
        # 置 ISegment export=True -> forward 返回 (out, proto) 双输出
        for _m in model.modules():
            if hasattr(_m, "export") and hasattr(_m, "proto"):
                _m.export = True

    def default_input_shape(self):
        return [1, 3, 640, 640]