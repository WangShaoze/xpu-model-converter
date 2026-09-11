# -*- coding: utf-8 -*-
"""YOLOv9 适配器 (PyTorch, 原生 pt→onnx→paddle)。

王建尧(WongKinYiu)团队的 YOLOv9。其 checkpoint 反序列化依赖作者仓库的
``models.*`` 包, 且 ``models/common.py -> utils/general.py`` 在 import 期会触发
``import torchvision``; 本环境 torch 与 torchvision ABI 不匹配, 需用 torchvision
占位模块隔离(见 :func:`xpu_converter.frontend.pytorch.base.install_torchvision_stub`)。

**不能**走 ultralytics 导出: thuyngch 的 forward 会把后处理/指标分支一并追踪进
ONNX, 产生大量动态 shape, 卡死 onnx2paddle。这里走原生导出:
把 Detect 头的 ``export`` 置 True(Forward 只返回合并后的静态 ``[1,4+nc,N]`` 主张量),
NMS 在 Runtime 侧完成。
"""
import os

from xpu_converter.frontend.pytorch.base import PyTorchAdapter


class YOLOv9Adapter(PyTorchAdapter):
    """YOLOv9: 列式 raw detection 输出, NMS 在 CPU 后处理完成。"""

    model_type = "yolov9"
    end2end_default = False
    # thuyngch 系: 需要作者仓库源码 + torchvision 占位才能反序列化 checkpoint
    requires_torchvision_stub = True
    model_source_root = os.environ.get("XPU_THUYNGCH_YOLOV9_REPO") or "/home/compose/develop/yolov9"
    # forward 返回 [主检测头, 指标旁支]; 只导出主检测头(静态), 去掉动态分支
    raw_output_index = 0

    def default_input_shape(self):
        return [1, 3, 640, 640]

    def _prepare_for_export(self, model):
        # 置 Detect 头 export=True: 让 forward 直接返回合并后的静态 [1,4+nc,N] 主张量
        self.set_export_flag(model, True)