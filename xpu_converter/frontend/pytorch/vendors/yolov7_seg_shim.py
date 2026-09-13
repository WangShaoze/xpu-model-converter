# -*- coding: utf-8 -*-
"""yolov7 分割(mask 分支)缺失依赖的本地补全。

离线拿不到 WongKinYiu/yolov7 的 ``mask`` 分支源码, 为保证 ``yolov7-seg.pt`` 可反序列化
并导出, 这里按作者 mask 分支的公开实现补全缺失类, 注入已加载的 ``models.yolo``。
设计对齐已在我方验证过的 YOLOv5-seg Segment 头: ``m`` 输出 ``na*(no+nm)``, 拆分出
检测(det, 含 objectness)与掩膜系数(mask coeff); ``proto`` 输出原型图。仅用于模型加载/导出,
训练路径不在覆盖范围。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Proto(nn.Module):
    """YOLOv7-seg 掩膜原型头: 256 -> 256 -> 上采样x2 -> 256 -> 32。"""

    def __init__(self, c1, c_=256, c2=32):
        super().__init__()
        from models.common import Conv
        self.cv1 = Conv(c1, c_, k=3)                 # 256
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)                 # 256
        self.cv3 = Conv(c_, c2)                      # 32

    def forward(self, x):
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class ISegment(nn.Module):
    """YOLOv7-seg 检测+分割头(等价作者 mask 分支的 ISegment/MergedDetect)。

    结构与 YOLOv5-seg 的 Segment 头一致: 每个尺度 ``m`` 输出 ``na*(no+nm)`` 通道,
    前 ``no`` 为带 objectness 的检测, 后 ``nm`` 为掩膜系数; ``proto`` 输出原型图。
    推理(export=False)时对检测做 img 坐标解码(NMS 下沉到 Runtime), 并与掩膜系数、
    原型图一并返回给 :class:`SegmentDecoder`。
    """

    export = False
    end2end = False
    include_nms = False
    concat = False

    def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=(), inplace=True):
        super().__init__()
        self.nc = nc
        self.no = nc + 5 + nm  # 检测+掩膜系数的总通道数
        self.nm = nm
        self.np = npr
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid = [torch.zeros(1)] * self.nl
        self.stride = torch.zeros(self.nl)
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))
        # m: 每个尺度输入通道 -> na*no
        self.m = nn.ModuleList(nn.Conv2d(int(x), int(self.na * self.no), 1) for x in ch)
        self.ia = nn.ModuleList(ImplicitA(x) for x in ch)
        self.im = nn.ModuleList(ImplicitM(self.na * (self.nc + 5)) for _ in ch)
        self.proto = Proto(ch[0], self.np, self.nm)
        self.inplace = inplace

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing='ij')
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def forward(self, x):
        # x = [f_p3, f_p4, f_p5]; 由 Model.forward_once 按 f=[102,103,104] 组装
        # 注意: 该实现的 ``no = nc + 5 + nm``(已含掩膜系数), det 为前 ``nc+5`` 列。
        bs = x[0].shape[0]
        det_cols = self.nc + 5
        p = self.proto(x[0])
        z = []  # 拼接各尺度的 (det+coeff)
        for i in range(self.nl):
            xi = self.ia[i](x[i])                       # (bs, c, ny, nx)
            xi = self.m[i](xi)                          # (bs, na*no, ny, nx)
            ny, nx = xi.shape[2:]
            xi = xi.view(bs, self.na, self.no, ny, nx)
            xi = xi.permute(0, 1, 3, 4, 2).contiguous()  # (bs, na, ny, nx, no)
            xdet = xi[..., :det_cols]                   # 检测(含 objectness)
            xm = xi[..., det_cols:]                     # 掩膜系数 (bs, na, ny, nx, nm)
            y = xdet * self.im[i].implicit.view(1, self.na, 1, 1, det_cols)  # ImplicitM
            y = y.sigmoid()
            if self.grid[i].shape[2:4] != (ny, nx):
                self.grid[i] = self._make_grid(nx, ny).to(y.device)
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]
            # 用已按 stride 归一化的 anchors 做 wh 解码(anchor_grid 为训练期按固定尺寸
            # 预生成的全网格, export 时尺寸不固定不能用)。
            ag = self.anchors[i].view(self.na, 1, 1, 2)
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * ag
            z.append(torch.cat([y, xm], -1).view(bs, self.na * ny * nx, self.no))
        out = torch.cat(z, 1)                            # (bs, N, no) bnc
        return (out, p) if self.export else (out, (None, z, p))


class ImplicitA(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))

    def forward(self, x):
        return self.implicit + x


class ImplicitM(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.implicit = nn.Parameter(torch.ones(1, channel, 1, 1))

    def forward(self, x):
        return self.implicit * x


def patch_yolov7_seg() -> None:
    """在主仓库 ``models.yolo`` 已装入 sys.path 的前提下注入缺失类(幂等)。"""
    import models.common
    import models.yolo
    from models.yolo import Model

    class SegmentationModel(Model):
        pass

    models.yolo.SegmentationModel = SegmentationModel
    models.yolo.ImplicitA = ImplicitA
    models.yolo.ImplicitM = ImplicitM
    models.yolo.ISegment = ISegment
    models.common.Proto = Proto


def dump_head(model, head_name='ISegment'):
    out = []
    for sn, sm in model.named_modules():
        if type(sm).__name__ == head_name:
            out.append('%s @%s attrs=%s' % (head_name, sn, sorted(
                k for k in sm.__dict__ if not k.startswith('_'))))
            for sn2, sm2 in sm.named_children():
                out.append('   %s %s' % (sn2, type(sm2).__name__))
            return '\n'.join(out)
    return 'no head %s' % head_name