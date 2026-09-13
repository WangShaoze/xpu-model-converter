# YOLO 系列多任务适配：覆盖矩阵与实施方案

> 范围限定：仅覆盖 `models/` 下**已提供**的权重文件，不在本范围内补齐官方全量权重。
> 任务范围：检测(detect)、实例分割(segment)、姿态(pose)、旋转框(obb)、分类(cls)、
> 深度估计(depth)、语义分割(sem)。
> 文档状态：方案评审稿（尚未进入实现）。

---

## 1. 背景与目标

当前平台已打通 YOLO 各版本的**检测(detect)**任务（pt→onnx→paddle→Docker 交付包，
容器 `/predict` 真图验证通过）。但 YOLO 系列同一版本往往带有不同体量与不同任务权重，
**不同任务的输出结构、解码方式、后处理均不相同**，不能复用检测的解码链路。

本方案的目标：
1. 盘点 `models/` 下已提供权重的「版本 × 体量 × 任务」全量清单；
2. 明确各任务的输出契约差异与所需新增能力；
3. 给出前端适配器与 Runtime 解码器的扩展方案与落地顺序。

---

## 2. 任务语义与输出差异总览

| 任务 | 典型输出形状(640, nc=80, N=8400) | 输出契约 layout | 解码/后处理 | Runtime 响应 |
|---|---|---|---|---|
| detect | `[1, 4+nc, N]` 或 `[1, N, 4+nc]` | `bnc` / `bcn` | xywh 反归一化 + 类别 + CPU NMS | 框/置信度/类别 |
| segment | det 分支 `[1, 4+nc+32, N]` + proto `[1, 32, H/4, W/4]` | `segment` | 先 det NMS → 用路上 mask 系数做矩阵乘 / sigmoid → 按框裁剪 → 缩放掩膜 | 框 + polygon/mask |
| pose | `[1, 4+nc+K*3, N]`(K=17) | `pose` | 框(head/bbox) + 关键点 x/y/可见度置信 | 框 + 关键点数组 |
| obb | `[1, 4+nc, N]`，rbox 语义 | `obb` | 旋转框(cx,cy,长边,短边,角度)解码 + NMS | 框(角度) |
| cls | `[1, nc]` | `cls` | softmax/argmax | 类别概率列表 |
| depth | `[1, 1, H, W]` | `dense` | 每像素深度归一化(可选) | 深度图(base64) |
| sem | `[1, C, H, W]` | `dense` | 每像素 argmax | 类别索引图(base64) |

> N=80²+40²+20²=8400；H/4/W/4=160 为 proto/mask 分辨率。

---

## 3. 已提供权重 × 体量 × 任务 覆盖矩阵

### 3.1 覆盖矩阵（★=已提供权重；绿=当前已适配，红=本方案需新增）

| 版本 | 来源 | 体量 | det | seg | pose | obb | cls | depth | sem |
|---|---|---|---|---|---|---|---|---|---|
| **YOLOv6** | 美团原生 | n,s,m,l,n6,s6,m6,l6,lite_s/m/l | ★🟢 | — | — | — | — | — | — |
| **YOLOv7** | 王建尧原生 | yolov7,x,d6,e6,e6e,u6,w6 | ★🟢 | ★🔴 | — | — | — | — | — |
| | | seg: yolov7-seg, yolov7-mask | | ★🔴 | | | | | |
| | | pose: yolov7-w6-pose | | | ★🔴 | | | | |
| **YOLOv9** | 王建尧原生 | yolov9-{c,e,m,s,t}, converted×4, gelan-{c,e,m,s,det,pan} | ★🟢 | — | — | — | — | — | — |
| | | seg: yolov9c-seg, yolov9e-seg, gelan-c-seg | | ★🔴 | | | | | |
| **YOLOv5** | ultralytics | n,s,m,l,x (u 格式) | ★🟢 | — | — | — | — | — | — |
| **YOLOv8** | ultralytics | n,s,m,l,x | ★🟢 | — | — | — | — | — | — |
| | | seg/pose/obb/cls: n, s | | ★🔴 | ★🔴 | ★🔴 | ★🔴 | | |
| **YOLOv10** | ultralytics | n,s,m,l,x | ★🟢 | — | — | — | — | — | — |
| **YOLOv11** | ultralytics | yolo11{n,s}(det) | ★🟢 | — | — | — | — | — | — |
| | | seg/pose/obb/cls: n, s | | ★🔴 | ★🔴 | ★🔴 | ★🔴 | | |
| **YOLOv12** | ultralytics | n,s,m,l,x | ★🟢 | — | — | — | — | — | — |
| **YOLO26** | ultralytics | yolo26{n,s}(det) | ★🟢 | — | — | — | — | — | — |
| | | seg/pose/obb/cls/depth/sem: n, s | | ★🔴 | ★🔴 | ★🔴 | ★🔴 | ★🔴 | ★🔴 |

### 3.2 结论

- **检测任务已全覆盖并交付**（8 个版本）。
- **需新增的非检测权重清单**（共 16 个类型 / 约 30 个文件，均为 `models/` 下实际存在者）：
  - YOLOv7：seg×2、pose×1
  - YOLOv9：seg×3
  - YOLOv8：seg/pose/obb/cls 各×2
  - YOLOv11：seg/pose/obb/cls 各×2
  - YOLO26：seg/pose/obb/cls/depth/sem 各×2

---

## 4. 现状差距（前端 + Runtime）

### 4.1 前端适配器
所有 YOLO 适配器当前**只声明 `task=detection`**，配置硬编码 `num_classes: 80`、
`output_layout: auto`。原生系（v6/v7/v9）适配器仅完成 detect 头 forward 覆盖，未覆盖
seg/pose 头的 `export=False` 输出结构。

### 4.2 Runtime
- 目录只有 `runtime/detection/`，解码器仅 [nwai_decoders.py](xpu_converter/../runtime/detection/nwai_decoders.py)
  的 `base/bcn/bnc/bnc6/pair`（检测）。
- 输出契约探测 [ModelOutputContract](xpu_converter/backend/kunlun/compiler.py) 仅实现检测布局。
- `/predict` 返回结构、`predict_image` 绘制均只含检测框，无 mask/keypoint/rbox/概率图。

---

## 5. 实施方案

### 5.1 通用机制
1. **任务注册扩展**：`registry/model_registry.py` 的 `ModelSupport` 增加任务维度；
   配置 `configs/models/*.yaml` 增加 `task: segment|pose|obb|cls|depth|sem`。
2. **输出契约扩展**：新增 layout：`segment`、`pose`、`obb`、`cls`、`dense`；
   `ModelOutputContract.detect()` 扩展为按任务识别布局并写入 `runtime.yaml`。
3. **前端导出**：
   - ultralytics 系（v5/8/10/11/12/26）：`YOLO(path)` 从权重自动推断任务，
     `.export(format="onnx")` 即可导出对应输出；adapter 按任务切换输出包装与输出命名。
   - 原生系（v7/v9）seg/pose：扩展 `_prepare_for_export`，按任务头重写 forward 输出，
     对齐 ultralytics 的导出布局（det+proto 双输出、det+kpt 单输出）。

### 5.2 Runtime 解码器（`runtime/detection/nwai_decoders.py` 扩展）
| layout | 解码器要点 |
|---|---|
| `segment` | det 分支先过 NMS → 取前后 N 个 mask 系数与 proto 做矩阵乘 → sigmoid → 按框裁剪 → 放大到原图 → 输出 polygon（或 mask 编码） |
| `pose` | 框解码 + `x,y,conf` 关键点反归一化（乘以 stride，映射回输入坐标） |
| `obb` | rbox(cx,cy,长边,短边,角度) 转五参数/角点，旋转框 NMS |
| `cls` | softmax → topk 类别与置信度列表（无框） |
| `dense` | depth：深度图归一化为 uint8 图；sem：逐像素 argmax 得类别索引图 |

### 5.3 Web 服务与响应协议
- `/predict` 返回扩展 `data` 结构：segment 加 `Polygon/Mask`、pose 加 `KeyPoints`、
  obb 加角度、cls 加 `Scores`、dense 返回编码图。
- `/predict_image` 支持绘制 mask/关键点/旋转框。

### 5.4 落地顺序（建议）
1. **ultralytics 系 seg/pose/obb/cls**（v8/v11/v26），改动集中、可复用 `YOLO.task=*` 导出；
2. **原生系 seg/pose**（yolov7、yolov9），需逐个重写检测头 forward；
3. **YOLO26 depth/sem**（dense 输出）；
4. 全量权重转换 → 打包 → 容器 `/predict` 真图回归。

---

## 6. 验收标准
- `models/` 下每个已提供权重均能完成 pt→onnx→paddle→Docker 交付包；
- 各任务容器 `/predict` 对测试图返回正确的检测框/掩膜/关键点/旋转框/类别/深度图/语义图；
- 全量回归（`python -m unittest discover -s tests`）通过，检测任务不受影响。