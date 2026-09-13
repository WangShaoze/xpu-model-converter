可以。我这次不再以“Adapter 有没有注册”为判断标准，而是按你要求的**真正交付链路**逐层审计：

> `模型 → Adapter → YAML → 输出契约 → Rewrite → Backend → Runtime → Tests → 当前缺陷 → 修改位置`

我核对的是当前仓库 `main`，而且把 README 的“声明能力”和实际代码行为分开看。仓库当前明确把 **YOLOv10 Detection** 定义为唯一 V1 stable，其余 YOLO 都是 experimental；README 自身也明确 V1 范围是 YOLOv10 Detection。([GitHub][1])

---

# 一、先给最终审计结论

## 当前真正完成的，不是“9 个 YOLO 都完成”

我给当前仓库分成 4 个等级：

| 等级                   | 定义                                                                                                    | 当前模型                                                                     |
| -------------------- | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| **A：真实完成**           | Frontend → ONNX → Rewrite → Kunlun/Paddle-XPU → Runtime → Validation → Package 有完整闭环设计，并被项目定义为 stable | **YOLOv10 Detection**                                                    |
| **B：前端基本完成**         | 模型加载、ONNX 导出、部分输出处理已有真实代码，但缺少完整 XPU/Runtime/真机验证                                                      | YOLOv5/v6/v7/v8/v9/v11/v12/YOLO26 Detection                              |
| **C：任务 Adapter 已声明** | Adapter/YAML 有任务定义，但 Runtime/输出解码/交付链并没有对应完整实现                                                        | YOLOv7 Pose/Seg、YOLOv8 Seg/Pose/OBB/Cls、YOLOv9 Seg、YOLO11 多任务、YOLO26 多任务 |
| **D：目前属于“名义支持”**     | 注册了 task，但没有相应任务级 Runtime/Backend/Validation 闭环                                                       | **YOLO26 Depth/Sem 尤其明显**                                                |

最重要的证据是：

```python
V1_STABLE_MODELS = ("yolov10",)
```

而 `ModelSupport.is_deliverable` 只有 `stable` 才返回 `True`。也就是说，**项目自身已经明确告诉我们：Adapter 存在 ≠ 可交付支持。** 

---

# 二、全局架构审计

当前项目的架构实际上是对的：

```text
best.pt
   │
   ▼
Model Registry
   │
   ▼
Frontend Adapter
   │
   ▼
ONNX
   │
   ├── Graph Check
   ├── Operator Analysis
   ├── Optimization
   └── Rewrite
   │
   ▼
Kunlun Backend
   │
   ├── Paddle XPU
   ├── XPU Toolkit
   └── Stub
   │
   ▼
Runtime
   │
   ▼
Validation
   │
   ▼
Docker Export
```

Pipeline 代码实际执行了这条链：模型识别、加载、ONNX、Graph Check、Operator Analysis、Optimization/Rewrite，然后才进入 Compile。

**问题不是架构错，而是“模型能力矩阵远远领先于后端和 Runtime 能力”。**

---

# 三、模型逐个审计

## 1. YOLOv5

### Adapter

文件：

```text
xpu_converter/frontend/pytorch/yolov5.py
```

实际上支持的是：

```text
YOLOv5u
```

而不是完整意义上的传统 Ultralytics YOLOv5。

代码：

```python
ultralytics_name = "yolov5u"
```

并明确写了 `yolov5nu.pt`。

### YAML

```text
configs/models/yolov5.yaml
```

存在，定义：

```text
task = detection
input = 1x3x640x640
opset = 13
NMS = CPU
```



### 输出契约

Adapter 假定：

```text
[1, 4 + num_classes, 8400]
```

raw detection。

### Rewrite

走通用 Rewrite，主要针对 NMS 等。

### Backend

可以进入通用 Kunlun Backend。

### Runtime

**问题开始出现。**

当前交付 Runtime 目录只有：

```text
runtime/common
runtime/detection
```

README 也明确写的是公共 Runtime + `detection/`。([GitHub][1])

所以 YOLOv5 Detection：

> **链路结构上成立，但没有证据证明真实 XPU 已验证。**

### 结论

**B：实验性 Detection。**

### 应修改

```text
xpu_converter/frontend/pytorch/yolov5.py
configs/models/yolov5.yaml
tests/models/test_yolov5.py
tests/e2e/test_yolov5_detection.py
```

增加：

```text
官方 yolov5u 权重
→ PyTorch output
→ ONNX output
→ ORT output
→ Paddle-XPU output
→ Runtime /predict
```

五层回归。

---

# 四、YOLOv6

这是目前前端实现比较认真的一个。

## Adapter

```text
xpu_converter/frontend/pytorch/yolov6.py
```

它没有伪装成 Ultralytics，而是明确：

```text
原生 YOLOv6
pt → onnx → paddle
```

而且需要：

```text
/home/compose/develop/yolov6
```

或者：

```text
XPU_MEITUAN_YOLOV6_REPO
```

源码才能反序列化 checkpoint。

它还修改了：

```python
model.forward
detect.export = False
```

把输出固定成：

```text
[B, N, 5+nc]
```

这属于**真正的模型适配逻辑**，不是简单注册。

### YAML

存在：

```text
configs/models/yolov6.yaml
```



### 最大问题

外部源码路径硬编码：

```text
/home/compose/develop/yolov6
```

这不适合交付环境。

### 应修改

增加：

```yaml
source:
  repository: ...
  revision: ...
  required: true
```

然后 Adapter 不直接依赖绝对路径。

建议：

```text
xpu_converter/frontend/pytorch/vendors/yolov6/
```

或者明确 `ModelDependencyResolver`。

### 结论

**B：前端真实实现，但不是稳定交付。**

---

# 五、YOLOv7

这是当前项目里前端工作量最大的一个。

文件：

```text
xpu_converter/frontend/pytorch/yolov7.py
```

包含：

```text
YOLOv7Adapter
YOLOv7PoseAdapter
YOLOv7SegAdapter
```



## Detection

输出：

```text
[1, N, 85]
```

并通过：

```python
detect.export = False
detect.concat = True
```

获得统一输出。

这是合理的。

## Pose

输出：

```text
[1, N, 4+(1+nc)+3K]
```

并且：

```python
raw_output_index = 0
```

避免导出旁支。



## Seg

这里更复杂：

```text
det + mask coefficients
+
proto
```

也就是双输出。

源码甚至有：

```text
vendors/yolov7_seg_shim.py
```

解决 checkpoint 中：

```text
SegmentationModel
ISegment
Proto
ImplicitA/M
```

等类缺失的问题。

### 这是“真的实现”吗？

**Frontend 层：是。**

但：

```text
Runtime
Output Decoder
Application Validation
Docker
```

目前并没有对应的完整 Seg/Pose Runtime。

所以不能把：

```text
yolov7-seg
```

标成生产支持。

### YAML

甚至已经有：

```text
configs/models/yolov7-pose.yaml
```

并定义：

```text
task: pose
input: 1280
num_classes: 1
```



这说明作者确实在往多任务走。

### 结论

| YOLOv7    | 判断 |
| --------- | -- |
| Detection | B  |
| Pose      | C  |
| Seg       | C  |

---

# 六、YOLOv8

Adapter：

```text
xpu_converter/frontend/pytorch/yolov8.py
```

有：

```text
YOLOv8
YOLOv8Seg
YOLOv8Pose
YOLOv8Obb
YOLOv8Cls
```



YAML 也对应存在：

```text
yolov8.yaml
yolov8-seg.yaml
yolov8-pose.yaml
yolov8-obb.yaml
yolov8-cls.yaml
```

例如 Seg/Pose/OBB/Cls 配置都已经存在。

### 但这里有一个非常关键的假完成

YOLOv8 的：

```python
YOLOv8SegAdapter
YOLOv8PoseAdapter
YOLOv8ObbAdapter
YOLOv8ClsAdapter
```

核心实现基本是：

```python
class YOLOv8SegAdapter(YOLOv8Adapter):
    task = "segment"
```

而不是：

```text
SegmentExporter
PoseExporter
OBBExporter
ClassificationExporter
```



也就是说：

> **模型类型注册已经完成，但任务级转换器没有真正分叉出来。**

### 更严重的问题

通用 PyTorch Adapter 对 Ultralytics 模型的策略是：

```text
YOLO(path)
  ↓
model.export(format="onnx")
```



这本身没问题，但你必须随后针对不同任务解析输出。

现在 Runtime 交付层却只有 Detection。

### 结论

| YOLOv8         | 判断 |
| -------------- | -- |
| Detection      | B  |
| Seg            | C  |
| Pose           | C  |
| OBB            | C  |
| Classification | C  |

---

# 七、YOLOv9

这个 Adapter 比 YOLOv8 更扎实。

```text
xpu_converter/frontend/pytorch/yolov9.py
```

Detection：

```text
raw_output_index = 0
```

并把 Detect 头：

```text
export=True
```

得到：

```text
[1, 4+nc, N]
```



Segmentation：

```text
output1 = det + mask coeff
output2 = proto
```

源码明确设计成：

```text
SegmentDecoder
```

可以消费的形式。

### YAML

Detection YAML 存在：

```text
configs/models/yolov9.yaml
```



### 但是

仍然依赖：

```text
XPU_THUYNGCH_YOLOV9_REPO
```

默认：

```text
/home/compose/develop/yolov9
```

而且还需要 torchvision stub。

### 结论

| YOLOv9       | 判断 |
| ------------ | -- |
| Detection    | B  |
| Segmentation | C  |

---

# 八、YOLOv10 —— 当前唯一真正应该当作 Golden Path

Adapter：

```text
xpu_converter/frontend/pytorch/yolov10.py
```

只有 18 行，但这是合理的。

因为它本身依赖 Ultralytics。

核心：

```python
model_type = "yolov10"
ultralytics_name = "yolov10"
end2end_default = True
```



### YAML

```text
configs/models/yolov10.yaml
```

明确：

```text
input: 1x3x640x640
opset: 13
end2end: false
NMS: CPU
```



### Rewrite

这里是关键：

```text
NonMaxSuppression
        ↓
CPU NMS
```

项目专门实现了：

```text
xpu_converter/rewrite/nms.py
```

而且不是简单删除节点，而是：

1. 找 NMS
2. 找 downstream
3. 判断 graph closure
4. 判断有没有其它 output
5. 最终将 graph output 替换成 boxes/scores



这个设计是目前项目里**比较成熟的一块**。

### Backend

真实路径：

```text
ONNX
 ↓
X2Paddle
 ↓
Paddle static graph
 ↓
Paddle Inference
 ↓
enable_xpu()
```

源码明确说明 Paddle XPU kernel 在加载时编译。

### Runtime

存在：

```text
PaddleXpuRuntimeSession
```

能够：

```python
config.enable_xpu()
config.set_xpu_device_id()
```

并执行 predictor。

### Validation

精度验证明确区分：

```text
Level 1:
ONNX vs optimized ONNX

Level 2:
optimized ONNX vs XPU

Level 3:
application mAP
```

这个设计也是正确的。

### Benchmark

还专门禁止：

```text
ONNX Runtime CPU
```

伪装成 XPU benchmark：

```text
simulated → NOT_AVAILABLE
```



### Docker

degraded artifact 默认禁止进入正式 package。

### 结论

**A：当前唯一真正应该作为 Stable/Golden Path 的模型。**

---

# 九、YOLO11

Adapter：

```text
yolov11.py
```

存在：

```text
Detection
Seg
Pose
OBB
Cls
```



YAML 也有对应任务配置，例如：

```text
yolov11.yaml
yolov11-seg.yaml
...
```



但和 YOLOv8 一样：

```python
class YOLOv11SegAdapter(YOLOv11Adapter):
    task = "segment"
```

主要还是 task declaration。

### 结论

| YOLO11    | 判断 |
| --------- | -- |
| Detection | B  |
| Seg       | C  |
| Pose      | C  |
| OBB       | C  |
| Cls       | C  |

---

# 十、YOLO12

Adapter：

```text
xpu_converter/frontend/pytorch/yolo12.py
```

实际就是：

```python
ultralytics_name = "yolo12"
```

raw detection：

```text
[1, 4+nc, 8400]
```



YAML：

```text
configs/models/yolov12.yaml
```

存在。

没有 Seg/Pose/OBB/Cls Adapter。

### 结论

**B：Detection experimental。**

---

# 十一、YOLO26 —— 当前最值得重点整改

Adapter：

```text
xpu_converter/frontend/pytorch/yolo26.py
```

注册：

```text
yolov26
yolov26-seg
yolov26-pose
yolov26-obb
yolov26-cls
yolov26-depth
yolov26-sem
```



这就是之前“YOLO26 支持 7 个任务”这个说法的来源。

但是看实现：

```python
class YOLO26DepthAdapter(YOLO26Adapter):
    model_type = "yolov26-depth"
    task = "depth"

class YOLO26SemAdapter(YOLO26Adapter):
    model_type = "yolov26-sem"
    task = "sem"
```

就结束了。

也就是说：

```text
Depth
Semantic
```

没有看到：

```text
Depth decoder
Semantic decoder
task-specific backend contract
task-specific runtime
task-specific validation
```

### YAML

反而已经有：

```text
configs/models/yolov26-depth.yaml
```

里面甚至写：

```text
逐像素深度图输出
```



**这恰恰暴露了一个设计问题：**

配置文件声称的能力 > Runtime 实际能力。

---

# 十二、输出契约是当前最大的结构性缺陷

现在项目在 YAML 里大量写：

```yaml
output_layout: auto
```

而 `ModelConfig` 的说明是：

> 输出契约为空时，由实际 ONNX 图探测。

这个对于 Detection 可以接受。

但对于：

```text
Detection
Segmentation
Pose
OBB
Classification
Depth
Semantic Segmentation
```

已经不够了。

因为它们的输出语义完全不同。

应该建立：

```text
OutputContract
```

例如：

```yaml
output:
  type: detection
  layout: BNC
  boxes:
    format: xywh
  objectness: true
  classes:
    activation: sigmoid
```

Seg：

```yaml
output:
  type: segmentation
  detection:
    layout: BCN
  mask_coeff:
    channels: 32
  proto:
    layout: BCHW
```

Pose：

```yaml
output:
  type: pose
  keypoints:
    count: 17
    dims: 3
```

OBB：

```yaml
output:
  type: obb
  angle:
    unit: radian
```

Cls：

```yaml
output:
  type: classification
  layout: BC
```

Depth：

```yaml
output:
  type: depth
  layout: BCHW
```

Sem：

```yaml
output:
  type: semantic_segmentation
  layout: BCHW
```

**现在缺的就是这一层“任务级标准契约”。**

---

# 十三、Rewrite 层审计

当前 Rewrite 做得比较好的地方是：

```text
xpu_converter/rewrite/nms.py
```

NMS rewrite 有比较严谨的安全检查。

但是当前 Rewrite 的思想还是：

```text
后端不支持某个 OP
        ↓
把 OP 改写掉
```

对于 YOLO 多任务，还应该增加：

```text
YOLO Task Rewrite
```

例如：

```text
Ultralytics Segment
        ↓
统一 Segment IR
```

而不是让 Runtime 猜：

```text
output1 是什么？
output2 是什么？
```

建议新增：

```text
xpu_converter/rewrite/yolo/
    detection.py
    segmentation.py
    pose.py
    obb.py
    classification.py
```

---

# 十四、Backend 审计：这里必须明确“真实 XPU”和“模拟 XPU”

这是当前代码里我认为做得比较诚实的地方。

Backend 有：

```text
PaddleXpuSdkAdapter
XpuToolkitSdkAdapter
StubSdkAdapter
```



其中：

### Paddle

是真实路线：

```text
ONNX → Paddle static graph
```

然后：

```text
Paddle Inference → XPU
```

### xpuctl

仍然是：

```text
TODO(SDK)
```

当前只是抽象接口。

### stub

明确：

```text
ONNX copy
```

并且：

```text
degraded
```

禁止正式交付。

### 所以

**不能说当前项目完全没有 XPU Backend。**

更准确：

> **Paddle-XPU 路线已经实现；原生 XPU Toolkit/XTCL 路线还是抽象预留。**

---

# 十五、Runtime 是整个项目当前最明显的“能力断层”

README 明确列的是：

```text
runtime/
├── common/
└── detection/
```

也明确写：

```text
Detection service:
/predict
/predict_image
/health
/setflag
```

([GitHub][1])

但你现在 Adapter 已经注册：

```text
segment
pose
obb
cls
depth
sem
```

所以出现：

```text
Frontend 能力
       ↓
       ↓
       ↓
Runtime 能力
```

中间断层。

### DockerExporter 更直接证明这一点

它根据：

```python
RuntimePackager(task=self.runtime)
```

去找：

```text
runtime/common
runtime/<task>
```



当前只有：

```text
runtime/detection
```

那么：

```text
runtime=segment
```

实际上就会因为目录不存在而失败：

```python
raise PackageError("Runtime 目录不存在")
```



### 这意味着

**YOLOv8-seg 虽然 Adapter 存在，但无法形成完整 Docker Runtime 交付。**

这就是一个非常典型的：

> **“代码存在 ≠ 功能完成”**

---

# 十六、Tests 审计

仓库有 `tests/` 目录，而且 README 的测试入口是：

```bash
python -m unittest discover -s tests -t .
```

但 README 明确说明：

> 没有 torch / ultralytics 时，测试使用**合成 ONNX 图**验证优化、改写、编译、校验、打包全链路。([GitHub][1])

这意味着目前测试体系主要验证：

```text
Pipeline Framework
```

而不是：

```text
YOLO Model Conformance
```

这是一个非常大的区别。

## 现在缺少的测试层

应该建立：

```text
tests/
├── unit/
├── model/
│   ├── yolov5/
│   ├── yolov6/
│   ├── yolov7/
│   ├── yolov8/
│   ├── yolov9/
│   ├── yolov10/
│   ├── yolov11/
│   ├── yolov12/
│   └── yolov26/
├── runtime/
├── backend/
├── e2e/
└── fixtures/
```

每个模型至少：

```text
test_load
test_export
test_output_contract
test_onnxruntime
test_xpu
test_runtime
test_package
```

---

# 十七、真正的模型能力矩阵

这是我建议你直接作为项目正式 `SUPPORT_MATRIX.yaml` 的第一版：

| Model       | Detection | Seg | Pose | OBB | Cls | Depth | Sem | Stable |
| ----------- | --------: | --: | ---: | --: | --: | ----: | --: | -----: |
| YOLOv5u     |        🟡 |   — |    — |   — |   — |     — |   — |      ❌ |
| YOLOv6      |        🟡 |   — |    — |   — |   — |     — |   — |      ❌ |
| YOLOv7      |        🟡 |  🟠 |   🟠 |   — |   — |     — |   — |      ❌ |
| YOLOv8      |        🟡 |  🟠 |   🟠 |  🟠 |  🟠 |     — |   — |      ❌ |
| **YOLOv10** |    **🟢** |   — |    — |   — |   — |     — |   — |  **✅** |
| YOLOv9      |        🟡 |  🟠 |    — |   — |   — |     — |   — |      ❌ |
| YOLO11      |        🟡 |  🟠 |   🟠 |  🟠 |  🟠 |     — |   — |      ❌ |
| YOLO12      |        🟡 |   — |    — |   — |   — |     — |   — |      ❌ |
| YOLO26      |        🟡 |  🟠 |   🟠 |  🟠 |  🟠 |    🔴 |  🔴 |      ❌ |

其中：

* 🟢 = 当前项目可以作为正式 Golden Path
* 🟡 = Frontend/ONNX 路线已有
* 🟠 = Adapter/YAML 有，但 Runtime/Validation 不完整
* 🔴 = **目前明显属于“声明式支持”，不能称为已实现**

---

# 十八、我认为当前仓库最重要的 10 个问题

按优先级排序：

## P0-1：Runtime 只有 Detection

修改：

```text
runtime/
```

至少增加：

```text
runtime/segmentation/
runtime/pose/
runtime/obb/
runtime/classification/
```

Depth/Sem 后续增加。

---

## P0-2：建立统一 OutputContract

新增：

```text
xpu_converter/contracts/
    base.py
    detection.py
    segmentation.py
    pose.py
    obb.py
    classification.py
    depth.py
    semantic.py
```

这是整个多任务扩展的核心。

---

## P0-3：Adapter 不应该自己“猜输出”

现在大量：

```text
output_layout: auto
```

应改成：

```text
Adapter
   ↓
OutputContract
   ↓
IR
   ↓
Runtime
```

---

## P0-4：YOLO26 Depth/Sem 立即降级

当前：

```text
yolov26-depth
yolov26-sem
```

不能继续挂在“支持列表”里让用户误以为可用。

建议：

```text
status = planned
```

或者：

```text
experimental
deliverable = false
```

直到 Runtime + Validation 完成。

---

## P0-5：YOLOv8/11 多任务也不能继续只靠 subclass

目前：

```python
class YOLOv11PoseAdapter(YOLOv11Adapter):
    task = "pose"
```

远远不够。

应该变成：

```text
YOLO11PoseAdapter
    ↓
PoseOutputContract
    ↓
PoseRuntime
    ↓
PoseValidator
```

---

## P0-6：外部 YOLOv6/7/9 源码依赖必须去绝对路径

现在：

```text
/home/compose/develop/yolov6
/home/compose/develop/yolov7
/home/compose/develop/yolov9
```

应该统一变成：

```text
ModelSourceResolver
```

通过：

```text
repository
revision
local_path
```

确定来源。

---

## P0-7：模型支持状态应该自动检查

现在：

```python
SUPPORT_TABLE
```

是人工声明。

应该自动生成：

```text
Adapter
+
YAML
+
OutputContract
+
Runtime
+
Test
+
Backend
```

只有全部存在：

```text
stable
```

否则：

```text
experimental
```

---

## P0-8：增加“模型级 Conformance Test”

例如：

```text
YOLOv10n
   ↓
golden input
   ↓
PyTorch
   ↓
ONNX
   ↓
Paddle
   ↓
XPU
```

每一步保存：

```text
tensor hash
shape
dtype
max_abs_error
cosine_similarity
```

---

## P0-9：Application-level Accuracy 不能只靠 generic tensor compare

当前 AccuracyValidator 已经很正确地把 Application Level 独立出来，但默认没有 dataset 时会标记 unavailable。

下一步应该真正实现：

```text
Detection:
mAP50
mAP50-95
Recall
Precision

Seg:
mask mAP

Pose:
OKS AP

OBB:
OBB mAP

Cls:
Top-1 / Top-5

Depth:
AbsRel / RMSE / δ1

Sem:
mIoU / Pixel Acc
```

---

## P0-10：Performance 必须绑定硬件指纹

当前 Benchmark 已经禁止模拟后端产生虚假性能数字，这是对的。

应该进一步强制：

```text
chip
sdk_version
driver_version
firmware_version
model
precision
input_shape
batch
warmup
iterations
```

全部写入 benchmark artifact。

---

# 十九、最终“真假完成”判断

我把你之前最关心的问题直接给一个非常明确的答案：

### 真完成

```text
YOLOv10 Detection
```

它是当前项目真正的：

```text
Golden Path
```

因为：

```text
Adapter
✓
YAML
✓
ONNX
✓
NMS Rewrite
✓
Operator Analysis
✓
Optimization
✓
Paddle-XPU Backend
✓
Runtime Session
✓
Accuracy Framework
✓
Benchmark Framework
✓
Docker Export
✓
degraded 防护
✓
Stable 生命周期
✓
```



---

### 基本真实，但还不能交付

```text
YOLOv5 Detection
YOLOv6 Detection
YOLOv7 Detection
YOLOv8 Detection
YOLOv9 Detection
YOLO11 Detection
YOLO12 Detection
YOLO26 Detection
```

原因不是 Adapter 不存在，而是：

```text
真实模型回归
+
XPU 真机验证
+
Runtime
+
Application Accuracy
+
Performance
+
正式交付
```

没有全部闭环。

---

### “假完成/过度声明”

尤其是：

```text
YOLOv8 Seg/Pose/OBB/Cls
YOLOv11 Seg/Pose/OBB/Cls
YOLOv26 Seg/Pose/OBB/Cls
YOLOv26 Depth
YOLOv26 Sem
```

这里的“假”不是说代码完全没写，而是：

> **代码已经表达了“支持这个任务”，但项目后端/Runtime/测试体系还没有足够证据支持“这个任务已经完成”。**

其中：

**YOLO26 Depth / Sem 是最典型的假完成。**

因为 Adapter 基本只有：

```python
task = "depth"
```

或：

```python
task = "sem"
```

而交付 Runtime 当前仍然是 detection-oriented。

---

# 二十、我建议下一阶段不要继续加模型，而是做这个重构

优先级应该从：

```text
❌ 再增加 YOLO13 / YOLO27 / 更多 Adapter
```

切换成：

```text
                         ┌── Detection
                         ├── Segmentation
                         ├── Pose
PT → Adapter → IR → Contract ├── OBB
                         ├── Classification
                         ├── Depth
                         └── Semantic
                                │
                                ▼
                         Rewrite / Optimizer
                                │
                                ▼
                         Kunlun Backend
                                │
                         ┌──────┴──────┐
                         │             │
                       XPU          CPU/ORT
                         │             │
                         └──────┬──────┘
                                ▼
                          Task Runtime
                                │
                                ▼
                       Application Validator
                                │
                                ▼
                         Docker Delivery
```

然后按照：

```text
Phase 1
YOLOv10 Detection
        ↓
100% Golden Path

Phase 2
YOLOv8 Detection
YOLO11 Detection
YOLOv9 Detection

Phase 3
YOLOv8 Seg
YOLOv8 Pose
YOLOv8 OBB
YOLOv8 Cls

Phase 4
YOLOv7/v9 Seg
YOLOv7 Pose

Phase 5
YOLO26 多任务
        ↓
Depth / Sem 最后做
```

逐个从：

```text
experimental
```

晋级：

```text
stable
```

---

## 最关键的一句话

**当前仓库不是“YOLO 转换没做完”，而是“Frontend 模型覆盖已经跑在 Backend/Runtime/Validation 前面了”。**

因此下一步最重要的工程任务不是继续写 Adapter，而是建立：

> **`OutputContract + Task Runtime + Model Conformance Test + Stable Gate`**

这四个东西一旦补上，当前已经写好的大量 YOLO Adapter 才真正有价值。

而且我建议下一步直接动代码：**先以 YOLOv10 Detection 为基准，把现有仓库逐文件改造成一套可复制的 `Model Conformance` 框架，然后把 YOLOv8 Detection 和 YOLO11 Detection 作为第一批从 experimental 晋级 stable 的模型。**这样后面 YOLOv5/6/7/9/12/26 就不需要再靠“写 Adapter + 人工判断”了，而是可以由自动化门禁判断到底是真支持还是假支持。

[1]: https://github.com/WangShaoze/xpu-model-converter "GitHub - WangShaoze/xpu-model-converter: 实现模型转换 · GitHub"
 