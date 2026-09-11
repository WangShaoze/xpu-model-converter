我已经把你上传的 `xpu-model-converter.zip` 按“**准备进入第一阶段实际开发/交付**”的标准做了一轮代码审查，而不是只看目录结构。

先给结论：

> **现在这套代码的架构方向是对的，已经有了 60%～70% 的工程骨架；但还不能作为真正的 YOLOv10 → 昆仑芯 XPU 转换器交付。**
>
> 最大的问题不是代码量，而是目前有几处“**模拟能力被设计成了真实能力的形状**”。尤其是 Kunlun Backend、算子白名单、YOLOv10 导出/后处理、Degraded 模式、精度验证这几处，如果不先改，后面接入真实昆仑 SDK 时会返工。

我建议**不要推倒重写**。保留现在的大体目录和 CLI，把核心链路做一次 V1.1 重构。

---

# 一、我对当前代码的总体评价

你现在的架构：

```text
xpu_converter
│
├── frontend
├── ir
├── optimizer
├── rewrite
├── backend
├── validator
├── exporter
├── registry
└── pipeline
```

这个方向是正确的。

尤其这几个设计值得保留：

* Frontend / Backend 解耦
* Model Registry
* Hardware Registry
* ONNX 作为中间表示
* Optimizer / Rewrite 独立
* Accuracy Validator
* Docker Exporter
* `manifest.json`
* Runtime 与模型解耦
* `runtime.tgz`
* CLI
* 测试目录

所以我**不建议推翻现有工程**。

但是目前存在下面几个等级的问题：

| 等级 | 问题                                                           |
| -- | ------------------------------------------------------------ |
| P0 | `stub` 默认可以一路生成“XPU 模型”和 Docker 包                            |
| P0 | Kunlun `xpuctl` 实际 API 是猜测性的，并不是真正 SDK 集成                    |
| P0 | 算子白名单是“假支持表”，没有绑定具体 XPU 型号/SDK 版本                            |
| P0 | YOLOv10 `.pt` 加载能力不足，无法覆盖真实客户模型交付场景                          |
| P0 | 精度验证实际上不能证明“PyTorch → XPU”正确                                 |
| P1 | 算子分析发生在 Rewrite 之前，不能代表最终编译图                                 |
| P1 | `validation_enabled=False` 仍然可能执行验证                          |
| P1 | benchmark 可以对 stub/CPU 仿真结果进行测试                              |
| P1 | Docker Runtime 中 XPU API 同样是猜测性的                             |
| P1 | YOLO 后处理逻辑过于泛化，容易出现“能跑但结果错”                                  |
| P1 | opset / dynamic / precision / shape 没有形成严格 Capability Matrix |
| P1 | manifest / config 的职责存在重复                                    |
| P2 | 版本、环境、SDK fingerprint 不够完整                                   |
| P2 | 缺少真正的 Golden Model / Golden Dataset                          |
| P2 | CI 目前更多是“结构测试”，不是转换正确性测试                                     |

---

# 二、最重要的问题：现在的 `stub` 设计需要改

你现在：

```python
StubSdkAdapter.compile()
```

实际上做的是：

```text
optimized.onnx
     ↓
copy
     ↓
model.xpu
```

然后：

```text
artifact_format = "stub"
degraded = true
```

这个设计用于**早期开发联调**是合理的。

问题在于：

```text
allow_degraded=True
```

现在是默认值。

CLI：

```python
allow_degraded=not getattr(args, "strict", False)
```

意味着用户直接：

```bash
xpu-converter convert ...
```

在没有昆仑 SDK 的机器上，会：

```text
ONNX
 ↓
stub
 ↓
model.xpu
 ↓
accuracy
 ↓
benchmark
 ↓
Docker ZIP
```

最终用户可能看到：

```text
yolov10_dockerimg_v1.0.zip
```

这非常危险。

---

# 三、必须把“开发模式”和“生产模式”彻底分开

建议修改为：

```text
PRODUCTION
    ↓
没有真实 Kunlun Backend
    ↓
直接失败
```

只有显式：

```bash
xpu-converter convert \
    --model best.pt \
    --allow-degraded
```

才允许：

```text
stub
```

而且：

> **`--allow-degraded` 只能用于开发测试，不能生成可交付 ZIP。**

建议：

```python
if artifact.degraded and export_docker:
    raise PackageError(
        "当前为 degraded/stub 产物，禁止生成正式 Docker 交付包"
    )
```

如果你希望测试 Docker Exporter，可以单独：

```bash
xpu-converter package \
    --model model.xpu \
    --allow-degraded-package
```

但生产 CLI 默认绝对不能这样。

---

# 四、P0：Kunlun Backend 目前实际上还没有真正实现

这是当前项目最大的技术空洞。

你现在：

```python
XPU_TOOLKIT_MODULES = (
    "xpu_toolkit",
    "xtcl",
    "xpu_inference"
)
```

然后：

```python
compiler_cls = _first_attr(
    module,
    ("XpuCompiler", "Compiler", "XPUCompiler")
)
```

再：

```python
compiler = compiler_cls(xpu_graph.onnx_path)

result = compile_method(
    output=str(target),
    precision=xpu_graph.precision
)
```

这个逻辑**不能作为真实昆仑 SDK 集成**。

因为它实际上假设：

```text
import xtcl
     ↓
Compiler(...)
     ↓
compile(output=..., precision=...)
```

但我们现在并不知道你们实际 SDK 是不是这种 API。

所以这段代码不能继续向下堆。

---

# 五、正确修改方式：把 Backend 变成 Capability-driven

现在：

```text
KunlunBackend
    ↓
猜 SDK API
```

改成：

```text
KunlunBackend
      │
      ▼
KunlunCapability
      │
      ├── chip
      ├── sdk_version
      ├── compiler_version
      ├── supported_ops
      ├── supported_precisions
      ├── supported_opsets
      ├── dynamic_shape
      └── layout
```

例如：

```python
@dataclass
class KunlunCapabilities:
    chip: str
    sdk_version: str
    compiler_version: str

    supported_ops: Set[OperatorSpec]

    supported_precisions: Set[str]
    supported_opsets: Set[int]

    dynamic_shape: bool
    layouts: Set[str]
```

然后：

```python
class KunlunBackend(BaseBackend):

    def capabilities(self):
        return self.adapter.capabilities()

    def compile(self, graph, config):
        self.preflight(graph)
        return self.adapter.compile(graph, config)
```

---

# 六、SDK Adapter 不应该“猜 API”

应该变成：

```text
backend/kunlun/
│
├── backend.py
├── capabilities.py
│
└── adapters/
    ├── base.py
    ├── sdk_v1.py
    ├── sdk_v2.py
    └── paddle_xpu.py
```

例如：

```python
class KunlunCompilerAdapter(ABC):

    @abstractmethod
    def probe(self) -> KunlunEnvironment:
        ...

    @abstractmethod
    def compile(
        self,
        graph: XpuGraph,
        config: KunlunCompileConfig,
    ) -> BackendArtifact:
        ...

    @abstractmethod
    def create_runtime(
        self,
        artifact: BackendArtifact,
    ):
        ...
```

拿到你们真实 SDK 后，只实现：

```python
class KunlunSdkVxAdapter(KunlunCompilerAdapter):
    ...
```

而不是继续：

```python
getattr(module, "Compiler")
getattr(module, "compile")
```

这种反射猜测。

---

# 七、P0：算子白名单现在不能当真实数据使用

当前：

```python
NATIVE_OPS = {
    "Conv",
    "BatchNormalization",
    "Resize",
    ...
}
```

问题非常大。

例如：

```text
Resize
```

是否支持，不应该只有：

```text
Resize = true
```

而应该是：

```text
Resize
├── mode=nearest
│     ├── FP32 ✓
│     ├── FP16 ✓
│     └── INT8 ?
│
└── mode=bilinear
      ├── FP32 ✓
      ├── FP16 ?
      └── INT8 ✗
```

再例如：

```text
Conv
```

还涉及：

```text
kernel
stride
padding
dilation
groups
layout
dtype
```

所以当前：

```python
Set[str]
```

太简单。

---

# 八、把 Operator Registry 升级成 Operator Capability

建议：

```python
@dataclass(frozen=True)
class OperatorCapability:
    op_type: str
    domain: str = ""

    dtypes: Set[str] = field(default_factory=set)

    attributes: Dict[str, Set[Any]] = field(
        default_factory=dict
    )

    min_opset: int = 1
    max_opset: Optional[int] = None
```

例如：

```yaml
operators:

  Conv:
    dtypes:
      - float32
      - float16

  Resize:
    dtypes:
      - float32
      - float16
    attributes:
      mode:
        - nearest

  MatMul:
    dtypes:
      - float16
```

最终分析：

```text
Resize
    mode=bilinear
    dtype=fp16
    opset=13

=> UNSUPPORTED

Reason:
    Resize(mode=bilinear, dtype=fp16)
    is not supported by Kunlun P800 SDK 3.2
```

这才是真正的 Converter。

---

# 九、算子分析顺序必须调整

当前：

```text
ONNX
 ↓
Operator Analysis
 ↓
Optimizer
 ↓
Rewrite
 ↓
Compile
```

有问题。

例如：

```text
SiLU
```

原始图里：

```text
SiLU
```

算子分析会认为：

```text
REWRITTEN
```

但是经过：

```text
SiLU
 ↓
Sigmoid + Mul
```

最终图已经不再有 SiLU。

所以真正应该：

```text
Raw ONNX
   ↓
Graph Check
   ↓
Normalize
   ↓
Rewrite
   ↓
Optimize
   ↓
Final Operator Analysis
   ↓
Preflight
   ↓
Compile
```

即：

### 第一次分析

用于：

```text
diagnostic
```

### 第二次分析

用于：

```text
compile gate
```

最终只认第二次。

---

# 十、建议把 Pipeline 改成这样

你现在的：

```text
10 steps
```

我建议保留 CLI 显示的 10 步，但内部改成：

```text
01 Inspect
02 Load
03 Export
04 Validate ONNX
05 Normalize / Rewrite
06 Optimize
07 Final Capability Check
08 Compile
09 Validate / Benchmark
10 Package
```

尤其：

```text
Final Capability Check
```

必须是硬门禁。

---

# 十一、P0：YOLOv10 `.pt` 加载器需要重点重构

你现在：

```python
torch.load(...)
```

然后：

```python
if isinstance(obj, torch.nn.Module):
    return obj
```

或者：

```python
for key in ("model", "ema", "net", "network"):
    ...
```

这只能覆盖一部分情况。

实际客户模型可能是：

```text
best.pt
```

里面：

```python
{
    "epoch": ...,
    "model": ...,
    "optimizer": ...,
    "ema": ...,
    "updates": ...,
    "train_args": ...,
}
```

也可能：

```python
{
    "state_dict": ...
}
```

甚至：

```python
{
    "model": "...",
    "yaml": "...",
}
```

---

# 十二、必须建立 Model Source Contract

这是我认为当前项目缺少的一个非常重要的概念。

不要认为：

```text
best.pt
```

就是一个完整模型。

应该定义：

```text
Model Package
```

例如：

```text
customer_model/
├── best.pt
├── model.yaml
├── classes.txt
├── requirements.txt
└── metadata.json
```

其中：

```yaml
framework: pytorch
model_type: yolov10

source:
  repository: ultralytics
  version: xxx

model:
  checkpoint: best.pt

input:
  shape: [1,3,640,640]
```

这样你才能保证：

> 转换机拥有恢复这个模型所需要的代码和版本。

---

# 十三、对于 Ultralytics 模型，增加 Environment Fingerprint

例如：

```json
{
  "framework": "pytorch",
  "framework_version": "2.x",
  "ultralytics_version": "x.x.x",
  "python_version": "3.10",
  "cuda_version": "",
  "checkpoint": "best.pt"
}
```

转换时：

```text
best.pt
 +
Ultralytics version
 +
PyTorch version
```

必须记录。

最终：

```text
metadata.json
```

里面必须有：

```text
source_framework
source_framework_version
source_library
source_library_version
python_version
exporter_version
```

否则半年后你根本无法复现一次转换。

---

# 十四、YOLOv10 Adapter 目前还有一个重要问题

你现在：

```python
end2end_default = True
```

同时注释：

> Ultralytics 导出的 ONNX 默认仍会带 NonMaxSuppression

这件事情**不能作为假设**。

必须对实际导出的 ONNX 做检测：

```text
Graph
 ├── outputs
 ├── NMS nodes
 ├── detection head
 └── output shape
```

然后自动判断：

```python
ModelOutputContract.detect(graph)
```

例如：

```text
Case A:

output:
[1, 300, 6]

=> end2end detection

Case B:

output:
[1, 4+nc, 8400]

=> raw detection

Case C:

NonMaxSuppression
+
Gather
+
output

=> postprocess embedded
```

不能简单：

```text
model_type == yolov10
    ↓
一定是某一种 output
```

---

# 十五、我建议增加 Output Contract

这是 YOLO Runtime 目前缺少的关键层。

```python
@dataclass
class OutputContract:

    type: str

    boxes_format: str
    score_format: str

    class_axis: Optional[int]

    has_objectness: bool

    normalized: bool

    end2end: bool

    nms_embedded: bool
```

例如：

```yaml
output:
  type: detection
  format: cxcywh
  layout: BCH
  has_objectness: false
  end2end: false
  nms: cpu
```

然后 Runtime 不需要“猜”。

---

# 十六、现在 Runtime 的 `_parse_outputs()` 太危险

现在：

```python
if array.shape[-1] == 6:
    ...
```

然后：

```python
if array.shape[0] < array.shape[1]:
    ...
```

这是典型：

> **heuristic parser**

短期能跑，长期一定出错。

例如：

```text
[1, 300, 6]
```

和：

```text
[1, 6, 300]
```

虽然能猜，但：

```text
4+nc
5+nc
end2end
objectness
```

语义完全不同。

所以必须由：

```text
model.yaml
```

明确声明。

---

# 十七、Runtime 应该变成 Contract-driven

不要：

```python
_parse_outputs()
```

猜格式。

改：

```python
decoder = DecoderFactory.create(
    config.output_contract
)
```

例如：

```text
runtime/detection/decoders/
├── base.py
├── yolov8_raw.py
├── yolov10_raw.py
├── end2end_6col.py
└── generic.py
```

然后：

```python
decoder.decode(outputs)
```

---

# 十八、P0：当前精度验证还不能证明真正的 XPU 精度

现在：

```text
reference = ONNX CPU
target = backend.create_runtime()
```

如果是 stub：

```text
target = ONNX Runtime
```

于是：

```text
ONNX
  ↓
ONNX Runtime
```

对比：

```text
ONNX
  ↓
ONNX Runtime
```

当然几乎完全一致。

这只能证明：

> stub 没有把文件复制坏。

不能证明：

> 昆仑 XPU 正确。

---

# 十九、V1 应该明确三种 Validation Level

## Level 1：Graph Validation

```text
ONNX
 ↓
Optimized ONNX
```

比较：

```text
原 ONNX
优化 ONNX
```

---

## Level 2：Backend Validation

```text
Optimized ONNX
       ↓
     CPU
       vs
       XPU
```

这才是真正：

```text
XPU numerical validation
```

---

## Level 3：Application Validation

YOLO：

```text
Image
 ↓
PyTorch
 ↓
Detection
```

vs：

```text
Image
 ↓
XPU
 ↓
Detection
```

比较：

```text
IoU
Precision
Recall
mAP
```

这三级必须区分。

---

# 二十、Accuracy Report 建议最终长这样

```json
{
  "graph": {
    "passed": true,
    "max_abs_error": 1.2e-6
  },

  "backend": {
    "reference": "onnxruntime-cpu",
    "target": "kunlun-xpu",
    "passed": true,
    "max_abs_error": 0.00031,
    "cosine_similarity": 0.999997
  },

  "application": {
    "task": "detection",
    "samples": 500,
    "precision": 0.9231,
    "recall": 0.9173,
    "map50": 0.9341,
    "map5095": 0.7212
  }
}
```

---

# 二十一、P1：`validation_enabled` 要真正生效

现在 pipeline：

```python
self._run_step(7, self._step_validate)
```

无论：

```text
validation_enabled
```

是什么，都会进入。

应该：

```python
if self.validation_enabled:
    self._run_step(7, self._step_validate)
else:
    self._skip_step(...)
```

并且不能把：

```text
SKIPPED
```

打印成：

```text
OK
```

建议：

```text
[08] Accuracy Validation SKIPPED
```

---

# 二十二、Benchmark 也一样

如果：

```text
backend = stub
```

则：

```text
Benchmark = NOT_AVAILABLE
```

而不是：

```text
avg=0.03ms
```

否则很容易有人把：

```text
ONNXRuntime CPU
```

的数据拿去宣传：

```text
昆仑 XPU 性能
```

---

# 二十三、Benchmark 必须记录硬件指纹

正式 benchmark：

```json
{
  "backend": "kunlun-xpu",
  "chip": "xxx",
  "device_id": 0,
  "sdk_version": "xxx",
  "driver_version": "xxx",
  "firmware_version": "xxx",

  "model": "yolov10n",
  "input_shape": [1,3,640,640],
  "precision": "fp16",

  "warmup": 20,
  "iterations": 500,

  "latency_ms": {
    "mean": 4.31,
    "p50": 4.20,
    "p90": 4.62,
    "p99": 5.13
  }
}
```

这样才有工程意义。

---

# 二十四、P1：Docker Runtime 的 XPU API 也不能继续猜

当前：

```python
XPU_TOOLKIT_MODULES = (
    "xpu_toolkit",
    "xtcl",
    "xpu_inference"
)
```

然后：

```python
XpuRuntime
Runtime
XpuInference
```

这是和 Compiler 层一样的问题。

必须变成：

```text
Backend
   │
   ├── CompilerAdapter
   │
   └── RuntimeAdapter
```

而且：

```text
Converter Runtime
```

和：

```text
Docker Runtime
```

最好共用同一个 Runtime Adapter。

否则会出现：

```text
转换器认为：
SDK API A

Docker：
SDK API B
```

最后转换成功但 Docker 跑不起来。

---

# 二十五、建议增加一个 `xpu_runtime` 抽象

```python
class XpuRuntimeAdapter(ABC):

    @classmethod
    def probe(cls):
        ...

    def load(self, artifact):
        ...

    def infer(self, inputs):
        ...

    def close(self):
        ...
```

然后：

```text
KunlunSdkAdapter
      │
      ├── compile()
      └── runtime()
```

保证编译和运行使用同一 SDK contract。

---

# 二十六、P1：`configs/hardware/kunlun.yaml` 现在混了太多东西

目前：

```yaml
base_image:
sdk_adapter:
target_chip:
precision:
device:
optimization_level:

runtime:
...

docker:
...
log:
...
```

建议拆：

```text
configs/
├── hardware/
│   └── kunlun/
│       ├── base.yaml
│       ├── capabilities.yaml
│       ├── sdk.yaml
│       └── docker.yaml
│
├── models/
│   └── yolov10.yaml
│
└── runtime/
    └── detection.yaml
```

职责清晰很多。

---

# 二十七、特别是 Docker 配置不能属于 Hardware Capability

例如：

```yaml
base_image:
packages_dir:
minio:
kafka:
```

这些不是：

```text
Kunlun XPU Capability
```

而是：

```text
Deployment Environment
```

建议：

```text
hardware/
    kunlun.yaml

deployment/
    kunlun_docker.yaml
```

---

# 二十八、P1：现在 `manifest.json` 和 `model.yaml` 有职责重叠

现在：

```text
model/model.yaml
```

实际上是 Build Manifest。

而：

```text
manifest.json
```

又是 Package Manifest。

这两个应该明确：

### model.yaml

描述：

> **如何产生这个模型**

例如：

```yaml
source:
target:
input:
optimization:
validation:
```

### manifest.json

描述：

> **这个 Docker 包里面有什么**

例如：

```json
{
  "package_name": "...",
  "files": [],
  "checksums": [],
  "runtime": {},
  "artifact": {}
}
```

不要让两者都成为“唯一事实来源”。

---

# 二十九、建议引入 Artifact Manifest

实际上最终应该有三个 manifest：

```text
1. build.yaml
   转换输入

2. artifact.json
   model.xpu 的来源与编译信息

3. manifest.json
   Docker Package 内容
```

关系：

```text
build.yaml
    ↓
artifact.json
    ↓
manifest.json
```

这样审计非常清楚。

---

# 三十、Artifact metadata 建议扩展

现在：

```json
{
  "sdk_adapter": "...",
  "precision": "...",
  "target_chip": "..."
}
```

远远不够。

建议：

```json
{
  "artifact_format": "kunlun_xpu",

  "converter": {
    "name": "xpu-model-converter",
    "version": "1.1.0"
  },

  "source": {
    "framework": "pytorch",
    "framework_version": "2.x",
    "model_type": "yolov10",
    "model_sha256": "..."
  },

  "export": {
    "onnx_opset": 17,
    "input_shape": [1,3,640,640]
  },

  "optimization": {
    "level": 2,
    "passes": [...]
  },

  "backend": {
    "name": "kunlun",
    "chip": "...",
    "sdk_version": "...",
    "compiler_version": "..."
  },

  "precision": "fp16",

  "artifact": {
    "file": "model.xpu",
    "sha256": "..."
  }
}
```

---

# 三十一、P1：必须增加 SHA256 的 Source Model Fingerprint

这是以后非常有用的东西。

比如：

```text
best.pt
SHA256:
b3d7....
```

转换得到：

```text
model.xpu
SHA256:
7f21....
```

最终：

```text
Docker Package
```

都记录。

以后客户说：

> “这个模型不是我给你的那个模型。”

你可以直接验证。

---

# 三十二、P1：需要增加 Preflight

现在流程是：

```text
compile
```

过程中才发现问题。

应该在 compile 前：

```bash
xpu-converter preflight ...
```

输出：

```text
Environment
=================================
Python             3.10
PyTorch            2.x
ONNX               1.x

Kunlun
=================================
Chip               P800
SDK                3.x
Compiler           x.x
Driver             x.x

Model
=================================
YOLOv10
Input              [1,3,640,640]
Precision          FP16

Operator
=================================
Conv               72      OK
BN                 72      FUSED
SiLU               72      REWRITTEN
Resize              5      OK

Unsupported:
0

Result:
READY
```

如果：

```text
GridSample
```

不支持：

```text
Result:
NOT READY
```

根本不要进入 compile。

---

# 三十三、P0：要建立 Capability Matrix

最终你会非常需要这个：

```text
                    Kunlun P800 SDK X
------------------------------------------------
YOLOv10n FP32       ✓
YOLOv10n FP16       ✓
YOLOv10s FP16       ✓
YOLOv10m FP16       ✓

Static Shape        ✓
Dynamic Shape       ✗

NCHW                ✓
NHWC                ?

Resize nearest     ✓
Resize bilinear     ?

NMS                 CPU
```

代码：

```text
capability/
├── model_capability.py
├── operator_capability.py
└── hardware_capability.py
```

---

# 三十四、优化器目前也需要一个重要原则

你现在有：

```text
constant_fold
fusion
graph_simplify
shape_inference
```

方向没问题。

但以后必须遵守：

> **所有 Graph Rewrite 必须经过 numerical equivalence test。**

尤其：

```text
Conv + BN
```

必须考虑：

```text
training/eval
epsilon
bias
weight sharing
dtype
```

你现在已经考虑了 weight sharing，这是好的。

继续保持这个思路。

---

# 三十五、Optimizer 应该变成 Pass Manager

建议：

```python
class PassManager:

    def run(self, graph):
        for pass_ in self.passes:

            before = snapshot(graph)

            result = pass_.run(graph)

            if result.changed:
                validate_graph(graph)
```

每个 pass 记录：

```json
{
  "pass": "conv_bn_fusion",
  "changed": true,
  "nodes_before": 143,
  "nodes_after": 71,
  "numerical_check": true
}
```

---

# 三十六、Rewrite 同样要支持“失败回滚”

这是当前设计里很值得加强的一点。

现在：

```python
rewrite.apply(graph)
```

直接修改。

建议：

```text
Graph A
  │
  ▼
Copy
  │
  ▼
Rewrite
  │
  ▼
ONNX Check
  │
  ▼
Numerical Check
  │
 ┌┴─────────┐
PASS       FAIL
 │           │
 ▼           ▼
commit     rollback
```

尤其：

```text
NMS rewrite
Resize rewrite
YOLO head rewrite
```

后面很容易出问题。

---

# 三十七、NMS Rewrite 目前不要作为“通用 ONNX Rewrite”

这是我建议你特别改的一点。

现在：

```text
rewrite/nms.py
```

直接：

```text
NonMaxSuppression → CPU
```

实际上这不是普通 graph rewrite。

它是：

> **模型输出契约改变**

所以应该放到：

```text
frontend/model_contract
```

而不是简单：

```text
rewrite
```

例如：

```text
YOLOv10 Adapter
    ↓
Output Contract
    ↓
strip_postprocess()
    ↓
raw output
```

这样更合理。

---

# 三十八、推荐新的模块结构

不推翻现有结构，只调整：

```text
xpu_converter/
│
├── frontend/
│   ├── pytorch/
│   │   └── yolov10.py
│   │
│   └── paddle/
│
├── contract/
│   ├── input.py
│   ├── output.py
│   ├── preprocess.py
│   └── postprocess.py
│
├── ir/
│
├── passes/
│   ├── normalize/
│   ├── rewrite/
│   └── optimize/
│
├── capability/
│   ├── operators.py
│   ├── hardware.py
│   └── matrix.py
│
├── backend/
│   └── kunlun/
│       ├── backend.py
│       ├── compiler.py
│       ├── runtime.py
│       ├── capabilities.py
│       └── adapters/
│
├── validator/
│   ├── graph.py
│   ├── backend.py
│   ├── application.py
│   └── benchmark.py
│
├── artifact/
│   ├── manifest.py
│   └── metadata.py
│
├── exporter/
│
└── pipeline/
```

---

# 三十九、Pipeline 最终应该是这个样子

这是我建议你接下来真正实施的核心。

```text
                 best.pt
                    │
                    ▼
             ┌─────────────┐
             │ Model Probe  │
             └──────┬──────┘
                    │
                    ▼
              YOLOv10Adapter
                    │
                    ▼
              Load / Restore
                    │
                    ▼
             PyTorch Model
                    │
                    ▼
             ONNX Export
                    │
                    ▼
             ONNX Validator
                    │
                    ▼
          ┌────────────────────┐
          │ Graph Normalize    │
          │ Shape Inference    │
          │ Constant Folding   │
          │ Model Rewrite      │
          │ Conv-BN Fusion     │
          └─────────┬──────────┘
                    │
                    ▼
             Final ONNX Graph
                    │
                    ▼
          Capability Analyzer
                    │
             ┌──────┴──────┐
             │             │
            FAIL           PASS
             │             │
             ▼             ▼
           STOP       Kunlun Compiler
                           │
                           ▼
                       model.xpu
                           │
                           ▼
                  Backend Validation
                           │
                           ▼
                    Application Test
                           │
                           ▼
                       Benchmark
                           │
                           ▼
                     Artifact Build
                           │
                           ▼
                     Docker Export
                           │
                           ▼
              yolov10_xxx_dockerimg_v1.0.zip
```

---

# 四十、Docker Exporter 这一部分其实已经做得比较好了

你当前：

```text
DockerExporter
RuntimePackager
TemplateRenderer
PackageManifest
```

这一层我评价比较高。

尤其：

```text
runtime.tgz
```

与模型解耦，这是正确方向。

我建议主要做**安全性和可复现性增强**，不要重写。

---

# 四十一、Docker 包必须禁止 degraded artifact

最终：

```python
DockerExporter.export(...)
```

开头直接：

```python
if artifact.degraded:
    raise PackageError(
        "degraded artifact 不允许生成正式 Docker 交付包"
    )
```

开发测试如果需要：

```bash
--dev-package
```

才允许。

---

# 四十二、你现在的 Docker 默认日志配置存在安全问题

`kunlun.yaml` 里：

```yaml
minio_access_key: "minioadmin"
minio_secret_key: "minioadmin123"
```

这个不应该进入正式交付模板。

特别是：

```text
Dockerfile
install.conf
readme.txt
manifest.json
```

都可能泄露。

应该改成：

```yaml
minio_host: ""
minio_port: ""
minio_bucket: ""
minio_access_key: ""
minio_secret_key: ""
```

现场：

```bash
docker run \
  -e MINIO_ACCESS_KEY=...
  -e MINIO_SECRET_KEY=...
```

或者使用：

```text
install.conf
```

但也不要提交默认密码。

---

# 四十三、Runtime 的 CPU fallback 也应该更严格

当前：

```text
没有 /dev/xpuctrl
       ↓
CPU
```

对于正式 XPU 模型：

```text
model.xpu
```

我建议：

```text
DEVICE=auto
 ↓
没有 XPU
 ↓
启动失败
```

不要自动 CPU。

只有：

```bash
DEVICE=cpu
```

并且模型本身是：

```text
ONNX / stub
```

才能 CPU。

否则：

> 客户部署失败时，系统悄悄 CPU 跑起来，性能突然掉 100 倍，排查非常困难。

---

# 四十四、最终 Runtime 规则

```text
artifact_format = xpu
        │
        ├── DEVICE=xpu
        │      ↓
        │    XPU
        │
        ├── DEVICE=auto
        │      ↓
        │    XPU exists?
        │      ├── YES → XPU
        │      └── NO  → FAIL
        │
        └── DEVICE=cpu
               ↓
             FAIL
```

除非：

```text
artifact_format = onnx
```

才：

```text
DEVICE=cpu
```

---

# 四十五、当前测试体系需要升级

现在测试：

```text
26 tests
```

由于环境没有 `onnx`，实际：

```text
24 ERROR
2 OK
```

这里有一个问题。

虽然测试环境缺少 ONNX 是环境问题，但：

> **测试套件不应该在缺失可选依赖时让大量测试全部 ERROR。**

应该：

```python
@unittest.skipUnless(
    onnx_available(),
    "requires onnx"
)
```

或者把依赖分层：

```text
tests/
├── unit/
│   ├── config
│   ├── registry
│   ├── manifest
│   └── runtime
│
├── graph/
│   ├── optimizer
│   └── rewrite
│
├── integration/
│   └── onnx
│
└── e2e/
    └── kunlun
```

---

# 四十六、测试必须增加四层

## Unit

```text
Config
Registry
Manifest
Graph
Rewrite
```

---

## ONNX Integration

```text
PyTorch
 ↓
ONNX
 ↓
ONNX Runtime
```

---

## Kunlun Integration

需要真实 XPU：

```text
ONNX
 ↓
Kunlun Compiler
 ↓
XPU
```

---

## E2E

真正：

```text
best.pt
 ↓
Docker ZIP
 ↓
docker build
 ↓
docker run
 ↓
curl /health
 ↓
curl /predict
```

这是最终交付的验收测试。

---

# 四十七、我建议建立 Golden Model

例如：

```text
tests/golden/
└── yolov10n/
    ├── model.pt
    ├── images/
    │   ├── bus.jpg
    │   └── zidane.jpg
    │
    ├── reference/
    │   ├── outputs.npy
    │   └── detections.json
    │
    └── expected.yaml
```

每次代码改动：

```text
YOLOv10
 ↓
ONNX
 ↓
Optimize
 ↓
XPU
```

都自动跑。

这样你以后修改：

```text
SiLU
Resize
Conv-BN
NMS
```

不会把 YOLO 搞坏。

---

# 四十八、你当前的 `images/yolov5-test-images` 也应该升级

现在：

```text
images/
└── yolov5-test-images
```

这个目录名已经和 V1 的 YOLOv10 不匹配。

改成：

```text
tests/assets/
└── detection/
    ├── bus.jpg
    └── zidane.jpg
```

不要绑定具体 YOLO 版本。

---

# 四十九、模型配置也需要收敛

现在你已经有：

```text
yolov5.yaml
yolov8.yaml
yolov9.yaml
yolov10.yaml
yolov11.yaml
yolov12.yaml
yolov26.yaml
```

但当前阶段**不要真的宣称支持这么多模型**。

建议：

```text
V1.0
    yolov10

V1.1
    yolov8
    yolov9
    yolov11

V2.0
    PaddleOCR
```

没有经过：

```text
export
compile
accuracy
benchmark
docker
```

完整验证的模型，不应该进入：

```text
available_model_types()
```

或者至少标记：

```yaml
status: experimental
```

---

# 五十、模型 Registry 应该支持生命周期状态

例如：

```python
@dataclass
class ModelSupport:

    model_type: str

    status: str
    # stable / experimental / deprecated

    framework: str

    min_version: str
```

CLI：

```bash
xpu-converter list-models
```

输出：

```text
Model       Framework    Status
-----------------------------------
yolov10     PyTorch      STABLE
yolov8      PyTorch      EXPERIMENTAL
yolov9      PyTorch      EXPERIMENTAL
yolov11     PyTorch      EXPERIMENTAL
ppocr       Paddle       PLANNED
```

---

# 五十一、我建议你把 V1 的范围重新锁死

现在 README 写：

```text
PyTorch / YOLOv10 / Detection
```

这是正确的。

但代码实际上已经偷偷扩展到了：

```text
YOLOv5
YOLOv8
YOLOv9
YOLO11
YOLO12
YOLO26
PaddleOCR
PaddleDetection
```

这会导致维护失控。

建议 V1：

```text
Framework:
    PyTorch

Model:
    YOLOv10

Task:
    Detection

Shape:
    Static

Batch:
    1

Precision:
    FP32 / FP16

Backend:
    Kunlun

Postprocess:
    CPU

Deployment:
    Docker
```

其他全部：

```text
experimental
```

---

# 五十二、最终我建议把版本规划成这样

## V1.0

完成：

```text
YOLOv10
 ↓
ONNX
 ↓
Normalize
 ↓
Capability Check
 ↓
Kunlun
 ↓
model.xpu
 ↓
Accuracy
 ↓
Docker
```

---

## V1.1

增加：

```text
YOLOv8
YOLOv9
YOLO11
```

并完善：

```text
FP16
multiple resolutions
multi-device
```

---

## V1.2

增加：

```text
INT8
calibration
```

---

## V2.0

增加：

```text
Paddle
 ↓
PaddleOCR
 ↓
PaddleDetection
```

---

# 五十三、我给当前项目的修改优先级

这是最实际的执行顺序。

## P0：必须先改

### 1

```text
禁止默认 degraded → Docker
```

### 2

```text
真实 Kunlun SDK Adapter 接口
```

### 3

```text
Capability Matrix
```

### 4

```text
Final Graph Operator Check
```

### 5

```text
YOLOv10 Output Contract
```

### 6

```text
YOLOv10 checkpoint loading
```

### 7

```text
真实 XPU Runtime Adapter
```

### 8

```text
三层 Accuracy Validation
```

---

# 五十四、P1

然后：

```text
PassManager
Rollback
Artifact metadata
Environment fingerprint
Source SHA256
Benchmark hardware fingerprint
Runtime CPU fallback policy
Config split
Manifest split
```

---

# 五十五、P2

最后：

```text
Model lifecycle
Capability database
CI model zoo
Docker E2E
INT8
Paddle
OCR
```

---

# 五十六、我建议你不要马上让我“重写所有代码”

最合理的开发方式是分 **4 个 Commit / Sprint**。

---

## Sprint 1：把“假转换器”变成“严谨转换器”

修改：

```text
pipeline.py
backend/base.py
backend/kunlun/compiler.py
backend/kunlun/operator_registry.py
backend/kunlun/config.py
```

目标：

```text
没有真实 SDK
     ↓
转换失败
```

而不是：

```text
生成 fake model.xpu
```

同时增加：

```text
Capability
Preflight
Final Operator Check
```

---

# 五十七、Sprint 2：真正把 YOLOv10 打通

重点：

```text
frontend/pytorch/base.py
frontend/pytorch/yolov10.py
contract/output.py
runtime/detection
```

做到：

```text
真实 best.pt
 ↓
正确恢复 YOLOv10
 ↓
ONNX
 ↓
输出 Contract
 ↓
Raw detection
```

这里是整个项目最核心的算法部分。

---

# 五十八、Sprint 3：接真实昆仑 SDK

你把实际环境给我：

```text
Kunlun XPU 型号
SDK
XTCL
Driver
Firmware
Python 包
编译示例
Runtime 示例
```

然后实现：

```text
backend/kunlun/adapters/<actual_sdk>.py
```

把：

```text
ONNX
 ↓
model.xpu
```

真正落地。

---

# 五十九、Sprint 4：Docker 交付

最后：

```text
model.xpu
 +
runtime
 +
config
 +
metadata
 +
Dockerfile
 +
manifest
 ↓
yolov10_xxx_dockerimg_v1.0.zip
```

并做真正：

```text
docker build
docker run
/health
/predict
```

E2E。

---

# 六十、最终验收标准

我建议不要以：

> “代码能运行”

作为 V1 完成标准。

而是以这一条作为标准：

```text
客户 best.pt
       │
       ▼
xpu-converter build
       │
       ▼
yolov10_xxx_dockerimg_v1.0.zip
       │
       ▼
拷贝到昆仑服务器
       │
       ▼
sh build.sh
       │
       ▼
Docker Container
       │
       ├── /health       → OK
       │
       └── /predict      → 正确检测结果
```

同时：

```text
PyTorch
   ≈
ONNX
   ≈
XPU
```

满足预先定义的误差阈值，并且：

```text
XPU latency
XPU throughput
```

是真实硬件数据。

---

# 六十一、最后给你一个我认为最重要的架构调整

你现在实际上是：

```text
Frontend
   ↓
ONNX
   ↓
Optimizer
   ↓
Backend
```

我建议最终升级成：

```text
                     Model Package
                           │
                           ▼
                    ┌──────────────┐
                    │ Model Adapter│
                    └──────┬───────┘
                           │
                           ▼
                    Model Contract
                           │
                           ▼
                       ONNX IR
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
        Graph Passes              Capability DB
              │                         │
              └────────────┬────────────┘
                           ▼
                     Final Graph
                           │
                           ▼
                       Preflight
                           │
                    ┌──────┴──────┐
                    │             │
                   FAIL          PASS
                                  │
                                  ▼
                         Kunlun Compiler
                                  │
                                  ▼
                            Artifact
                                  │
                   ┌──────────────┼──────────────┐
                   ▼              ▼              ▼
               Validation     Benchmark       Metadata
                   │              │              │
                   └──────────────┼──────────────┘
                                  ▼
                           Docker Exporter
                                  │
                                  ▼
                   yolov10_xxx_dockerimg_v1.0.zip
```

**这个版本我认为就已经可以作为正式项目的技术基线。**

---

## 现在这份代码我建议不要推倒

目前的：

```text
frontend
ir
optimizer
rewrite
backend
validator
exporter
runtime
```

**全部可以保留。**

真正需要做的是把三个“危险的假设”拿掉：

```text
① 假设有 Kunlun SDK
② 假设 ONNX 算子白名单就代表 XPU 支持
③ 假设 YOLO 输出格式可以靠 shape 猜
```

然后建立三个核心机制：

```text
Capability
Contract
Artifact
```

这三个东西一旦建立起来，后面的 **YOLOv8 / YOLOv9 / YOLO11 / PaddleOCR / PaddleDetection** 就会变成“增加 Adapter + Capability + Contract”，而不是每增加一个模型就重新写一套转换器。

另外，我这次实际运行了项目测试：当前环境没有安装 `onnx`，所以 26 个测试中有 24 个因依赖缺失直接 ERROR；Python 源码本身可以通过 `py_compile`。这意味着**当前测试失败首先是测试环境依赖问题，但也暴露出测试依赖分层不够完善**。下一版应该把 `onnx/onnxruntime` 集成测试和纯 Python unit test 分开。

**下一步最合适的工作不是继续写泛泛的设计文档，而是直接进入代码级改造：我可以按照上面 P0 → P1 的顺序，基于你这份 `xpu-model-converter.zip`，逐个文件给出“现有代码 → 修改后代码”的具体方案，第一批先改 `pipeline.py + backend/base.py + Kunlun Backend + operator capability + YOLOv10 Adapter/Output Contract`，并给出可以直接替换进仓库的代码。**
