我:直接针对 fd9a58b 的完整源码做一次“验收级代码审计”：逐目录检查 pipeline → frontend → IR → optimizer → rewrite → Kunlun backend → validator → benchmark → exporter → runtime，最后给你一张 P0/P1/P2 缺陷清单（文件、函数、问题、严重级别、修改方案、验收标准

ChatCPT:
好。我已经按你指定的 **`fd9a58b742401d6ad68d862da0fb2c1f55986f56`** 做了一轮“验收级”源码审计，重点不是再看功能列表，而是检查：

`pipeline → frontend → IR → optimizer → rewrite → Kunlun backend → validator → benchmark → exporter → runtime`

以及 **“能否真的作为 YOLOv10 + 昆仑 XPU 交付链路”**。

先给结论：

> **这版已经具备比较完整的软件工程骨架，但目前仍有 3 个 P0 级验收问题，其中最关键的是：Validator/Benchmark 存在把 CPU 执行误当成目标 XPU 执行的风险；Kunlun backend 当前的 `paddle` 路径本质上是“ONNX → Paddle 静态图 + 运行时 XPU 编译”，还不能等同于已经完成真正的 XPU 编译产物链路。**

所以我现在不会建议你继续扩模型，而是应该**优先把下面的 P0 问题清掉，然后做一次真实 YOLOv10 + 昆仑卡验收**。

---

# 一、总体审计结论

我把当前状态重新评级：

| 模块                  | 当前评级  | 结论                                           |
| ------------------- | ----- | -------------------------------------------- |
| Pipeline            | 🟢 B+ | 主流程已经完整                                      |
| Frontend            | 🟢 B+ | YOLO 体系覆盖很广                                  |
| IR                  | 🟢 B  | 架构合理                                         |
| Optimizer           | 🟢 B  | 基础能力齐全                                       |
| Rewrite             | 🟢 B  | 有实际用途                                        |
| Operator Capability | 🟢 B+ | 已有编译前硬门禁                                     |
| Validator           | 🟡 B  | 功能完整，但存在真实设备语义问题                             |
| Benchmark           | 🟡 C+ | 有硬件指纹，但仍有 CPU/XPU 语义漏洞                       |
| Kunlun Backend      | 🟡 C+ | 架构好，但“真实编译”尚未完全落地                            |
| Runtime             | 🟡 C+ | Paddle XPU 路径可用性较高，但 Toolkit runtime 仍是 TODO |
| Exporter            | 🟢 B+ | 已经比较成熟                                       |
| Docker              | 🟢 B  | 包结构已经比较完整                                    |
| E2E                 | 🔴 C  | 缺少真正 XPU acceptance gate                     |
| V1 YOLOv10          | 🟡 C+ | 代码链路基本齐，但还没形成强验收闭环                           |

---

# 二、P0-1：最大的 bug —— Validator 可能把 CPU 当成 XPU

这是我这轮审计发现的**最重要问题**。

问题出在：

```text
xpu_converter/backend/kunlun/runtime.py
```

里面的 `PaddleXpuRuntimeSession`。

当前逻辑：

```python
DEVICE=auto
    ↓
_xpu_available()
    ↓
有 XPU → xpu
无 XPU → cpu
```

源码明确如此：

> `DEVICE=auto : 仅当本机确实有可用 XPU 卡时才用 XPU, 否则退 CPU`。 

也就是说：

```text
Paddle + X2Paddle
        ↓
Paddle adapter available
        ↓
compile 成功
        ↓
Validator create_runtime()
        ↓
DEVICE=auto
        ↓
没有 XPU
        ↓
CPU
        ↓
Accuracy PASS
```

这就有一个非常危险的问题：

### 你实际上验证的是

```text
ONNX CPU
    vs
Paddle CPU
```

而不是：

```text
ONNX CPU
    vs
Paddle XPU
```

---

## 为什么这个问题是 P0？

因为 Pipeline 的第 8 步：

```python
target = self.backend.create_runtime(artifact)
```

然后：

```python
AccuracyValidator(...).validate(
    reference,
    target,
    ...
)
```

没有检查：

```text
target 实际 device == XPU
```

源码可以直接看到这一点。 

所以只要 Paddle 能装，XPU 没插上：

```text
Accuracy Validation OK
```

完全可能出现。

这对你的项目是非常危险的，因为 CLI 又宣称：

```text
[07] Kunlun Compile OK
[08] Accuracy Validation OK
[09] Performance Benchmark OK
```

这样很容易让人误以为：

> “昆仑 XPU 已经验证通过。”

实际上可能只是 CPU。

---

# 三、P0-2：Benchmark 也存在同一个语义漏洞

这个问题甚至更加严重。

Benchmark 自己已经非常努力地防止：

> CPU benchmark 被误认为 XPU benchmark。

例如它明确写了：

```text
仿真/占位后端不得输出任何时延数字
```

以及：

```text
缺芯片指纹 → NOT_AVAILABLE
```

这是正确的设计。 

但是它判断“是不是模拟”的主要依据是：

```python
_is_simulated(session)
```

而 `PaddleXpuRuntimeSession` 的：

```python
backend_name = "paddle-xpu"
```

即使 `_resolve_device()` 最后选择了：

```text
cpu
```

它仍然是：

```text
paddle-xpu
```

而不是：

```text
onnxruntime
```

于是可能出现：

```text
PaddleXpuRuntimeSession
        ↓
实际 device = CPU
        ↓
BenchmarkRunner
        ↓
backend = paddle-xpu
        ↓
hardware.chip = auto
        ↓
credible = True
```

---

# 四、这里还有一个更隐蔽的问题：`chip="auto"`

`HardwareCapability` 默认：

```python
chip = "auto"
```

而 Benchmark 的可信条件是：

```python
return self.available and bool(self.chip)
```

注意：

```python
bool("auto") == True
```

所以：

```text
chip = "auto"
```

实际上会被当成：

> 有芯片标识。

这显然不够严格。

源码就是这样定义的。 

---

# 五、所以 P0-1/P0-2 应该合并修复

我建议你不要打补丁式修改，而是建立一个明确的：

# Execution Provenance

即：

```text
RuntimeSession
    │
    ├── backend = paddle-xpu
    ├── requested_device = auto
    ├── actual_device = xpu
    ├── device_id = 0
    ├── device_available = true
    └── execution_mode = hardware
```

或者：

```text
actual_device = cpu
execution_mode = fallback_cpu
```

---

## 然后 Accuracy Validator 必须增加硬门禁

V1 要求：

```text
target.actual_device == "xpu"
```

否则：

```text
Accuracy status = NOT_AVAILABLE
```

而不是：

```text
PASS
```

---

## Benchmark 也一样

必须：

```text
execution_mode == hardware
AND
actual_device == xpu
AND
hardware.device_available == true
AND
chip != auto
```

才能：

```text
status = OK
available = true
credible = true
```

否则：

```text
NOT_AVAILABLE
```

---

# 六、P0-3：当前所谓“Kunlun Compile”需要重新定义

这是第二个非常重要的问题。

现在：

```text
PaddleXpuSdkAdapter.compile()
```

实际上做的是：

```text
ONNX
 ↓
X2Paddle
 ↓
model.pdmodel
model.pdiparams
```

源码明确说明：

> “昆仑 XPU kernel 由 Paddle Inference 在加载时编译；交付产物为设备无关静态图。” 

也就是说：

# 当前的 compile 不是传统意义上的：

```text
ONNX
 ↓
XTCL / XPU compiler
 ↓
model.xpu
```

而是：

```text
ONNX
 ↓
X2Paddle
 ↓
Paddle Static Graph
 ↓
Runtime load
 ↓
XPU kernel compilation
 ↓
XPU execution
```

这两者不是一回事。

---

# 七、这件事情本身不一定是错的

这里我要特别区分。

如果你最终的交付要求是：

> **“使用 Paddle Inference + paddlepaddle-xpu 在昆仑芯上部署”**

那么这个方案完全可以成立。

实际上它有一个优点：

```text
converter
    ↓
Paddle static graph
    ↓
客户 XPU Runtime
```

而不是把某个 SDK 版本硬编码进 converter。

这是合理的。

---

但是如果你的原始项目要求严格是：

> `.pt → ONNX → 昆仑编译器 → model.xpu`

那么目前：

**还没有完成。**

---

# 八、因此建议把 Backend 的 artifact model 重新明确

目前代码已经开始支持：

```text
artifact_format:
    paddle
    xpu
    stub
```

这是好的。

但 V1 应该明确：

### 路线 A

```text
artifact_format = paddle
runtime = paddle-xpu
```

表示：

> Paddle XPU Deployment Artifact

### 路线 B

```text
artifact_format = xpu
runtime = kunlun-xpu
```

表示：

> Native Kunlun Compiled Artifact

不要再把两者统一称为：

```text
Kunlun Compile
```

否则验收时容易产生歧义。

---

# 九、当前 `xpuctl` 实际还是 TODO

这一点源码也非常诚实。

```python
class XpuToolkitSdkAdapter
```

目前是：

> 昆仑 XPU Toolkit 适配器（预留）

而真正 compile 的 SDK module 需要外部注入。

源码明确写了：

> “昆仑 SDK 为 C/C++ 库，无官方 Python API；本适配器仅在显式配置真实 `sdk_module` 时才可用。” 

所以：

```text
xpuctl
```

目前不能算已经完成。

而 Runtime：

```python
XpuToolkitRuntimeSession
```

也明确存在：

```text
TODO(SDK)
```

源码甚至明确说：

> 拿到昆仑 SDK 后补全真实运行时调用。 

因此这一块目前：

**架构完成，具体 SDK implementation 未完成。**

---

# 十、P1：Pipeline 的“10 步 OK”语义需要继续收紧

Pipeline 当前结构其实很好。

它明确：

```text
01 inspect
02 load
03 export ONNX
04 graph check
05 operator analysis
06 optimize
07 compile
08 validate
09 benchmark
10 package
```

并且已经实现异常传播，不会简单吞异常。 

这是一个很好的基础。

但是：

```text
OK
```

现在表达的仍然有点宽泛。

比如：

```text
[07] Kunlun Compile OK
```

实际上可能意味着：

```text
paddle static graph created
```

而不是：

```text
XPU binary successfully compiled
```

所以建议最终改成：

```text
[07] Backend Artifact Build OK
     adapter=paddle
     format=paddle_static_graph
     execution=xpu_runtime
```

如果是 native：

```text
[07] Kunlun Native Compile OK
     adapter=xpuctl
     format=xpu
```

这样验收报告不会产生歧义。

---

# 十一、P1：Accuracy Validator 现在还缺少“真实后端资格”

现在 Validator 本身的功能已经很不错。

你已经有：

```text
Graph level
Backend level
Application level
```

Pipeline 里也已经明确：

```text
original ONNX
      ↓
optimized ONNX
      ↓
target backend
```

三层验证。 

但是现在：

```text
target backend
```

应该再带：

```text
execution provenance
```

最终 AccuracyReport 至少应该包含：

```json
{
  "reference": "onnxruntime-cpu",
  "target": "paddle-xpu",
  "requested_device": "xpu",
  "actual_device": "xpu",
  "device_id": 0,
  "hardware": {
    "chip": "...",
    "sdk": "...",
    "driver": "...",
    "firmware": "..."
  },
  "status": "PASS"
}
```

而不能只有：

```text
PASS
```

---

# 十二、P1：Benchmark 的计时方式还需要针对 XPU 做同步

现在 benchmark 是：

```python
start = time.perf_counter()
session.run(feeds)
timings.append(time.perf_counter() - start)
```

源码就是这个模式。 

对于真正异步的 accelerator runtime，这个设计存在潜在问题：

```text
CPU start
    ↓
enqueue kernel
    ↓
session.run() return
    ↓
CPU stop
```

如果底层是异步执行：

```text
测到的 ≠ kernel 完整执行时间
```

---

所以 XPU benchmark 应该由 RuntimeSession 提供：

```python
synchronize()
```

或者：

```python
run_sync()
```

然后：

```text
warmup
 ↓
synchronize
 ↓
start
 ↓
run
 ↓
synchronize
 ↓
stop
```

否则 P95/P99/FPS 都可能不准确。

---

# 十三、P1：Benchmark 当前只有 latency，没有真正的资源指标

目前：

```text
mean
p90
throughput
```

已经有了。

但生产验收至少建议增加：

```text
P50
P90
P95
P99
min
max
std
FPS
device memory
power（如果 SDK 可取）
```

尤其是：

```text
P50 / P95 / P99
```

比单纯：

```text
average
```

更有意义。

---

# 十四、P1：`throughput_fps = 1000 / mean_latency`

现在：

```python
throughput_fps = 1000.0 / mean_latency
```

对于：

```text
batch = 1
single stream
```

成立。

但以后如果：

```text
batch > 1
```

就不严谨。

应该：

```text
FPS = batch / latency_seconds
```

并且最好区分：

```text
latency_ms
samples_per_second
images_per_second
```

---

# 十五、P1：Exporter 已经相当不错，但 artifact 命名仍有概念混乱

Exporter 已经做得比较完整：

```text
Dockerfile
build.sh
readme.txt
packages/
算法目录/
runtime.yaml
confidence.json
metadata.json
artifact.json
model.yaml
start.sh
```

而且对于 Paddle：

```text
.pdmodel
.pdiparams
```

也已经考虑到了。 

这是这次代码里我比较认可的一部分。

但是 README 仍然大量使用：

```text
model.xpu
```

而真实 Paddle artifact 是：

```text
model.pdmodel
model.pdiparams
```

README 的抽象和实际实现需要统一。

---

# 十六、P1：Runtime 的 CPU 回退策略已经修得比较好

这一点反而值得表扬。

`runtime/common/nwai_backend.py` 已经明确：

> 真实 XPU 产物在 `DEVICE=auto` 且无卡时直接启动失败，不允许静默 CPU fallback。 

这是正确的。

即：

```text
artifact_format = paddle
DEVICE=auto
       ↓
no XPU
       ↓
FAIL
```

而不是：

```text
XPU → CPU
```

这一点说明你已经意识到了“CPU fallback 会污染生产验收”的问题。

**但遗憾的是，这个原则目前只在 Runtime 层执行，Validator/Benchmark 层没有完全同步。**

这就是我把它列为 P0 的原因。

---

# 十七、P1：`probe_kunlun()` 目前的硬件身份还不够强

现在：

```python
chip = target_chip
```

默认可能就是：

```text
auto
```

而：

```text
device_count
device_available
sdk_version
paddle_version
```

可以探测。

但是：

```text
具体芯片型号
driver
firmware
compiler
```

并没有形成可靠的真实 fingerprint。

源码里甚至：

```text
driver_version = ""
firmware_version = ""
compiler_version = ""
```

默认空。 

所以目前：

```text
hardware fingerprint
```

还不能达到生产 benchmark 数据库级别。

---

# 十八、P1：Capability Matrix 已经形成正确方向

这个部分我反而认为是当前架构里比较好的地方。

Pipeline 在真正 compile 前：

```text
优化
 ↓
rewrite
 ↓
final capability check
 ↓
unsupported → stop
 ↓
compile
```

源码明确实现了：

> 对最终优化 + 改写后的图重新做 Capability Matrix，不允许 UNSUPPORTED 图进入编译。 

这是正确顺序：

```text
Raw ONNX
 ↓
Optimize
 ↓
Rewrite
 ↓
Final Graph
 ↓
Capability Check
 ↓
Compile
```

而不是：

```text
Raw ONNX
 ↓
Capability
 ↓
Optimize
 ↓
Rewrite
 ↓
Compile
```

这一点保留。

---

# 十九、P2：目前最不需要继续投入的是“更多模型”

现在 README 已经覆盖：

```text
YOLOv5
YOLOv6
YOLOv7
YOLOv8
YOLOv9
YOLOv10
YOLOv11
YOLOv12
YOLOv26
```

甚至：

```text
seg
pose
obb
cls
depth
sem
```

而官方状态已经把：

```text
YOLOv10 detection = stable
其他 = experimental
Paddle = planned
```

分开。 

这个策略是对的。

所以：

### 暂时不要再加

```text
YOLOv13
YOLOv14
更多 task
更多 decoder
```

现在这些工作的 ROI 很低。

---

# 二十、我给你的正式 P0/P1/P2 清单

## 🔴 P0 —— 必须先修

### P0-1：禁止 Validator 自动 CPU fallback

**文件：**

```text
xpu_converter/backend/kunlun/runtime.py
xpu_converter/pipeline.py
xpu_converter/validator/accuracy.py
```

**问题：**

```text
target = PaddleXpuRuntimeSession
DEVICE=auto
没有 XPU
→ CPU
→ Accuracy PASS
```

**修改：**

增加：

```python
actual_device
execution_mode
device_id
```

Accuracy 对 V1 要求：

```text
actual_device == xpu
```

否则：

```text
NOT_AVAILABLE
```

---

### P0-2：禁止 Benchmark 把 CPU timing 当 XPU timing

**文件：**

```text
validator/benchmark.py
backend/kunlun/runtime.py
capability/hardware.py
```

必须满足：

```text
actual_device == xpu
device_available == true
chip != auto
driver != ""
sdk != ""
```

才能：

```text
available = true
credible = true
status = OK
```

否则：

```text
NOT_AVAILABLE
```

---

### P0-3：明确 `paddle` artifact 和 native `xpu` artifact

当前：

```text
paddle
```

实际上是：

```text
ONNX → Paddle Static Graph
```

不是：

```text
ONNX → Native XPU binary
```

所以必须在产品模型中明确：

```text
artifact_format = paddle_static_graph
```

和：

```text
artifact_format = xpu_native
```

不能混用。

---

# 二十一、P1 —— P0 修完之后马上做

### P1-1

Benchmark 增加：

```text
P50
P95
P99
std
min
max
```

### P1-2

增加：

```python
session.synchronize()
```

保证 XPU 异步执行计时正确。

### P1-3

Hardware fingerprint 增加：

```text
chip
sdk
compiler
driver
firmware
runtime
paddle
device_id
```

### P1-4

AccuracyReport 增加：

```text
reference execution
target execution
actual device
hardware fingerprint
```

### P1-5

Pipeline Step 07 改名/改语义：

```text
Backend Artifact Build
```

而不是无条件：

```text
Kunlun Compile
```

### P1-6

README 与实际 artifact 统一：

```text
Paddle:
    model.pdmodel
    model.pdiparams

Native:
    model.xpu
```

---

# 二十二、P2 —— 后面再做

```text
PaddleOCR
PaddleDetection
更多 YOLO
更多任务
自动性能数据库
多卡 benchmark
功耗
自动 regression dashboard
```

这些目前都不影响 V1 是否成功。

---

# 二十三、我建议你下一版直接做成这样的状态机

这是我现在最建议你改的核心设计。

```text
                     Conversion
                         │
                         ▼
                 Backend Resolution
                         │
            ┌────────────┴────────────┐
            │                         │
       Native XPU                 Paddle XPU
            │                         │
            ▼                         ▼
     xpu_native artifact      paddle_static_graph
            │                         │
            └────────────┬────────────┘
                         ▼
                  Runtime Load
                         │
                         ▼
                  Device Preflight
                         │
               ┌─────────┴─────────┐
               │                   │
             XPU                  CPU
               │                   │
               ▼                   ▼
           ACCEPT              REJECT
               │
               ▼
          Accuracy Test
               │
               ▼
         Application mAP
               │
               ▼
         Synchronize
               │
               ▼
          Benchmark
               │
               ▼
        Hardware Fingerprint
               │
               ▼
          Docker Package
               │
               ▼
        Delivery Acceptance
```

关键是：

> **“实际运行设备”必须成为一等公民。**

而不是现在这种：

```text
backend = paddle-xpu
```

就默认认为：

```text
actual device = XPU
```

---

# 二十四、V1 最终验收标准，我建议现在就固定下来

最终 YOLOv10 V1 不要再看“代码有没有”。

直接看这 10 个 Gate：

| Gate | 验收条件                                             |
| ---- | ------------------------------------------------ |
| G1   | YOLOv10 `.pt` 正确识别                               |
| G2   | PyTorch → ONNX 成功                                |
| G3   | ONNX checker + shape inference PASS              |
| G4   | Final graph capability PASS                      |
| G5   | Backend artifact 真正生成                            |
| G6   | **真实 XPU runtime 加载成功**                          |
| G7   | PyTorch / ONNX / XPU 三路数值验证 PASS                 |
| G8   | Application mAP 满足阈值                             |
| G9   | XPU P50/P95/P99 benchmark 有效                     |
| G10  | Docker build + run + `/health` + `/predict` PASS |

最终只有：

```text
G1 ~ G10 = PASS
```

才能把：

```text
YOLOv10 = stable
```

真正当成“可交付”。

---

# 二十五、因此我对 fd9a58b 的最终评价

这次审计后，我会把之前的判断稍微调整：

### 软件工程完成度

**≈ 85%**

### V1 converter 代码完成度

**≈ 80%**

### Validator

**≈ 85%**

### Docker exporter

**≈ 85%**

### Kunlun backend

**≈ 60%**

### 真实 XPU runtime

**≈ 50%**

### 真实 YOLOv10 交付闭环

**≈ 55～60%**

### 整个项目距离真正 V1 Stable

**≈ 65%**

这里最大的原因不是“代码少”，而是：

> **现在已经有很多代码，但还缺最后一层“真实性约束”：必须证明运行的是 XPU，而不是 CPU；必须证明 benchmark 是 XPU 数据；必须明确 Paddle Static Graph 与 Native XPU Artifact 的区别。**

---

## 我建议你下一步不要继续加功能

直接按这个优先级：

```text
① 修 P0-1
   Accuracy 禁止 CPU fallback

② 修 P0-2
   Benchmark 禁止 CPU timing 冒充 XPU

③ 修 P0-3
   paddle artifact / native xpu artifact 语义彻底分开

④ 加 execution provenance

⑤ 加 XPU synchronize

⑥ 固定 yolov10n.pt

⑦ 在真实昆仑机器跑 G1~G10

⑧ 根据真实结果再决定哪些 frontend 可以升级 experimental → stable
```

我: 如果我想把当前的项目做成一个网站，客户可以登录这个网站，并且可以上传他需要转换的模型，执行转换路程，保存各阶段的产物，最终获取部署包。应该怎么实现？

ChatGPT:
可以，而且我建议你**不要把现有 `xpu-model-converter` 直接改造成 Web 项目**。更合理的方式是：

> **把当前 Converter 做成“转换引擎（Conversion Engine）”，在它上面增加一层多租户的 Web 控制平面（Control Plane），再用 Worker/Queue 调度实际转换任务。**

最终形成：

```text
                    ┌──────────────────────────┐
                    │       Web Browser        │
                    │  Login / Upload / Jobs   │
                    │  Pipeline / Artifacts    │
                    └────────────┬─────────────┘
                                 │ HTTPS
                                 ▼
                    ┌──────────────────────────┐
                    │       Web Frontend        │
                    │        Next.js            │
                    └────────────┬─────────────┘
                                 │ REST / WS
                                 ▼
                    ┌──────────────────────────┐
                    │       API Server          │
                    │ FastAPI / Python         │
                    │                          │
                    │ Auth / User / Project     │
                    │ Job / Artifact / Pipeline │
                    └──────┬───────────┬───────┘
                           │           │
                 PostgreSQL│           │Redis
                           │           │
                           ▼           ▼
                    ┌──────────┐   ┌──────────┐
                    │ Database │   │  Queue   │
                    └──────────┘   └────┬─────┘
                                        │
                         ┌──────────────┼──────────────┐
                         ▼              ▼              ▼
                    ┌─────────┐   ┌─────────┐   ┌─────────┐
                    │ Worker 1│   │ Worker 2│   │ Worker N│
                    │ CPU     │   │ XPU #0  │   │ XPU #1  │
                    └────┬────┘   └────┬────┘   └────┬────┘
                         │             │              │
                         └─────────────┼──────────────┘
                                       ▼
                              xpu-model-converter
                                       │
                                       ▼
                               Artifact Storage
                              MinIO / S3 / OSS
```

这个架构非常适合你现在的项目。

---

# 一、先明确：你的网站实际上是一个“模型转换 SaaS 平台”

客户看到的是：

```text
登录
  ↓
创建转换任务
  ↓
上传 .pt / .onnx / ...
  ↓
选择目标平台
  ↓
选择模型/任务
  ↓
开始转换
  ↓
实时查看 Pipeline
  ↓
查看每个阶段产物
  ↓
下载最终部署包
```

而后台实际上是在执行：

```text
Upload
 ↓
Inspect
 ↓
Load
 ↓
Export ONNX
 ↓
Graph Check
 ↓
Capability Analysis
 ↓
Optimize
 ↓
Rewrite
 ↓
Compile
 ↓
Accuracy Validation
 ↓
Benchmark
 ↓
Package
```

这与你现在已经建立的 10-step Pipeline **高度吻合**。

所以你的现有项目其实非常适合作为后端转换 Engine。

---

# 二、我建议把系统拆成 4 层

不要让 Web API 直接调用：

```python
pipeline.run(...)
```

然后 HTTP 请求一直阻塞。

应该拆成：

```text
┌──────────────────────────────────────┐
│             Control Plane            │
│                                      │
│ Web / API / Auth / Job / DB          │
└──────────────────┬───────────────────┘
                   │
                   │ Job
                   ▼
┌──────────────────────────────────────┐
│            Orchestration             │
│                                      │
│ Queue / Scheduler / State Machine    │
└──────────────────┬───────────────────┘
                   │
                   │ Task
                   ▼
┌──────────────────────────────────────┐
│          Conversion Engine           │
│                                      │
│ xpu-model-converter                  │
│ Pipeline / Frontend / IR / Backend   │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│              Storage                 │
│                                      │
│ Model / ONNX / Logs / Reports / ZIP  │
└──────────────────────────────────────┘
```

---

# 三、第一件事：把现在的项目改造成 Engine

这是整个架构最重要的一步。

现在你的项目大概是：

```text
xpu-model-converter/
├── xpu_converter/
│   ├── frontend/
│   ├── backend/
│   ├── validator/
│   ├── optimizer/
│   ├── exporter/
│   └── pipeline.py
└── cli.py
```

不要把 Web 代码塞进去。

改成：

```text
xpu-model-converter/
├── xpu_converter/
│   ├── frontend/
│   ├── backend/
│   ├── validator/
│   ├── optimizer/
│   ├── exporter/
│   │
│   ├── pipeline/
│   │   ├── runner.py
│   │   ├── context.py
│   │   ├── stage.py
│   │   └── events.py
│   │
│   └── engine/
│       ├── converter.py
│       ├── job.py
│       └── artifact.py
│
├── cli/
│
├── tests/
│
└── pyproject.toml
```

核心目标：

```python
result = converter.convert(request)
```

CLI 和 Web 都调用这个 Engine。

---

# 四、一定要把 Pipeline 做成“可持久化状态机”

这是你现在项目要做 Web 化时最重要的改造。

现在 Pipeline 是：

```text
Step 1
 ↓
Step 2
 ↓
Step 3
 ↓
...
```

Web 系统需要的是：

```text
Job
 │
 ├── Stage 01 INSPECT
 │       status = SUCCESS
 │       artifact = metadata.json
 │
 ├── Stage 02 LOAD
 │       status = SUCCESS
 │
 ├── Stage 03 ONNX_EXPORT
 │       status = SUCCESS
 │       artifact = model.onnx
 │
 ├── Stage 04 GRAPH_CHECK
 │       status = SUCCESS
 │       artifact = graph_report.json
 │
 ├── Stage 05 CAPABILITY
 │       status = SUCCESS
 │       artifact = capability.json
 │
 ├── Stage 06 OPTIMIZE
 │       status = SUCCESS
 │       artifact = optimized.onnx
 │
 ├── Stage 07 COMPILE
 │       status = RUNNING
 │
 ├── Stage 08 VALIDATE
 │
 ├── Stage 09 BENCHMARK
 │
 └── Stage 10 PACKAGE
```

也就是说：

> **每一个 Stage 都必须有独立状态。**

---

# 五、建议定义 Job State Machine

例如：

```text
CREATED
   ↓
QUEUED
   ↓
RUNNING
   ↓
SUCCESS
```

失败：

```text
RUNNING
   ↓
FAILED
```

取消：

```text
RUNNING
   ↓
CANCEL_REQUESTED
   ↓
CANCELLED
```

整个 Job：

```text
CREATED
QUEUED
RUNNING
SUCCESS
FAILED
CANCELLED
```

Stage：

```text
PENDING
RUNNING
SUCCESS
FAILED
SKIPPED
```

---

# 六、数据库应该这样设计

我推荐 PostgreSQL。

核心表：

```text
users
projects
models
conversion_jobs
job_stages
artifacts
workers
audit_logs
```

---

## users

```text
users
------
id
email
password_hash
status
created_at
```

---

## projects

一个用户可以有多个项目：

```text
projects
--------
id
user_id
name
description
created_at
```

例如：

```text
我的 YOLO 项目
智能摄像头
边缘检测
```

---

# 七、Models

用户上传的原始模型。

```text
models
------
id
project_id
filename
storage_key
size
sha256
model_format
created_at
```

例如：

```text
yolov10n.pt
```

实际文件不放 PostgreSQL。

数据库只存：

```text
s3://models/tenant123/project456/model789/yolov10n.pt
```

---

# 八、Conversion Job

这是整个系统的核心。

```text
conversion_jobs
---------------
id
project_id
model_id

source_format
model_family
task

target_backend
target_chip
precision
input_shape
batch_size

status
current_stage

worker_id

created_at
started_at
finished_at

error_code
error_message
```

比如：

```json
{
  "model": "yolov10n.pt",
  "task": "detection",
  "target_backend": "paddle-xpu",
  "target_chip": "KUNLUN_X",
  "precision": "fp16",
  "input_shape": [1,3,640,640],
  "batch_size": 1
}
```

---

# 九、Stage 必须独立建表

```text
job_stages
----------
id
job_id

stage_name
stage_index

status

started_at
finished_at

progress

worker_id

input_artifact_id
output_artifact_id

metrics
error_code
error_message
```

例如：

```text
job = 123

stage 1
INSPECT
SUCCESS

stage 2
LOAD
SUCCESS

stage 3
ONNX_EXPORT
SUCCESS

stage 4
GRAPH_CHECK
SUCCESS

stage 5
CAPABILITY
SUCCESS

stage 6
OPTIMIZE
SUCCESS

stage 7
COMPILE
RUNNING
```

这样前端就可以实时显示。

---

# 十、Artifact 是另一个核心对象

这是你的项目非常适合 Web 化的地方。

每个阶段都产生 Artifact。

例如：

```text
Artifact
```

不是简单的：

```text
filename
```

而是：

```text
artifacts
---------
id
job_id
stage_id

artifact_type
filename

storage_backend
storage_key

size
sha256

mime_type

created_at
```

---

# 十一、最终你的网站就可以显示完整 Artifact Tree

例如：

```text
YOLOv10 Conversion #20260913-001
│
├── 01 Inspect
│   └── metadata.json
│
├── 02 Load
│   └── model_info.json
│
├── 03 ONNX Export
│   └── model.onnx
│
├── 04 Graph Check
│   └── graph_report.json
│
├── 05 Capability
│   └── capability_report.json
│
├── 06 Optimize
│   ├── optimized.onnx
│   └── optimization_report.json
│
├── 07 Compile
│   ├── model.pdmodel
│   └── model.pdiparams
│
├── 08 Validation
│   ├── accuracy.json
│   └── comparison.json
│
├── 09 Benchmark
│   └── benchmark.json
│
└── 10 Package
    └── deployment.zip
```

这就是你的网站最有价值的地方之一。

---

# 十二、文件存储不要放本机磁盘

开发环境可以：

```text
/mnt/jobs/123/
```

生产环境一定建议：

```text
MinIO
```

或者：

```text
S3
OSS
COS
OBS
```

对于你的场景，我非常推荐：

> **MinIO + PostgreSQL**

因为它适合私有化部署。

结构：

```text
MinIO
│
├── models/
│   └── tenant/
│
├── artifacts/
│   └── jobs/
│
├── logs/
│
└── packages/
```

数据库只保存：

```text
storage_key
```

---

# 十三、为什么一定需要 Redis + Worker

因为模型转换可能：

```text
30 秒
5 分钟
20 分钟
1 小时
```

HTTP 请求不能：

```text
POST /convert

等待 30 分钟

200 OK
```

正确方式：

```text
POST /jobs
        ↓
202 Accepted
        ↓
job_id = abc123
```

后台：

```text
Redis Queue
     ↓
Worker
     ↓
Pipeline
```

---

# 十四、Worker 才是真正执行转换的地方

例如：

```text
worker-cpu-01
```

负责：

```text
Inspect
ONNX
Graph
Optimize
```

而：

```text
worker-xpu-01
```

负责：

```text
Compile
Accuracy
Benchmark
```

这非常重要。

因为你的真实昆仑机器不应该承担：

```text
Web API
Database
Frontend
```

而应该只是：

```text
XPU Worker
```

---

# 十五、最终可以做成 Worker Pool

例如：

```text
                    Redis
                      │
             ┌────────┼────────┐
             │        │        │
             ▼        ▼        ▼
         CPU Queue  XPU Queue  GPU Queue
             │        │        │
             ▼        ▼        ▼
          Worker    Worker    Worker
                    XPU #0
                    XPU #1
                    XPU #2
```

以后你甚至可以：

```text
Kunlun XPU
NVIDIA GPU
Ascend NPU
```

共用同一个 Web 平台。

---

# 十六、这时候你的项目架构会发生一个非常漂亮的变化

原来的：

```text
xpu-model-converter
```

会逐渐变成：

```text
                    Model Conversion Platform
                              │
              ┌───────────────┼───────────────┐
              │               │               │
           Kunlun           NVIDIA          Ascend
              │               │               │
              ▼               ▼               ▼
        xpu-converter     trt-converter    cann-converter
```

Web 层完全不关心：

```text
Paddle
TensorRT
XTCL
CANN
```

它只知道：

```text
ConversionJob
```

---

# 十七、前端我建议直接用 Next.js

技术栈：

```text
Frontend
--------
Next.js
React
TypeScript
Tailwind
shadcn/ui
```

页面：

```text
/login
/dashboard
/projects
/projects/:id
/jobs
/jobs/:id
/jobs/:id/artifacts
/jobs/:id/logs
/settings
```

---

# 十八、Dashboard 应该长这样

用户登录之后：

```text
┌────────────────────────────────────────────────┐
│ Model Converter                         User ▼ │
├──────────────┬─────────────────────────────────┤
│ Dashboard    │                                 │
│ Projects     │  Overview                       │
│ Models       │                                 │
│ Conversions  │  ┌────────┐ ┌────────┐        │
│ Artifacts    │  │ 12     │ │ 8      │        │
│ Workers      │  │ Jobs   │ │ Success│        │
│ Settings     │  └────────┘ └────────┘        │
│              │                                 │
│              │  Recent Conversions             │
│              │                                 │
│              │  YOLOv10    XPU    SUCCESS     │
│              │  YOLOv8     XPU    RUNNING     │
│              │  YOLOv10    XPU    FAILED      │
└──────────────┴─────────────────────────────────┘
```

---

# 十九、上传页面

例如：

```text
Create Conversion

┌────────────────────────────────────────┐
│                                        │
│       Drag model here                  │
│                                        │
│       .pt .onnx .pth                   │
│                                        │
└────────────────────────────────────────┘

Model:
YOLOv10

Task:
○ Detection
○ Classification
○ Segmentation
○ Pose
○ OBB

Target:
Kunlun XPU

Precision:
○ FP32
● FP16

Input:
1 × 3 × 640 × 640

Batch:
1

                [ Start Conversion ]
```

---

# 二十、Job Detail 页面是整个产品的核心

我建议设计成：

```text
YOLOv10n → Kunlun XPU

Status: RUNNING
Progress: 63%

Pipeline

✓ 01 Inspect
✓ 02 Load
✓ 03 ONNX Export
✓ 04 Graph Check
✓ 05 Capability
✓ 06 Optimize
● 07 Compile
○ 08 Accuracy
○ 09 Benchmark
○ 10 Package
```

右侧：

```text
Stage Output

model.onnx
optimized.onnx
capability.json
```

下面：

```text
Live Logs

[18:32:01] Start compile
[18:32:02] Loading Paddle XPU backend
[18:32:04] Compiling graph
...
```

---

# 二十一、实时日志不要轮询数据库

可以使用：

```text
WebSocket
```

或者：

```text
Server-Sent Events
```

我更推荐：

```text
SSE
```

因为你的场景主要是：

```text
Server → Browser
```

日志流。

例如：

```text
GET /api/jobs/{id}/events
```

服务器持续：

```text
stage_started
progress
log
artifact_created
stage_finished
job_finished
```

---

# 二十二、Pipeline 内部应该产生 Event

这是对你现有项目非常重要的一次架构升级。

现在：

```python
pipeline.run()
```

建议改成：

```python
pipeline.run(
    context,
    event_sink=event_sink
)
```

每一步：

```python
event_sink.emit(
    StageStarted(...)
)
```

然后：

```python
event_sink.emit(
    LogEvent(...)
)
```

结束：

```python
event_sink.emit(
    StageCompleted(...)
)
```

这样：

```text
CLI
Web
Test
```

都可以消费同一套 Pipeline。

---

# 二十三、建议定义统一 Event Schema

例如：

```json
{
  "job_id": "job_123",
  "stage": "compile",
  "event": "stage_progress",
  "progress": 65,
  "message": "Compiling graph",
  "timestamp": "2026-09-13T18:42:11Z"
}
```

Artifact：

```json
{
  "event": "artifact_created",
  "artifact_id": "artifact_123",
  "type": "onnx",
  "filename": "optimized.onnx"
}
```

---

# 二十四、权限系统一定要从第一天设计

因为这是客户上传模型的平台。

至少：

```text
User
Organization
Project
Job
Artifact
```

权限：

```text
User
  │
  ├── Project A
  │     ├── Model
  │     ├── Job
  │     └── Artifacts
  │
  └── Project B
```

用户绝对不能：

```text
GET /artifacts/another_user_job.zip
```

所以所有 API 都必须经过：

```text
tenant_id
project_id
ownership check
```

---

# 二十五、模型文件安全是这个系统的另一个 P0

这一点千万不要忽略。

用户上传的是：

```text
.pt
.onnx
.pth
```

这些不是普通图片。

尤其 `.pt` 可能涉及：

```text
pickle
```

因此：

> **绝对不要在 Web API 进程里直接 `torch.load()` 用户上传的文件。**

应该：

```text
Upload
 ↓
Object Storage
 ↓
Queue
 ↓
Sandbox Worker
 ↓
Model Loader
```

Worker 运行在：

```text
Docker
```

甚至更严格：

```text
Kubernetes Job
```

并且：

```text
non-root
network disabled
resource limit
filesystem isolated
timeout
```

---

# 二十六、转换 Worker 必须沙箱化

建议：

```text
CPU Worker

Docker container
    │
    ├── read-only converter image
    ├── /input
    ├── /workspace
    └── /output
```

限制：

```text
CPU
Memory
Disk
Execution time
Network
```

例如：

```text
CPU: 8
RAM: 32GB
Disk: 100GB
Timeout: 30min
```

XPU Worker：

```text
Docker
    │
    └── /dev/xpu
```

只给容器暴露 XPU。

---

# 二十七、不要让 API Server 直接拥有 XPU

错误：

```text
FastAPI
   │
   ├── Web
   ├── Database
   └── XPU
```

正确：

```text
FastAPI
   │
   └── Redis
          │
          ▼
       XPU Worker
          │
          ▼
       Kunlun XPU
```

这样以后：

```text
API server scale 3
Worker scale 10
```

完全独立。

---

# 二十八、任务调度需要考虑 XPU 独占

例如：

```text
XPU #0
busy
```

新的 Job：

```text
QUEUED
```

而不是：

```text
两个模型同时抢一张 XPU
```

Worker 可以注册：

```text
workers
-------
id
hostname
device_type
device_id
chip
status
current_job
last_heartbeat
```

例如：

```text
worker-xpu-01
KUNLUN
device=0
AVAILABLE
```

---

# 二十九、最终可以做资源调度

例如 Job：

```json
{
  "target_chip": "KUNLUN_X",
  "required_memory": 8
}
```

Scheduler：

```text
find worker
    ↓
chip match
    ↓
device available
    ↓
memory enough
    ↓
assign
```

这会让你的系统从：

> 一个 Web 上传工具

升级成：

> **模型转换计算平台。**

---

# 三十、数据库 + Object Storage + Queue 是核心三件套

我建议第一版技术栈直接确定：

```text
Frontend
Next.js + TypeScript

Backend
FastAPI + Python

Database
PostgreSQL

Queue
Redis

Task Worker
Celery / RQ / Dramatiq

Object Storage
MinIO

Realtime
SSE

Reverse Proxy
Nginx

Container
Docker

Deployment
Docker Compose
```

第一版甚至不用 Kubernetes。

---

# 三十一、为什么我推荐 FastAPI

因为你现在的 Converter 本身就是 Python。

这样：

```text
FastAPI
    │
    └── xpu_converter
```

共享：

```text
ModelRegistry
Pipeline
Backend
Validator
Exporter
```

不需要：

```text
Java API
 ↓
Python RPC
```

避免大量额外复杂度。

---

# 三十二、Repository 我建议改成 Monorepo

例如：

```text
xpu-model-platform/
│
├── apps/
│   │
│   ├── web/
│   │   └── Next.js
│   │
│   ├── api/
│   │   └── FastAPI
│   │
│   └── worker/
│       └── Worker
│
├── packages/
│   │
│   └── converter/
│       └── xpu-model-converter
│
├── deploy/
│   ├── docker-compose.yml
│   ├── nginx/
│   └── minio/
│
├── migrations/
│
└── docs/
```

也可以暂时保持两个 repo：

```text
xpu-model-converter
xpu-model-platform
```

我个人更推荐**前期两个 repo**：

```text
xpu-model-converter
        ↑
        │ Python package
        │
xpu-model-platform
```

这样 Converter 可以独立测试和发布。

---

# 三十三、API 大致应该这样设计

### Auth

```text
POST /api/auth/register
POST /api/auth/login
POST /api/auth/logout
GET  /api/auth/me
```

### Projects

```text
GET    /api/projects
POST   /api/projects
GET    /api/projects/{id}
DELETE /api/projects/{id}
```

### Models

```text
POST /api/projects/{id}/models/upload
GET  /api/projects/{id}/models
GET  /api/models/{id}
```

### Jobs

```text
POST /api/jobs
GET  /api/jobs
GET  /api/jobs/{id}
POST /api/jobs/{id}/cancel
POST /api/jobs/{id}/retry
```

### Pipeline

```text
GET /api/jobs/{id}/stages
GET /api/jobs/{id}/events
GET /api/jobs/{id}/logs
```

### Artifacts

```text
GET /api/jobs/{id}/artifacts
GET /api/artifacts/{id}
GET /api/artifacts/{id}/download
```

---

# 三十四、上传模型最好不要经过 FastAPI

如果模型：

```text
1 GB
```

不要：

```text
Browser
 ↓
FastAPI
 ↓
MinIO
```

而是：

```text
Browser
   │
   │ presigned URL
   ▼
MinIO
```

流程：

```text
POST /models/upload-url
        ↓
presigned URL
        ↓
Browser PUT
        ↓
MinIO
        ↓
POST /models/complete
```

这样 API server 不承担大文件流量。

---

# 三十五、最终下载也不要让 FastAPI 中转

应该：

```text
GET /artifacts/{id}/download
          ↓
permission check
          ↓
presigned URL
          ↓
MinIO
```

浏览器直接下载。

---

# 三十六、你的最终 Job 数据结构可以非常漂亮

例如：

```json
{
  "id": "job_20260913_001",
  "status": "success",

  "source": {
    "filename": "yolov10n.pt",
    "sha256": "..."
  },

  "target": {
    "backend": "paddle-xpu",
    "chip": "KUNLUN_X",
    "precision": "fp16"
  },

  "pipeline": {
    "current_stage": 10,
    "total_stages": 10
  },

  "accuracy": {
    "status": "pass",
    "map50": 0.97,
    "map5095": 0.91
  },

  "benchmark": {
    "status": "pass",
    "p50_ms": 8.2,
    "p95_ms": 8.9,
    "fps": 121
  },

  "package": {
    "artifact_id": "artifact_xyz",
    "filename": "yolov10-kunlun-deployment.zip"
  }
}
```

前端基本可以直接渲染。

---

# 三十七、我尤其建议你保存“每一次转换的完整 Manifest”

最终：

```text
conversion-manifest.json
```

里面记录：

```text
Source Model
SHA256
Converter Version
Git Commit
Frontend Version
Backend
Target Chip
SDK Version
Driver Version

Input Shape
Batch
Precision

ONNX
Optimized ONNX

Capability Result

Accuracy Result
Benchmark Result

Artifacts
```

例如：

```json
{
  "converter": {
    "version": "0.1.0",
    "commit": "fd9a58b"
  },

  "source": {
    "sha256": "..."
  },

  "target": {
    "backend": "paddle-xpu",
    "chip": "KUNLUN_X"
  },

  "validation": {
    "status": "PASS"
  },

  "benchmark": {
    "p50": 8.1,
    "p95": 8.8
  }
}
```

这对于后续：

```text
问题追溯
客户投诉
模型复现
版本回滚
审计
```

都非常重要。

---

# 三十八、你现在的 Converter 最应该增加一个东西

不是 Web。

而是：

# `ConversionContext`

例如：

```python
@dataclass
class ConversionContext:
    job_id: str
    workspace: Path

    source_model: Path

    artifacts_dir: Path
    logs_dir: Path
    reports_dir: Path

    config: ConversionConfig

    event_sink: EventSink
```

Pipeline 所有阶段只操作：

```text
Context
```

例如：

```python
ctx.artifacts.save(
    stage="onnx_export",
    path=onnx_path
)
```

这样 Web Worker 很容易接进去。

---

# 三十九、再加一个 ArtifactStore 抽象

不要让 Pipeline 直接：

```python
shutil.copy(...)
```

定义：

```python
class ArtifactStore:
    def put(...)
    def get(...)
    def delete(...)
    def exists(...)
```

开发：

```text
LocalArtifactStore
```

生产：

```text
S3ArtifactStore
MinioArtifactStore
```

这样你的 Converter 完全不知道：

```text
MinIO
S3
本地磁盘
```

---

# 四十、再加一个 EventSink

```python
class EventSink:
    def emit(self, event):
        ...
```

实现：

```text
ConsoleEventSink
DatabaseEventSink
RedisEventSink
```

CLI：

```text
ConsoleEventSink
```

Web：

```text
RedisEventSink
```

于是同一套 Pipeline：

```text
CLI
       ↓
Converter
       ↓
Pipeline
       ↓
EventSink
```

和：

```text
Web
 ↓
Worker
 ↓
Converter
 ↓
Pipeline
 ↓
EventSink
```

完全复用。

---

# 四十一、我建议你的开发顺序不要从“登录页面”开始

这是非常重要的。

很多人会：

```text
先做 React
先做登录
先做 Dashboard
```

最后发现 Converter 无法被可靠调度。

你应该反过来：

## Phase 1

先把 Converter Engine 化：

```text
Pipeline
Context
ArtifactStore
EventSink
```

---

## Phase 2

做 Worker：

```text
Redis
Worker
Job
```

证明：

```text
submit job
 ↓
worker
 ↓
converter
 ↓
artifact
```

---

## Phase 3

PostgreSQL：

```text
Job
Stage
Artifact
```

把状态持久化。

---

## Phase 4

FastAPI：

```text
POST /jobs
GET /jobs/:id
GET /stages
GET /artifacts
```

---

## Phase 5

实时事件：

```text
SSE
```

---

## Phase 6

MinIO：

```text
Upload
Artifact
Download
```

---

## Phase 7

Next.js：

```text
Login
Dashboard
Upload
Job Detail
Artifacts
```

---

## Phase 8

XPU Worker：

```text
Kunlun Worker
```

---

## Phase 9

安全：

```text
sandbox
quota
permission
virus scan
resource limits
```

---

# 四十二、第一版 MVP 我建议只做这一条链

不要一开始做：

```text
所有模型
所有任务
所有 XPU
多租户计费
团队
RBAC
```

先锁死：

```text
YOLOv10 Detection
       ↓
Kunlun XPU
       ↓
FP16
       ↓
Batch 1
       ↓
640×640
```

用户：

```text
登录
 ↓
上传 yolov10n.pt
 ↓
创建 Job
 ↓
Pipeline
 ↓
查看 10 个 Stage
 ↓
下载 deployment.zip
```

只要这条链跑通，你的平台架构就成立了。

---

# 四十三、第一版的最终产品形态

客户看到：

```text
┌──────────────────────────────────────────────┐
│ XPU Model Converter                          │
├───────────────┬──────────────────────────────┤
│               │                              │
│ Dashboard     │ Create Conversion            │
│ Projects      │                              │
│ Models        │ Model                        │
│ Conversions   │ [ yolov10n.pt ]              │
│ Artifacts     │                              │
│               │ Task                         │
│               │ [ Detection ▼ ]              │
│               │                              │
│               │ Target                       │
│               │ [ Kunlun XPU ▼ ]             │
│               │                              │
│               │ Precision                    │
│               │ [ FP16 ▼ ]                   │
│               │                              │
│               │ [ Start Conversion ]         │
└───────────────┴──────────────────────────────┘
```

然后：

```text
Conversion #001

✓ Inspect
✓ Load
✓ Export ONNX
✓ Graph Check
✓ Capability
✓ Optimize
● Compile
○ Accuracy
○ Benchmark
○ Package
```

最终：

```text
Conversion Successful

Accuracy
mAP50      0.972
mAP50-95   0.913

Performance
P50        8.2 ms
P95        8.9 ms
FPS        121

Artifacts

[ model.onnx ]
[ optimized.onnx ]
[ validation-report.json ]
[ benchmark-report.json ]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[ Download Deployment Package ]
```

这就是一个非常完整的第一版产品。

---

# 四十四、最关键的架构原则

我建议你以后一直坚持这 6 条：

### 1. Web ≠ Converter

```text
Web Platform
      ↓
Conversion Engine
```

---

### 2. API ≠ Worker

```text
API
 ↓
Queue
 ↓
Worker
```

---

### 3. Database ≠ File Storage

```text
PostgreSQL
    ↓
metadata

MinIO
    ↓
model/artifact
```

---

### 4. Pipeline ≠ HTTP Request

Pipeline 是：

```text
long-running job
```

不是：

```text
HTTP request
```

---

### 5. Artifact 是一等公民

每个阶段都产生：

```text
Artifact
Report
Log
Metric
```

---

### 6. XPU 是 Worker Resource

不要：

```text
Web Server + XPU
```

而是：

```text
Web Server
     ↓
Scheduler
     ↓
XPU Worker
     ↓
XPU
```

---

# 四十五、如果由我来带这个项目，我会把整体路线定成

```text
                 ┌─────────────────────┐
                 │      Web SaaS       │
                 │ Next.js             │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │       FastAPI       │
                 │ Auth / Job / API    │
                 └──────────┬──────────┘
                            │
             ┌──────────────┼──────────────┐
             ▼              ▼              ▼
        PostgreSQL        Redis          MinIO
             │              │              │
             │              ▼              │
             │        ┌───────────┐        │
             │        │ Scheduler │        │
             │        └─────┬─────┘        │
             │              │              │
             │       ┌──────┴──────┐       │
             │       ▼             ▼       │
             │   CPU Worker     XPU Worker │
             │       │             │       │
             │       └──────┬──────┘       │
             │              ▼              │
             │     xpu-model-converter     │
             │              │              │
             │       ConversionContext     │
             │              │              │
             │       ┌──────┴──────┐       │
             │       ▼             ▼       │
             │   Artifacts       Events    │
             │       │             │       │
             └───────┼─────────────┘       │
                     ▼                     │
                   MinIO ◄─────────────────┘
```

**这套架构与你现在已经完成的 Converter 非常匹配。**

而且有一个很好的战略意义：

> 你现在正在做的 `xpu-model-converter` 不再只是一个命令行转换工具，而可以成为整个“国产 AI 芯片模型转换云平台”的 **Conversion Engine**。

以后你只需要继续扩展：

```text
KunlunBackend
AscendBackend
CambriconBackend
HygonBackend
```

Web/API/Job/Artifact/Pipeline/用户体系基本都不用重新做。

**当前最值得做的下一步，不是直接写前端，而是先把现有 `pipeline.py` 重构成 `ConversionContext + Stage + ArtifactStore + EventSink` 四个核心抽象。** 这一步做好，后面的 FastAPI、Redis Worker、MinIO、Next.js 都会非常顺。