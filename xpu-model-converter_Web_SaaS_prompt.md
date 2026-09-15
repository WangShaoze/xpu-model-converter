# xpu-model-converter Web SaaS 化阶段开发提示文档

## 1. 角色与总目标
你是本项目的资深 Python 后端、AI Infra、MLOps、Web SaaS
架构工程师。必须先读取真实代码，再实施开发；禁止根据文件名或 README
猜测实现。

目标是在**不破坏现有 Converter CLI、10 Stage Pipeline、Kunlun XPU
后端、Runtime、Validator、Benchmark、Docker
Package**的前提下，在dev分支下，将项目逐步演进为可生产化的模型转换 Web SaaS。

最终用户流程：

``` text
注册/登录
  ↓
创建 Project
  ↓
上传模型
  ↓
创建 Conversion Job
  ↓
排队
  ↓
XPU Worker 执行
  ↓
10 Stage Pipeline
  ↓
保存所有 Artifact / Log / Report
  ↓
实时查看进度
  ↓
Accuracy Validation
  ↓
XPU Benchmark
  ↓
Docker Package
  ↓
下载 deployment.zip
```

核心原则：

``` text
Converter = 转换能力
Engine    = 转换流程编排
Worker    = 计算执行
Queue     = 异步调度
Postgres  = 元数据/状态
MinIO     = 模型/Artifact
FastAPI   = API/Auth/Job Control
SSE       = 实时事件
Next.js   = UI
```

禁止把 Web 层直接改造成 Converter 核心。

------------------------------------------------------------------------

## 2. 当前仓库状态

当前已具备：

-   CLI
-   10 Stage Pipeline
-   Kunlun Backend
-   Accuracy Validator
-   Performance Benchmark
-   Docker Runtime Package
-   `ConversionContext`
-   `ArtifactStore`
-   `EventSink`
-   Pipeline 与 Context 的初步接入
-   Manifest/审计能力的初步实现
-   最新提交已经增加真实 XPU 执行门禁，CPU fallback 不能冒充 XPU 成功

当前缺失完整 SaaS：

-   FastAPI
-   用户登录
-   Project
-   RBAC
-   PostgreSQL
-   Redis Queue
-   Worker
-   XPU Worker 调度
-   MinIO/S3 ArtifactStore
-   SSE
-   Web Frontend
-   Retry
-   Cancel
-   Resume
-   Web Download
-   完整 Audit Log

因此不要认为当前仓库已经完成 Web SaaS。

------------------------------------------------------------------------

# 3. 开发规则

必须遵守：

1.  先扫描仓库，再编码。
2.  读取最新 Engine、`pipeline.py`、CLI、ArtifactStore、EventSink
    和测试。
3.  不重复实现已经存在的功能。
4.  不为了 Web 化重写稳定 Converter。
5.  CLI 必须继续可用。
6.  Web API 不得同步执行长时间转换。
7.  长任务必须进入 Queue，由 Worker 执行。
8.  模型和 Artifact 不进入 PostgreSQL BLOB。
9.  大文件通过 MinIO/S3。
10. 用户上传模型必须在隔离 Worker 中处理。
11. `.pt/.pth` 不允许在 FastAPI 进程中直接 `torch.load`。
12. XPU 成功必须基于真实 XPU 执行。
13. CPU fallback 必须是 `NOT_AVAILABLE` 或失败，不能是 SUCCESS。
14. stub backend 不能作为生产成功路径。
15. 所有关键状态必须持久化。
16. 所有关键事件必须可审计。
17. 所有 Artifact 必须有 SHA256。
18. 每个阶段完成后运行测试。
19. 不做无关的大规模重构。
20. 每个独立功能使用独立 Git Commit。

------------------------------------------------------------------------

# 4. 目标目录结构

如果当前仓库结构不同，以真实代码为准，采用最小侵入方式逐步演进：

``` text
xpu-model-converter/
├── xpu_converter/
│   ├── engine/
│   │   ├── context.py
│   │   ├── events.py
│   │   ├── artifact_store.py
│   │   ├── job.py
│   │   ├── stage.py
│   │   ├── manifest.py
│   │   └── runner.py
│   ├── pipeline.py
│   ├── backend/
│   ├── runtime/
│   ├── validator/
│   ├── exporter/
│   └── ...
├── platform/
│   ├── api/
│   │   ├── main.py
│   │   ├── dependencies.py
│   │   ├── routers/
│   │   ├── schemas/
│   │   ├── services/
│   │   └── middleware/
│   ├── worker/
│   │   ├── main.py
│   │   ├── tasks.py
│   │   ├── executor.py
│   │   └── scheduler.py
│   ├── db/
│   │   ├── models/
│   │   ├── repositories/
│   │   ├── migrations/
│   │   └── session.py
│   └── storage/
│       ├── minio_store.py
│       └── local_store.py
├── frontend/
├── deploy/
├── tests/
└── docs/
```

------------------------------------------------------------------------

# 5. Phase 1：完成 Engine 化

这是当前第一优先级。

## 5.1 ConversionContext

保持类似：

``` python
@dataclass
class ConversionContext:
    job_id: str
    workspace: Path
    source_model: Path
    config: ConversionConfig
    event_sink: EventSink
    artifact_store: ArtifactStore
```

要求：

-   Stage 不自行创建全局 workspace。
-   Artifact 统一通过 `ArtifactStore`。
-   Event 统一通过 `EventSink`。
-   Context 不依赖 FastAPI。
-   Context 不依赖 PostgreSQL。
-   Context 不依赖 Redis。
-   Worker、CLI、测试都可以构造同一 Engine Context。

------------------------------------------------------------------------

# 6. Job / Stage State Machine

## JobStatus

``` text
CREATED
QUEUED
RUNNING
SUCCESS
FAILED
CANCELLED
```

## StageStatus

``` text
PENDING
RUNNING
SUCCESS
FAILED
SKIPPED
CANCELLED
```

必须使用 Enum，并实现合法状态迁移。

例如禁止：

``` text
SUCCESS → RUNNING
SUCCESS → QUEUED
CANCELLED → RUNNING
```

状态迁移必须有测试。

------------------------------------------------------------------------

# 7. Job 数据模型

逻辑字段至少：

``` text
id
project_id
source_model_id
status
pipeline_version
config
created_at
queued_at
started_at
finished_at
error_code
error_message
worker_id
```

------------------------------------------------------------------------

# 8. JobStage 数据模型

至少：

``` text
id
job_id
stage_name
stage_order
status
progress
started_at
finished_at
duration_ms
error_code
error_message
metrics
```

当前 Pipeline 必须保持：

``` text
01 Model Inspect
02 Model Load
03 Export ONNX
04 ONNX Graph Check
05 XPU Operator Analysis
06 Graph Optimization
07 Backend Artifact Build
08 Accuracy Validation
09 Performance Benchmark
10 Docker Package
```

------------------------------------------------------------------------

# 9. Event 模型

Event 至少：

``` python
Event(
    event_id,
    job_id,
    stage,
    event,
    timestamp,
    sequence,
    progress,
    message,
    data,
)
```

必须增加：

-   `event_id`
-   `timestamp`
-   `sequence`

同一 Job 内：

``` text
1, 2, 3, 4, ...
```

严格递增。

Event 类型统一：

``` text
JOB_CREATED
JOB_QUEUED
JOB_STARTED
STAGE_STARTED
STAGE_PROGRESS
LOG
ARTIFACT_CREATED
STAGE_FINISHED
STAGE_FAILED
JOB_FINISHED
JOB_FAILED
JOB_CANCELLED
```

------------------------------------------------------------------------

# 10. EventSink

保留抽象：

``` python
class EventSink(ABC):
    def emit(self, event: Event) -> None:
        ...
```

实现：

``` text
CollectingEventSink
ConsoleEventSink
DatabaseEventSink
RedisEventSink
```

生产推荐：

``` text
Worker
 ↓
Redis Stream
 ↓
Event Consumer
 ↓
PostgreSQL
 ↓
SSE
 ↓
Browser
```

EventSink 故障不能让转换主流程失败，但不能静默吞掉生产事件错误；必须记录
warning/error 并具备监控或 fallback。

------------------------------------------------------------------------

# 11. ArtifactStore

当前已有 `ArtifactStore` 和 `LocalArtifactStore`，必须继续抽象。

生产实现：

``` text
LocalArtifactStore
MinioArtifactStore
```

Artifact key 推荐：

``` text
jobs/{job_id}/
├── source/model.pt
├── stages/01-inspect/...
├── stages/02-load/...
├── stages/03-onnx/...
├── stages/04-graph-check/...
├── stages/05-op-analysis/...
├── stages/06-optimization/...
├── stages/07-backend/...
├── stages/08-validation/...
├── stages/09-benchmark/...
├── stages/10-package/...
├── reports/manifest.json
├── reports/accuracy.json
├── reports/benchmark.json
├── logs/job.log
└── final/deployment.zip
```

重点修复：

``` python
Path(key).name
```

导致的 key 扁平化问题。

必须允许安全层级路径，同时防止：

``` text
../
绝对路径
root escape
```

Artifact 元数据至少：

``` text
artifact_id
job_id
stage
artifact_type
filename
storage_key
size
sha256
mime_type
created_at
```

------------------------------------------------------------------------

# 12. Manifest

每个 Job 最终必须有：

``` text
manifest.json
```

至少包含：

``` text
job_id
pipeline_version
converter_version
source_model
source_sha256
config
stages
artifacts
validation
benchmark
runtime
worker
created_at
finished_at
```

目标是能够追溯：

``` text
谁
什么模型
什么版本
什么配置
哪个 Worker
什么 XPU
产生什么 Artifact
Accuracy 如何
Benchmark 如何
```

------------------------------------------------------------------------

# 13. Phase 1 测试

至少增加：

``` text
Context
Job state transition
Stage state transition
Event sequence
Event timestamp
ArtifactStore
Artifact path traversal
Manifest
Pipeline lifecycle
Pipeline failure lifecycle
```

第一阶段完成后必须确保现有测试不回归。

------------------------------------------------------------------------

# 14. Phase 2：MinIO

实现：

``` text
MinioArtifactStore
```

支持：

``` text
put
get
exists
delete
presigned upload
presigned download
```

Storage Backend：

``` text
STORAGE_BACKEND=local
STORAGE_BACKEND=minio
```

Engine 不知道具体是 Local 还是 MinIO。

------------------------------------------------------------------------

# 15. Phase 3：PostgreSQL

使用：

``` text
SQLAlchemy 2.x
Alembic
PostgreSQL
```

至少建立：

``` text
users
projects
models
conversion_jobs
job_stages
artifacts
workers
audit_logs
```

后续：

``` text
project_members
worker_heartbeats
job_events
api_keys
```

PostgreSQL 只保存 metadata，不保存模型二进制。

------------------------------------------------------------------------

# 16. User

字段：

``` text
id
username
email
password_hash
is_active
created_at
```

密码使用成熟密码哈希：

``` text
Argon2
```

禁止明文密码。

API：

``` http
POST /api/v1/auth/register
POST /api/v1/auth/login
POST /api/v1/auth/refresh
GET  /api/v1/auth/me
```

MVP 可以使用 JWT Access Token + Refresh Token。

------------------------------------------------------------------------

# 17. Project

字段：

``` text
id
owner_id
name
description
created_at
updated_at
```

API：

``` http
POST   /api/v1/projects
GET    /api/v1/projects
GET    /api/v1/projects/{id}
PATCH  /api/v1/projects/{id}
DELETE /api/v1/projects/{id}
```

------------------------------------------------------------------------

# 18. Model

字段：

``` text
id
project_id
name
filename
framework
model_type
storage_key
sha256
size
created_at
```

状态：

``` text
UPLOADING
READY
INVALID
DELETED
```

支持：

``` text
.pt
.pth
.onnx
```

第一版优先 YOLOv10。

------------------------------------------------------------------------

# 19. 模型上传

禁止：

``` text
Browser → FastAPI → 大模型文件 → MinIO
```

推荐：

``` text
Browser
 ↓
POST /models/upload-url
 ↓
FastAPI
 ↓
Presigned URL
 ↓
Browser
 ↓
MinIO
 ↓
POST /models/{id}/complete
```

Complete 时验证：

``` text
size
sha256
extension
mime
```

模型解析必须进入隔离 Worker。

------------------------------------------------------------------------

# 20. 模型安全

`.pt/.pth` 可能包含 pickle。

禁止 API 进程直接：

``` python
torch.load(...)
```

Worker 必须：

``` text
non-root
isolated workspace
network disabled
resource limits
timeout
limited filesystem
```

不得把用户上传模型视为可信输入。

------------------------------------------------------------------------

# 21. Phase 4：Redis Queue

MVP 推荐：

``` text
Celery + Redis
```

如果当前项目已有其他成熟 Queue 实现，则优先沿用。

流程：

``` text
API
 ↓
DB Job = CREATED
 ↓
Queue
 ↓
DB Job = QUEUED
 ↓
Worker
 ↓
DB Job = RUNNING
```

API 不直接执行 Pipeline。

------------------------------------------------------------------------

# 22. Worker

Worker 执行：

``` text
获取 Job
 ↓
读取 DB metadata
 ↓
获取 source model
 ↓
创建 isolated workspace
 ↓
构建 ConversionContext
 ↓
执行 run_pipeline()
 ↓
保存 Artifact
 ↓
更新 Stage
 ↓
更新 Job
```

禁止 Worker：

``` text
处理登录
管理权限
直接返回 HTTP
```

------------------------------------------------------------------------

# 23. XPU Worker

Worker 必须声明：

``` text
worker_id
hostname
device_type
chip
device_id
sdk_version
driver_version
status
last_heartbeat
```

例如：

``` json
{
  "worker_id": "xpu-worker-01",
  "device_type": "xpu",
  "chip": "KUNLUNXIN",
  "device_id": "0",
  "status": "READY"
}
```

Job 要求：

``` text
target = kunlun_xpu
```

只能调度到：

``` text
device_type = xpu
status = READY
```

------------------------------------------------------------------------

# 24. XPU 成功门禁

必须继续保持当前项目的硬门禁：

``` text
actual_device == xpu
device_available == true
chip != auto
SDK 可探测
driver 可探测
```

CPU fallback：

``` text
NOT_AVAILABLE
```

不能：

``` text
SUCCESS
```

------------------------------------------------------------------------

# 25. Phase 5：FastAPI

API 层：

``` text
platform/api/
├── main.py
├── dependencies.py
├── routers/
│   ├── auth.py
│   ├── projects.py
│   ├── models.py
│   ├── jobs.py
│   ├── artifacts.py
│   └── workers.py
├── schemas/
├── services/
└── middleware/
```

统一前缀：

``` text
/api/v1
```

------------------------------------------------------------------------

# 26. Job API

创建：

``` http
POST /api/v1/jobs
```

请求：

``` json
{
  "project_id": "...",
  "model_id": "...",
  "config": {
    "model_type": "yolov10",
    "task": "detection",
    "input_size": [640, 640],
    "batch_size": 1,
    "precision": "fp16",
    "target": "kunlun_xpu"
  }
}
```

返回：

``` json
{
  "job_id": "...",
  "status": "CREATED"
}
```

API 只创建任务并入队。

------------------------------------------------------------------------

# 27. Job API 完整集合

``` http
GET  /api/v1/jobs
GET  /api/v1/jobs/{job_id}
POST /api/v1/jobs
POST /api/v1/jobs/{job_id}/retry
POST /api/v1/jobs/{job_id}/cancel
DELETE /api/v1/jobs/{job_id}
```

Job Detail 返回：

``` json
{
  "id": "...",
  "status": "RUNNING",
  "progress": 63,
  "stages": [
    {
      "name": "onnx_export",
      "status": "SUCCESS",
      "progress": 100
    },
    {
      "name": "accuracy_validation",
      "status": "RUNNING",
      "progress": 40
    }
  ]
}
```

------------------------------------------------------------------------

# 28. SSE

实现：

``` http
GET /api/v1/jobs/{job_id}/events
```

发送：

``` text
event: stage_started
data: {...}

event: stage_progress
data: {...}

event: log
data: {...}

event: artifact_created
data: {...}

event: stage_finished
data: {...}
```

保留：

``` http
GET /api/v1/jobs/{job_id}
```

用于断线恢复。

Event 使用：

``` text
sequence
```

支持：

``` text
Last-Event-ID
```

------------------------------------------------------------------------

# 29. Artifact API

实现：

``` http
GET /api/v1/jobs/{job_id}/artifacts
GET /api/v1/artifacts/{artifact_id}
GET /api/v1/artifacts/{artifact_id}/download
```

大文件下载：

``` text
FastAPI
 ↓
Presigned URL
 ↓
MinIO
```

不要由 FastAPI 读取整个 ZIP 再返回。

------------------------------------------------------------------------

# 30. Audit Log

至少记录：

``` text
user_id
action
resource_type
resource_id
timestamp
ip
metadata
```

Actions：

``` text
MODEL_UPLOADED
JOB_CREATED
JOB_STARTED
JOB_CANCELLED
JOB_RETRIED
ARTIFACT_DOWNLOADED
```

------------------------------------------------------------------------

# 31. Retry

必须防止同一 Job 被两个 Worker 同时执行。

需要：

``` text
lease / lock
task id
job state validation
```

失败 Job 可以 Retry。

Retry 时：

``` text
创建新的 execution attempt
```

不要破坏历史审计记录。

------------------------------------------------------------------------

# 32. Cancel

实现：

``` http
POST /api/v1/jobs/{job_id}/cancel
```

Worker 定期检查：

``` text
cancel_requested
```

Stage 必须安全退出。

不要粗暴杀宿主机进程。

------------------------------------------------------------------------

# 33. Resume

第一版允许：

``` text
失败后从失败 Stage 重新执行
```

后续支持：

``` text
从指定 Stage 恢复
```

恢复前检查：

``` text
上游 Artifact 存在
sha256 正确
config 没有变化
```

------------------------------------------------------------------------

# 34. Idempotency

创建 Job 推荐支持：

``` text
Idempotency-Key
```

重复请求不能产生多个相同 Job。

Worker 对重复 Queue 消息也必须幂等。

------------------------------------------------------------------------

# 35. Phase 6：Frontend

推荐：

``` text
Next.js
TypeScript
Tailwind CSS
shadcn/ui
```

页面：

``` text
/login
/register
/dashboard
/projects
/projects/[id]
/models/[id]
/jobs
/jobs/[id]
/settings
```

------------------------------------------------------------------------

# 36. Dashboard

展示：

``` text
Project 数量
Model 数量
Running Jobs
Success Jobs
Failed Jobs
XPU Worker 状态
```

------------------------------------------------------------------------

# 37. Upload 页面

展示：

``` text
文件名
大小
SHA256
上传进度
上传状态
模型框架
模型类型
```

------------------------------------------------------------------------

# 38. Job Detail

这是最重要页面。

必须展示：

``` text
Job Status
Overall Progress
Stage Timeline
Live Logs
Artifacts
Accuracy
Benchmark
Deployment Package
Worker
Runtime
```

建议：

``` text
✓ Inspect
✓ Load
✓ Export ONNX
✓ Graph Check
✓ Operator Analysis
✓ Optimization
● Backend Build
○ Accuracy
○ Benchmark
○ Docker Package
```

------------------------------------------------------------------------

# 39. Frontend 状态

Job：

``` text
CREATED
QUEUED
RUNNING
SUCCESS
FAILED
CANCELLED
```

Stage：

``` text
PENDING
RUNNING
SUCCESS
FAILED
SKIPPED
CANCELLED
```

前端不要自己推断后端状态。

------------------------------------------------------------------------

# 40. Phase 7：Docker Compose

开发环境：

``` yaml
services:
  postgres:
  redis:
  minio:
  api:
  worker:
  frontend:
```

启动：

``` bash
docker compose up
```

必须实现完整开发链路：

``` text
Browser
 ↓
FastAPI
 ↓
PostgreSQL
 ↓
Redis
 ↓
Worker
 ↓
Engine
 ↓
MinIO
```

XPU Worker 根据宿主机实际 XPU 环境单独部署。

------------------------------------------------------------------------

# 41. 配置

统一使用 Pydantic Settings。

至少：

``` text
DATABASE_URL
REDIS_URL
MINIO_ENDPOINT
MINIO_ACCESS_KEY
MINIO_SECRET_KEY
MINIO_BUCKET
STORAGE_BACKEND
JWT_SECRET
WORKSPACE_ROOT
LOG_LEVEL
```

Secret 禁止提交 Git。

------------------------------------------------------------------------

# 42. 错误格式

统一：

``` json
{
  "code": "JOB_NOT_FOUND",
  "message": "Conversion job not found",
  "request_id": "..."
}
```

前端不得依赖 Python exception message。

------------------------------------------------------------------------

# 43. Request ID

所有 API 请求：

``` text
X-Request-ID
```

并进入：

``` text
API log
Worker log
Audit log
```

------------------------------------------------------------------------

# 44. Logging

API / Worker / Engine 使用 structured logging。

至少：

``` text
timestamp
level
service
request_id
job_id
stage
message
```

------------------------------------------------------------------------

# 45. MVP 模型范围

第一版严格收敛：

``` text
YOLOv10
Detection
NCHW
Static Shape
Batch 1
640x640
FP16
Kunlun XPU
CPU NMS
Docker Deployment Package
```

完整流程：

``` text
best.pt
 ↓
Inspect
 ↓
Load
 ↓
PyTorch → ONNX
 ↓
Graph Check
 ↓
XPU Operator Analysis
 ↓
Graph Optimization
 ↓
Backend Artifact Build
 ↓
XPU Accuracy Validation
 ↓
XPU Benchmark
 ↓
Docker Package
 ↓
deployment.zip
```

第一版不要扩展：

``` text
YOLOv8/v9/v11 等
PaddleOCR
PaddleDetection
复杂多任务
模型市场
计费
Kubernetes Operator
复杂多租户
```

------------------------------------------------------------------------

# 46. 安全要求

必须具备：

``` text
上传大小限制
文件类型限制
MIME 检查
SHA256
路径穿越防护
JWT expiration
密码哈希
Job workspace 隔离
Worker non-root
容器资源限制
执行 timeout
网络隔离
```

特别注意：

``` text
.pt
.pth
.onnx
.zip
Docker package
pickle
```

------------------------------------------------------------------------

# 47. Docker Package 安全

生产下载前验证：

``` text
artifact exists
sha256 matches
package status = valid
validation passed
benchmark valid
not degraded
```

禁止用户下载作为正式部署包：

``` text
FAILED
DEGRADED
UNVERIFIED
```

------------------------------------------------------------------------

# 48. Cache

设计：

``` text
source_sha256
pipeline_version
config_hash
converter_version
```

计算：

``` text
conversion_fingerprint
```

后续可以实现相同 fingerprint 的 Artifact 复用。

第一阶段只设计，不强制实现完整缓存。

------------------------------------------------------------------------

# 49. CI

至少执行：

``` bash
pytest
ruff
mypy
```

如果仓库已有其他规范，优先沿用现有规范。

------------------------------------------------------------------------

# 50. Git Commit

每个功能独立 Commit，例如：

``` text
feat(engine): complete job stage state machine
feat(engine): add event sequence and timestamp
fix(storage): preserve hierarchical artifact keys
feat(storage): add minio artifact store
feat(platform): add postgres models
feat(worker): add redis job worker
feat(api): add conversion job endpoints
feat(api): add job event sse
feat(frontend): add job detail page
```

------------------------------------------------------------------------

# 51. 开发阶段顺序

严格按照：

``` text
Phase 1
Engine Job/Stage/Event/Artifact/Manifest

Phase 2
MinIO

Phase 3
PostgreSQL

Phase 4
Redis + Worker

Phase 5
FastAPI

Phase 6
SSE

Phase 7
Next.js Frontend

Phase 8
Docker Compose

Phase 9
真实 XPU Worker

Phase 10
E2E
```

不要跳过 Engine 直接开发 UI。

------------------------------------------------------------------------

# 52. 第一轮立即执行

现在不要直接开始写 Web 页面。

首先：

1.  扫描当前仓库。
2.  读取 `xpu_converter/engine/` 全部实现。
3.  读取 `pipeline.py`。
4.  读取 CLI。
5.  读取 `ArtifactStore`。
6.  读取 `EventSink`。
7.  读取最新测试。
8.  检查当前是否真的存在 Job / Stage / Manifest 实现。
9.  输出"实际代码 vs 本文档"的差异。
10. 只修 Phase 1。
11. 不提前实现 PostgreSQL / Redis / Frontend。

------------------------------------------------------------------------

# 53. Phase 1 的具体开发任务

## Task 1：Job State Machine

实现：

``` text
JobStatus
合法 transition
非法 transition exception
```

加入测试。

## Task 2：Stage State Machine

实现：

``` text
StageStatus
合法 transition
非法 transition exception
```

加入测试。

## Task 3：Event

补充：

``` text
event_id
timestamp
sequence
```

加入测试。

## Task 4：ArtifactStore

修复：

``` text
Path(key).name
```

导致的层级丢失。

加入：

``` text
nested key
path traversal
absolute path
root escape
```

测试。

## Task 5：Manifest

确认最终 Manifest 能描述：

``` text
Job
Config
Source
Stages
Artifacts
Validation
Benchmark
Worker
Runtime
```

## Task 6：Pipeline

确保：

``` text
Context
 ↓
Stage
 ↓
Event
 ↓
Artifact
 ↓
Manifest
```

真正贯通。

## Task 7：Regression

运行全部现有测试。

------------------------------------------------------------------------

# 54. Phase 1 验收

必须达到：

``` text
CLI
 ↓
ConversionContext
 ↓
Job
 ↓
Stage State Machine
 ↓
Pipeline
 ↓
ArtifactStore
 ↓
EventSink
 ↓
Manifest
```

并且：

``` text
pytest 全部通过
```

不能因为 Engine 改造破坏现有：

``` text
KunlunBackend
AccuracyValidator
Benchmark
Runtime
RuntimePackager
CLI
```

------------------------------------------------------------------------

# 55. 第二轮立即执行

Phase 1 完成后才进入：

``` text
MinIO
```

验收：

``` text
同一个 Conversion Engine
可以切换：

LocalArtifactStore
MinioArtifactStore
```

------------------------------------------------------------------------

# 56. 第三轮

然后：

``` text
PostgreSQL
```

实现完整状态持久化：

``` text
Job
Stage
Artifact
Event
Worker
Audit
```

------------------------------------------------------------------------

# 57. 第四轮

然后：

``` text
Redis
Worker
Scheduler
```

验收：

``` text
API 创建 Job
 ↓
Queue
 ↓
Worker
 ↓
Engine
 ↓
DB 状态更新
 ↓
Artifact 保存
```

------------------------------------------------------------------------

# 58. 第五轮

然后：

``` text
FastAPI
Auth
Project
Model
Job
Artifact
SSE
```

------------------------------------------------------------------------

# 59. 第六轮

然后：

``` text
Next.js
Login
Dashboard
Project
Upload
Job Detail
Artifact
Download
```

------------------------------------------------------------------------

# 60. 第七轮

最后接真实 XPU Worker。

必须真实执行：

``` text
YOLOv10
→ ONNX
→ Kunlun Backend
→ XPU Accuracy
→ XPU Benchmark
→ Docker Package
```

------------------------------------------------------------------------

# 61. E2E 最终验收

只有下面流程完整跑通，MVP 才算完成：

``` text
用户注册
 ↓
用户登录
 ↓
创建 Project
 ↓
上传 YOLOv10 best.pt
 ↓
上传完成
 ↓
创建 Conversion Job
 ↓
QUEUED
 ↓
XPU Worker 获取
 ↓
RUNNING
 ↓
Stage 01
 ↓
Stage 02
 ↓
Stage 03
 ↓
Stage 04
 ↓
Stage 05
 ↓
Stage 06
 ↓
Stage 07
 ↓
Stage 08 XPU Accuracy
 ↓
Stage 09 XPU Benchmark
 ↓
Stage 10 Docker Package
 ↓
SUCCESS
 ↓
manifest.json
 ↓
deployment.zip
 ↓
Presigned Download
```

同时必须满足：

``` text
实时进度可见
实时日志可见
Artifact 可查询
Accuracy 可查询
Benchmark 可查询
失败原因可见
Retry 可用
Cancel 可用
Manifest 可审计
部署包可下载
```

------------------------------------------------------------------------

# 62. 最终架构验收

最终必须形成：

``` text
                           Browser
                              │
                             HTTPS
                              │
                         ┌────▼────┐
                         │ FastAPI │
                         └──┬───┬──┘
                            │   │
                 ┌──────────┘   └──────────┐
                 ▼                         ▼
             PostgreSQL                  MinIO
                 │                         │
                 └──────────┬──────────────┘
                            ▼
                       Redis Queue
                            │
                            ▼
                         Worker
                            │
                            ▼
                    Conversion Engine
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
           Pipeline     ArtifactStore   EventSink
              │             │             │
              ▼             ▼             ▼
          Converter       MinIO         Redis/SSE
              │
              ▼
         Kunlun XPU
              │
              ▼
       deployment.zip
```

------------------------------------------------------------------------

# 63. 开发过程中的强制输出

每完成一个阶段，必须输出：

``` text
## 已完成

- ...

## 修改文件

- ...

## 数据库变化

- ...

## API 变化

- ...

## 测试结果

pytest:
X passed
Y failed
Z skipped

## 当前架构

...

## 当前未完成

- ...

## 风险

- ...

## 下一步

...
```

------------------------------------------------------------------------

# 64. 最终要求

不要把这个项目理解成：

``` text
“给现有 Converter 加一个网页”
```

正确目标是：

``` text
xpu-model-converter
        ↓
Production Conversion Engine
        ↓
Asynchronous Worker Platform
        ↓
Web SaaS
```

最终系统必须具备：

``` text
可登录
可上传
可排队
可执行
可实时观察
可持久化
可审计
可重试
可取消
可恢复
可下载
可验证
可扩展 XPU Worker
```

所有新代码必须先判断它属于：

``` text
Converter
Engine
Worker
Queue
Storage
Database
API
SSE
Frontend
```
如果职责无法明确归类，先重新设计再编码。