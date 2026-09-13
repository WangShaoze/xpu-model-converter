# -*- coding: utf-8 -*-
"""Engine 抽象层(第2轮 Web SaaS 架构, Phase 1)。

把转换流水线从\"CLI 一次性脚本\"升级为可被 Worker / FastAPI / 前端复用的运转核心:

- :mod:`context`        : ConversionContext —— 一次转换的执行上下文
- :mod:`stage`          : Job / Stage 状态机
- :mod:`artifact_store` : ArtifactStore —— 产物存取抽象(本地可换 S3/MinIO)
- :mod:`events`         : EventSink —— 结构化事件流(CLI/Web/测试复用)
- :mod:`converter`      : 把 Pipeline 接入 Context 并产出可审计 manifest
"""
from xpu_converter.engine.artifact_store import (
    ArtifactStore,
    InvalidArtifactKey,
    LocalArtifactStore,
)
from xpu_converter.engine.context import ConversionContext, new_context
from xpu_converter.engine.events import (
    Event,
    EventSink,
    CollectingEventSink,
    ConsoleEventSink,
)
from xpu_converter.engine.stage import (
    ConversionJob,
    InvalidTransitionError,
    JobStage,
    JobStatus,
    StageStatus,
)

__all__ = [
    "ArtifactStore", "LocalArtifactStore", "InvalidArtifactKey",
    "ConversionContext", "new_context",
    "Event", "EventSink", "CollectingEventSink", "ConsoleEventSink",
    "ConversionJob", "JobStage", "JobStatus", "StageStatus", "InvalidTransitionError",
]