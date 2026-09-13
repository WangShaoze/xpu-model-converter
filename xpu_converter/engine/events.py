# -*- coding: utf-8 -*-
"""Engine 事件模型(第2轮 Web SaaS 架构, Phase 1/§22-§23)。

把\"流水线跑到哪一步了\"从 CLI 的 stdout 抽象成结构化事件流, 使同一个 Pipeline
可以被 CLI / Web(SSE) / Worker / 测试 复用。CLI 用 :class:`ConsoleEventSink`,
Web 用 Redis/DB EventSink 订阅后推送给浏览器。
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Event:
    """统一事件 Schema(建议格式 §23)。

    ``{"job_id": ..., "stage": ..., "event": ..., "progress": ..., "message": ...,
        "timestamp": ...}``
    """
    event: str
    job_id: str = ""
    stage: str = ""
    progress: Optional[int] = None
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": self.event,
            "job_id": self.job_id,
            "stage": self.stage,
            "progress": self.progress,
            "message": self.message,
            "data": dict(self.data),
        }


def stage_started(job_id: str, stage: str) -> Event:
    return Event("stage_started", job_id=job_id, stage=stage)


def stage_progress(job_id: str, stage: str, progress: int, message: str = "") -> Event:
    return Event("stage_progress", job_id=job_id, stage=stage, progress=int(progress), message=message)


def log_event(job_id: str, stage: str, message: str) -> Event:
    return Event("log", job_id=job_id, stage=stage, message=message)


def artifact_created(job_id: str, stage: str, artifact_type: str = "",
                     filename: str = "", data: Optional[Dict[str, Any]] = None) -> Event:
    return Event("artifact_created", job_id=job_id, stage=stage,
                 data=dict(data or {}, **{"type": artifact_type, "filename": filename}))


def stage_finished(job_id: str, stage: str, status: str, progress: int = 100) -> Event:
    return Event("stage_finished", job_id=job_id, stage=stage,
                 progress=progress, data={"status": status})


def job_finished(job_id: str, status: str) -> Event:
    return Event("job_finished", job_id=job_id, data={"status": status})


class EventSink(ABC):
    """事件接收器。实现方例如：控制台、数据库、Redis。

    所有实现在 :meth:`emit` 内把事件落地; 抛错不得阻塞主流程。
    """

    @abstractmethod
    def emit(self, event: Event) -> None:
        """把一条事件交给下游。"""

    def emit_typed(self, event: Event) -> None:
        try:
            self.emit(event)
        except Exception:
            # 事件流失败不应影响转换主流程本身
            pass


class CollectingEventSink(EventSink):
    """测试/调试用: 把事件回收到内存列表, 也可顺带打印到控制台。"""

    def __init__(self, echo: bool = False) -> None:
        self.echo = echo
        self.events: List[Event] = []

    def emit(self, event: Event) -> None:
        if self.echo:
            print(event.to_dict())
        self.events.append(event)


class ConsoleEventSink(CollectingEventSink):
    """CLI 默认: 按可读格式输出事件。"""

    def __init__(self) -> None:
        super().__init__(echo=False)
        self.lines: List[str] = []

    def emit(self, event: Event) -> None:
        self.events.append(event)
        if event.event in ("log", "stage_progress"):
            return
        prefix = "[{}]".format(event.stage) if event.stage else "[job]"
        self.lines.append("{} {} {}".format(prefix, event.event, event.message or ""))


def emit(collect: List[Event], event: Event) -> None:
    """便捷函数: 收集到列表(缺省 sink 时的最小实现)。"""
    collect.append(event)