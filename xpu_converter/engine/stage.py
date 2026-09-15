# -*- coding: utf-8 -*-
"""Job / Stage 状态机(第2轮 Web SaaS 架构, §5)。

定义 Job 与 Stage 的状态流转, 供 Worker / API / 前端共享同一套状态语义:

- Job:   CREATED -> QUEUED -> RUNNING -> SUCCESS / FAILED; RUNNING -> CANCEL_REQUESTED -> CANCELLED
- Stage: PENDING -> RUNNING -> SUCCESS / FAILED / SKIPPED / CANCELLED
- 所有迁移经过合法迁移表校验, 非法迁移抛 :class:`InvalidTransitionError`
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class JobStatus(str, Enum):
    CREATED = "CREATED"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELLED = "CANCELLED"


class StageStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    CANCELLED = "CANCELLED"


class InvalidTransitionError(ValueError):
    """非法状态迁移(§6: 例如 SUCCESS -> RUNNING 禁止)。"""


# Job 合法迁移表: 目标状态集合
_JOB_TRANSITIONS = {
    JobStatus.CREATED: {JobStatus.QUEUED},
    JobStatus.QUEUED: {JobStatus.RUNNING},
    JobStatus.RUNNING: {JobStatus.SUCCESS, JobStatus.FAILED, JobStatus.CANCEL_REQUESTED},
    JobStatus.CANCEL_REQUESTED: {JobStatus.CANCELLED},
    JobStatus.SUCCESS: set(),
    JobStatus.FAILED: set(),
    JobStatus.CANCELLED: set(),
}

# Stage 合法迁移表
_STAGE_TRANSITIONS = {
    StageStatus.PENDING: {StageStatus.RUNNING},
    StageStatus.RUNNING: {StageStatus.SUCCESS, StageStatus.FAILED, StageStatus.SKIPPED, StageStatus.CANCELLED},
    StageStatus.SUCCESS: set(),
    StageStatus.FAILED: set(),
    StageStatus.SKIPPED: set(),
    StageStatus.CANCELLED: set(),
}


def _now() -> str:
    return datetime.utcnow().isoformat() + "Z"


@dataclass
class JobStage:
    """单个 Stage 的独立状态与产物对账(§9 job_stages 表)。"""
    name: str
    index: int
    status: StageStatus = StageStatus.PENDING
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    progress: int = 0
    input_artifact: str = ""
    output_artifact: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    error_message: str = ""

    def _transition(self, target: "StageStatus") -> "StageStatus":
        allowed = _STAGE_TRANSITIONS.get(self.status, set())
        if target not in allowed:
            raise InvalidTransitionError(
                "非法 Stage 状态迁移: {} -> {}".format(self.status.value, target.value))
        self.status = target
        return self.status

    def start(self) -> "JobStage":
        self._transition(StageStatus.RUNNING)
        self.started_at = _now()
        self.progress = 0
        return self

    def finish(self, status: StageStatus = StageStatus.SUCCESS, error: str = "") -> "JobStage":
        self._transition(status)
        self.finished_at = _now()
        self.progress = 100 if status in (StageStatus.SUCCESS, StageStatus.SKIPPED) else self.progress
        self.error_message = error
        return self

    def cancel(self) -> "JobStage":
        self._transition(StageStatus.CANCELLED)
        self.finished_at = _now()
        self.error_message = "cancelled"
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "index": self.index,
            "status": self.status.value,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "progress": self.progress,
            "input_artifact": self.input_artifact,
            "output_artifact": self.output_artifact,
            "metrics": dict(self.metrics),
            "error_message": self.error_message,
        }


@dataclass
class ConversionJob:
    """一次转换任务的状态机(§8 conversion_jobs 表)。"""
    job_id: str
    status: JobStatus = JobStatus.CREATED
    current_stage: str = ""
    stages: List[JobStage] = field(default_factory=list)
    created_at: str = field(default_factory=_now)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)

    def _transition(self, target: "JobStatus") -> "JobStatus":
        allowed = _JOB_TRANSITIONS.get(self.status, set())
        if target not in allowed:
            raise InvalidTransitionError(
                "非法 Job 状态迁移: {} -> {}".format(self.status.value, target.value))
        self.status = target
        return self.status

    def queue(self) -> "ConversionJob":
        self._transition(JobStatus.QUEUED)
        return self

    def run(self) -> "ConversionJob":
        self._transition(JobStatus.RUNNING)
        self.started_at = self.started_at or _now()
        return self

    def succeed(self) -> "ConversionJob":
        self._transition(JobStatus.SUCCESS)
        self.finished_at = _now()
        return self

    def fail(self, error: str = "") -> "ConversionJob":
        self._transition(JobStatus.FAILED)
        self.finished_at = _now()
        self.payload["error_message"] = error
        return self

    def request_cancel(self) -> "ConversionJob":
        self._transition(JobStatus.CANCEL_REQUESTED)
        return self

    def cancel(self) -> "ConversionJob":
        self._transition(JobStatus.CANCELLED)
        self.finished_at = _now()
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "current_stage": self.current_stage,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "payload": dict(self.payload),
            "stages": [stage.to_dict() for stage in self.stages],
        }