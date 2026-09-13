# -*- coding: utf-8 -*-
"""ConversionContext(第2轮 Web SaaS 架构, §38)。

把所有与"一次转换"相关的路径、配置、事件与存储抽象收敛到一个对象, 使 Pipeline
各阶段只操作 ``ctx``, 而不再各自拼接路径; Worker / FastAPI / 测试都能用同一套
对象驱动转换。
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from xpu_converter.engine.artifact_store import ArtifactStore, LocalArtifactStore
from xpu_converter.engine.events import ConsoleEventSink, Event, EventSink


@dataclass
class ConversionContext:
    """一次转换任务的执行上下文。"""

    job_id: str
    workspace: Path
    source_model: Path
    config: Dict[str, Any] = field(default_factory=dict)
    event_sink: EventSink = field(default_factory=ConsoleEventSink)
    artifact_store: ArtifactStore = field(default_factory=lambda: LocalArtifactStore(str(Path.cwd() / "artifacts")))

    @property
    def artifacts_dir(self) -> Path:
        return self.workspace / "artifacts"

    @property
    def logs_dir(self) -> Path:
        return self.workspace / "logs"

    @property
    def reports_dir(self) -> Path:
        return self.workspace / "reports"

    def prepare(self) -> "ConversionContext":
        for directory in (self.workspace, self.artifacts_dir, self.logs_dir, self.reports_dir):
            Path(directory).mkdir(parents=True, exist_ok=True)
        return self

    def emit(self, event: Event) -> None:
        """发一条事件(失败不阻断主流程)。"""
        self.event_sink.emit_typed(event)

    def save_artifact(self, stage: str, path: str, artifact_type: str = "") -> str:
        """保存某阶段产物到 artifact_store, 返回存储 key。"""
        name = Path(path).name
        key = "{}/{}-{}".format(stage, artifact_type, name) if artifact_type else "{}/{}".format(stage, name)
        from xpu_converter.engine import events

        stored = self.artifact_store.put(key, path)
        self.emit(events.artifact_created(
            self.job_id, stage, artifact_type=artifact_type, filename=name, data={"key": key},
        ))
        return stored


def new_context(
    job_id: str,
    workspace: str,
    source_model: str,
    config: Optional[Dict[str, Any]] = None,
    event_sink: Optional[EventSink] = None,
    artifact_store: Optional[ArtifactStore] = None,
) -> ConversionContext:
    """工厂: 校验输入并准备目录。"""
    from xpu_converter.engine.events import CollectingEventSink

    return ConversionContext(
        job_id=job_id,
        workspace=Path(workspace),
        source_model=Path(source_model),
        config=dict(config or {}),
        event_sink=event_sink or CollectingEventSink(),
        artifact_store=artifact_store or LocalArtifactStore(str(Path(workspace) / "artifacts")),
    ).prepare()