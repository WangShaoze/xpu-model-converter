# -*- coding: utf-8 -*-
"""Worker 执行器(Phase 4 §22/§24)。

流程: 取 Job → 读 DB 元数据 → 取源模型 → 隔离 workspace → 构造
ConversionContext → run_pipeline → 落 Artifact → 更新 Stage → 更新 Job。
Worker 不处理登录/权限/HTTP。
"""
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from sqlalchemy.orm import Session

from xpu_platform.db.models import Model
from xpu_platform.db.repositories import (
    ArtifactRepository,
    JobEventRepository,
    JobRepository,
)
from xpu_converter.engine import CollectingEventSink, LocalArtifactStore, new_context
from xpu_converter.engine.converter import run_pipeline

# pipeline_factory(model_path, config) -> ConversionPipeline 兼容对象
PipelineFactory = Callable[[str, Dict[str, Any]], Any]


def parse_engine_ts(value: Optional[str]) -> Optional[datetime]:
    """Engine 的 finished_at 是 ISO 字符串(如 2026-09-13T18:21:01.414209Z), 落库转 datetime。"""
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


@dataclass
class WorkerRuntime:
    """Worker 设备声明(§23): 只有 device_type=xpu 且 status=READY 才可被调度。"""
    worker_id: str
    hostname: str = ""
    device_type: str = "xpu"
    chip: str = ""
    device_id: str = "0"
    sdk_version: str = ""
    driver_version: str = ""
    status: str = "READY"

    def to_row(self) -> Dict[str, Any]:
        from xpu_platform.db.models import Worker as WorkerRow
        return {
            "worker_id": self.worker_id, "hostname": self.hostname,
            "device_type": self.device_type, "chip": self.chip, "device_id": self.device_id,
            "sdk_version": self.sdk_version, "driver_version": self.driver_version,
            "status": self.status,
        }


def fetch_source_model(db: Session, model_id: str, workspace: Path) -> Path:
    """从存储取回源模型到 workspace(§22: Worker 内隔离处理, API 进程不 touch)。"""
    model = db.get(Model, model_id)
    if model is None:
        raise FileNotFoundError("源模型记录不存在: {}".format(model_id))
    if not model.storage_key:
        raise FileNotFoundError("源模型无 storage_key: {}".format(model_id))
    # 第一版: storage_key 指向本地 workspace 内文件(MinIO 取回后落本地再进入引擎)
    src = Path(model.storage_key)
    if not src.is_absolute():
        src = Path.cwd() / src
    if not src.is_file():
        raise FileNotFoundError("源模型文件不存在: {}".format(src))
    target = workspace / "source" / (model.filename or src.name)
    target.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != target.resolve():
        import shutil
        shutil.copyfile(str(src), str(target))
    return target


def run_job(
    job_id: str,
    session_factory,
    pipeline_factory: PipelineFactory,
    worker: WorkerRuntime,
    workspace_root: str,
    artifact_store=None,
) -> Dict[str, Any]:
    """执行单个 Job, 返回 ``{"status": ..., "manifest": ...}``。

    通过乐观锁 :meth:`JobRepository.claim` 抢占, 防双 Worker 同时执行。
    """
    db: Session = session_factory()
    try:
        job_repo = JobRepository(db)
        job = job_repo.get(job_id)
        if job is None:
            return {"status": "NOT_FOUND"}
        claimed = job_repo.claim(job_id, from_status=["CREATED", "QUEUED", "FAILED"])
        if claimed is None:
            return {"status": "SKIPPED"}  # 已被其它 Worker 抢占

        job_repo.update_status(job_id, "RUNNING", worker_id=worker.worker_id)
        workspace = Path(workspace_root) / job_id
        workspace.mkdir(parents=True, exist_ok=True)
        source = fetch_source_model(db, job.source_model_id, workspace)
        store = artifact_store or LocalArtifactStore(str(workspace / "artifacts"))

        event_sink = CollectingEventSink()
        ctx = new_context(job_id=job_id, workspace=str(workspace), source_model=str(source),
                          config=dict(job.config), event_sink=event_sink, artifact_store=store)
        pipeline = pipeline_factory(str(source), dict(job.config))
        job_obj, manifest, _ = run_pipeline(ctx, pipeline)

        # 落 Stage 行(10 阶段)
        for stage in job_obj.stages:
            row = job_repo.add_stage(job_id, stage.name, stage.index)
            job_repo.update_stage(
                row.id, status=stage.status.value, progress=stage.progress,
                error_message=stage.error_message,
            )
        # 落 Artifact 元数据行
        artifact_repo = ArtifactRepository(db)
        for atype, path in (manifest.get("artifacts") or {}).items():
            p = Path(path)
            artifact_repo.create(
                job_id=job_id, filename=p.name, storage_key=path, artifact_type=atype,
                size=p.stat().st_size if p.is_file() else 0,
            )
        # 落 Event 行(sequence 严格递增)
        event_repo = JobEventRepository(db)
        for ev in event_sink.events:
            event_repo.append(job_id, ev.event, stage=ev.stage, message=ev.message,
                              progress=ev.progress, data=ev.data)

        final_status = "SUCCESS" if job_obj.status.value == "SUCCESS" else "FAILED"
        job_repo.update_status(
            job_id, final_status,
            finished_at=parse_engine_ts(job_obj.finished_at),
            error_message=manifest.get("error", ""),
        )
        return {"status": final_status, "manifest": manifest}
    finally:
        db.close()
