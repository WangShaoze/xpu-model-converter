# -*- coding: utf-8 -*-
"""Worker 执行器(Phase 4 §22/§24/§48)。

流程: 取 Job → 读 DB 元数据 → 取源模型 → 隔离 workspace → 构造
ConversionContext → run_pipeline → 落 Artifact → 更新 Stage → 更新 Job。
Worker 不处理登录/权限/HTTP。
"""
import hashlib
import json
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


def compute_fingerprint(source_sha256: str, pipeline_version: str,
                        config: Dict[str, Any], converter_version: str = "") -> str:
    """§48 缓存指纹: source_sha256 + pipeline_version + config_hash + converter_version。

    相同 fingerprint 的 Job 可复用已有 Artifact, 跳过重复转换。
    """
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]
    raw = "{}|{}|{}|{}".format(source_sha256, pipeline_version, config_hash, converter_version)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


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
    timeout_seconds: int = 0,
) -> Dict[str, Any]:
    """执行单个 Job, 返回 ``{"status": ..., "manifest": ...}``。

    通过乐观锁 :meth:`JobRepository.claim` 抢占, 防双 Worker 同时执行。
    ``timeout_seconds > 0`` 时, Pipeline 执行超过 wall-clock 即中断标记 FAILED(§46)。
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

        # §48 计算缓存指纹并写入
        source_model = db.get(Model, job.source_model_id)
        fingerprint = compute_fingerprint(
            source_sha256=(source_model.sha256 if source_model else ""),
            pipeline_version=job.pipeline_version,
            config=dict(job.config),
        )
        job.conversion_fingerprint = fingerprint
        db.commit()

        workspace = Path(workspace_root) / job_id
        workspace.mkdir(parents=True, exist_ok=True)
        source = fetch_source_model(db, job.source_model_id, workspace)
        store = artifact_store or LocalArtifactStore(str(workspace / "artifacts"))

        event_sink = CollectingEventSink()
        ctx = new_context(job_id=job_id, workspace=str(workspace), source_model=str(source),
                          config=dict(job.config), event_sink=event_sink, artifact_store=store)
        pipeline = pipeline_factory(str(source), dict(job.config))

        # §46 执行 timeout: signal.alarm 实现 wall-clock 中断(仅主线程, Unix only)
        timed_out = False
        if timeout_seconds and timeout_seconds > 0:
            import signal

            def _timeout_handler(signum, frame):
                raise TimeoutError("Job 执行超时({}s)".format(timeout_seconds))

            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(int(timeout_seconds))
            try:
                job_obj, manifest, _ = run_pipeline(ctx, pipeline)
            except TimeoutError:
                timed_out = True
                job_obj = None
                manifest = {"error": "Job 执行超时({}s)".format(timeout_seconds), "artifacts": {}}
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        else:
            job_obj, manifest, _ = run_pipeline(ctx, pipeline)

        # 落 Stage 行(10 阶段) — timeout 时跳过, 已无 stage 对象
        if job_obj is not None:
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
            size = 0
            storage_key = path
            if p.is_file():
                # local store: manifest 值为宿主机/共享卷绝对路径
                size = p.stat().st_size
            elif store is not None:
                # minio store: manifest 值为对象 key, 大小走对象元数据查询
                stat = getattr(store, "_stat", None)
                if callable(stat):
                    try:
                        obj = stat(path)
                        size = int(getattr(obj, "size", 0) or 0)
                    except Exception:
                        size = 0
            artifact_repo.create(
                job_id=job_id, filename=p.name, storage_key=storage_key, artifact_type=atype,
                size=size,
            )
        # 落 Event 行(sequence 严格递增)
        event_repo = JobEventRepository(db)
        for ev in event_sink.events:
            event_repo.append(job_id, ev.event, stage=ev.stage, message=ev.message,
                              progress=ev.progress, data=ev.data)

        final_status = "SUCCESS" if (job_obj and job_obj.status.value == "SUCCESS") else "FAILED"
        finished_at = parse_engine_ts(job_obj.finished_at) if job_obj else datetime.utcnow()
        job_repo.update_status(
            job_id, final_status,
            finished_at=finished_at,
            error_message=manifest.get("error", ""),
        )
        return {"status": final_status, "manifest": manifest}
    finally:
        db.close()
