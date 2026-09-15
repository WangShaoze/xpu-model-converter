# -*- coding: utf-8 -*-
"""产物路由(Phase 5 §29 / Phase 7 / §47 安全下载): 列表 / 详情 / 下载。

§47 下载安全门禁: 禁止下载 FAILED / DEGRADED / UNVERIFIED 状态的产物。
"""
import mimetypes
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session

from xpu_platform.api.config import get_settings
from xpu_platform.api.dependencies import get_current_user, get_db
from xpu_platform.api.schemas import ArtifactOut
from xpu_platform.db.models.artifact import Artifact
from xpu_platform.db.models.job import ConversionJob
from xpu_platform.db.repositories import AuditLogRepository, ProjectRepository
from xpu_platform.storage_factory import build_artifact_store

router = APIRouter(prefix="/artifacts", tags=["artifacts"])

# §47 禁止下载的 Job 状态: 非 SUCCESS 一律拒绝
_DOWNLOAD_BLOCKED_STATUS = {"FAILED", "CANCELLED", "QUEUED", "RUNNING", "CREATED"}


def _owned_artifact(artifact_id: str, db: Session, owner_id: str) -> Artifact:
    artifact = db.get(Artifact, artifact_id)
    if artifact is None:
        raise HTTPException(404, detail={"code": "ARTIFACT_NOT_FOUND", "message": "产物不存在"})
    job = db.get(ConversionJob, artifact.job_id)
    if job is None:
        raise HTTPException(404, detail={"code": "ARTIFACT_NOT_FOUND", "message": "产物不存在"})
    if ProjectRepository(db).get_owned(owner_id, job.project_id) is None:
        raise HTTPException(404, detail={"code": "ARTIFACT_NOT_FOUND", "message": "产物不存在或无权访问"})
    return artifact


@router.get("/job/{job_id}", response_model=List[ArtifactOut])
def list_artifacts(job_id: str, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    job = db.get(ConversionJob, job_id)
    if job is None:
        raise HTTPException(404, detail={"code": "JOB_NOT_FOUND", "message": "任务不存在"})
    if ProjectRepository(db).get_owned(current_user.id, job.project_id) is None:
        raise HTTPException(404, detail={"code": "JOB_NOT_FOUND", "message": "任务不存在或无权访问"})
    rows = db.query(Artifact).filter(Artifact.job_id == job_id).order_by(Artifact.created_at)
    return [ArtifactOut.model_validate(a) for a in rows]


@router.get("/{artifact_id}", response_model=ArtifactOut)
def get_artifact(artifact_id: str, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    return ArtifactOut.model_validate(_owned_artifact(artifact_id, db, current_user.id))


@router.get("/{artifact_id}/download")
def download_artifact(artifact_id: str, db: Session = Depends(get_db),
                      current_user=Depends(get_current_user)):
    artifact = _owned_artifact(artifact_id, db, current_user.id)

    # §47 安全门禁: 检查所属 Job 状态, 非 SUCCESS 拒绝下载
    job = db.get(ConversionJob, artifact.job_id)
    if job is None or job.status in _DOWNLOAD_BLOCKED_STATUS:
        raise HTTPException(403, detail={
            "code": "DOWNLOAD_FORBIDDEN",
            "message": "产物未通过校验或任务未完成, 禁止下载(状态: {})".format(
                job.status if job else "UNKNOWN")})

    # §47 degraded 检查: config 中标记降级的产物禁止下载
    job_config = job.config or {}
    if job_config.get("degraded") or job_config.get("allow_degraded"):
        raise HTTPException(403, detail={
            "code": "DOWNLOAD_FORBIDDEN",
            "message": "产物为降级占位产物(DEGRADED), 禁止作为正式部署包下载"})

    settings = get_settings()
    storage = Path(artifact.storage_key)
    media_type = artifact.mime_type or mimetypes.guess_type(artifact.filename)[0] or "application/octet-stream"
    disposition = {"Content-Disposition": 'attachment; filename="{}"'.format(artifact.filename)}

    # 1) 本地/共享卷: storage_key 是文件绝对路径
    if storage.is_file():
        AuditLogRepository(db).record(current_user.id, "ARTIFACT_DOWNLOADED",
                                      resource_type="artifact", resource_id=artifact.id,
                                      metadata={"filename": artifact.filename, "backend": "local"})
        return FileResponse(path=storage, media_type=media_type, filename=artifact.filename)

    # 2) MinIO: storage_key 是对象 key, 流式回传(不整包进 API 内存)
    if settings.storage_backend in ("minio", "s3"):
        store = build_artifact_store(settings)
        if not store.exists(artifact.storage_key):
            raise HTTPException(404, detail={"code": "STORAGE_MISSING", "message": "存储对象缺失"})
        response = store.client.get_object(store.bucket, artifact.storage_key)

        def _iter_chunks():
            try:
                for chunk in response.stream(1 << 20):
                    yield chunk
            finally:
                response.close()
                response.release_conn()

        AuditLogRepository(db).record(current_user.id, "ARTIFACT_DOWNLOADED",
                                      resource_type="artifact", resource_id=artifact.id,
                                      metadata={"filename": artifact.filename, "backend": "minio"})
        return StreamingResponse(_iter_chunks(), media_type=media_type, headers=disposition)

    raise HTTPException(404, detail={"code": "STORAGE_MISSING", "message": "存储文件缺失"})