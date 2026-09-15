# -*- coding: utf-8 -*-
"""产物路由(Phase 5 §29 / Phase 7): 列表 / 详情 / 下载(local 卷或 MinIO 流式)。"""
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