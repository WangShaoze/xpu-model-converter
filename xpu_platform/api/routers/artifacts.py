# -*- coding: utf-8 -*-
"""产物路由(Phase 5 §29): 列表 / 详情 / 下载(读存储, 不整包进内存)。"""
import mimetypes
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from xpu_platform.api.dependencies import get_current_user, get_db
from xpu_platform.api.schemas import ArtifactOut
from xpu_platform.db.models.artifact import Artifact
from xpu_platform.db.models.job import ConversionJob
from xpu_platform.db.repositories import AuditLogRepository, ProjectRepository

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
    storage = Path(artifact.storage_key)
    if not storage.is_file():
        raise HTTPException(404, detail={"code": "STORAGE_MISSING", "message": "存储文件缺失"})
    AuditLogRepository(db).record(current_user.id, "ARTIFACT_DOWNLOADED",
                                  resource_type="artifact", resource_id=artifact.id,
                                  metadata={"filename": artifact.filename})
    media_type = artifact.mime_type or mimetypes.guess_type(artifact.filename)[0] or "application/octet-stream"
    return FileResponse(path=storage, media_type=media_type, filename=artifact.filename)