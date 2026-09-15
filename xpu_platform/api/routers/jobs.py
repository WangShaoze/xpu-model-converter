# -*- coding: utf-8 -*-
"""任务路由(Phase 5 §26-§28/§31/§32): 创建入队 / 查询 / retry / cancel / delete。"""
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from xpu_platform.api.dependencies import get_current_user, get_db, require_owned_project
from xpu_platform.api.schemas import JobCreate, JobOut, StageOut
from xpu_platform.db.models.job import ConversionJob
from xpu_platform.db.models.model import Model
from xpu_platform.db.repositories import (
    AuditLogRepository,
    JobEventRepository,
    JobRepository,
)

router = APIRouter(prefix="/jobs", tags=["jobs"])

_QUEUED_LIKE = {"CREATED", "QUEUED"}


def _enqueue(request: Request, job_id: str) -> None:
    queue = getattr(request.app.state, "job_queue", None)
    if queue is not None:
        queue.enqueue(job_id)


def _progress_from_stages(stages) -> int:
    if not stages:
        return 0
    return max(0, min(100, sum(s.progress for s in stages) // len(stages)))


def _to_out(job: ConversionJob, db: Session) -> JobOut:
    stages = JobRepository(db).list_stages(job.id)
    return JobOut(
        id=job.id, project_id=job.project_id, source_model_id=job.source_model_id,
        status=job.status, pipeline_version=job.pipeline_version,
        progress=_progress_from_stages(stages), worker_id=job.worker_id,
        error_message=job.error_message, created_at=job.created_at,
        queued_at=job.queued_at, started_at=job.started_at, finished_at=job.finished_at,
        stages=[StageOut(name=s.stage_name, status=s.status, progress=s.progress,
                         error_message=s.error_message) for s in stages],
    )


@router.post("", response_model=JobOut, status_code=201)
def create_job(body: JobCreate, request: Request, db: Session = Depends(get_db),
               current_user=Depends(get_current_user)):
    require_owned_project(body.project_id, db, current_user.id)
    model = db.get(Model, body.model_id)
    if model is None or model.project_id != body.project_id:
        raise HTTPException(404, detail={"code": "MODEL_NOT_FOUND", "message": "模型不存在"})
    if model.status != "READY":
        raise HTTPException(400, detail={"code": "MODEL_NOT_READY", "message": "模型未就绪"})

    config = dict(body.config)
    config.setdefault("model_type", model.model_type or "yolov10")
    job = JobRepository(db).create(body.project_id, body.model_id, config)
    _enqueue(request, job.id)  # API 只入队, Worker 执行(§21)
    JobRepository(db).update_status(job.id, "QUEUED", queued_at=None)
    AuditLogRepository(db).record(current_user.id, "JOB_CREATED",
                                  resource_type="conversion_job", resource_id=job.id,
                                  metadata={"model_id": body.model_id})
    return _to_out(job, db)


@router.get("", response_model=List[JobOut])
def list_jobs(project_id: str = "", limit: int = 100, db: Session = Depends(get_db),
              current_user=Depends(get_current_user)):
    repo = JobRepository(db)
    if project_id:
        require_owned_project(project_id, db, current_user.id)
        jobs = repo.list_by_project(project_id, limit)
    else:
        # 第一版: 返回当前用户 project 下 Job(遍历即可, 后续可按 user 建索引)
        from xpu_platform.db.repositories import ProjectRepository
        projects = ProjectRepository(db).list_by_owner(current_user.id)
        ids = {p.id for p in projects}
        jobs = [j for j in db.query(ConversionJob).order_by(ConversionJob.created_at.desc()).limit(limit * 5)
                if j.project_id in ids][:limit]
    return [_to_out(j, db) for j in jobs]


@router.get("/{job_id}", response_model=JobOut)
def get_job(job_id: str, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    job = _owned_job(job_id, db, current_user.id)
    return _to_out(job, db)


@router.post("/{job_id}/retry", response_model=JobOut)
def retry_job(job_id: str, request: Request, db: Session = Depends(get_db),
              current_user=Depends(get_current_user)):
    job = _owned_job(job_id, db, current_user.id)
    if job.status != "FAILED":
        raise HTTPException(400, detail={"code": "INVALID_STATE", "message": "仅 FAILED Job 可重试"})
    repo = JobRepository(db)
    repo.update_status(job_id, "QUEUED")
    _enqueue(request, job_id)
    AuditLogRepository(db).record(current_user.id, "JOB_RETRIED",
                                  resource_type="conversion_job", resource_id=job_id)
    return _to_out(db.get(ConversionJob, job_id), db)


@router.post("/{job_id}/cancel", response_model=JobOut)
def cancel_job(job_id: str, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    job = _owned_job(job_id, db, current_user.id)
    repo = JobRepository(db)
    if job.status in _QUEUED_LIKE or job.status == "FAILED":
        repo.update_status(job_id, "CANCELLED")
    elif job.status == "RUNNING":
        config = dict(job.config)
        # Worker 将定期检查 cancel_requested(§32)
        config["cancel_requested"] = True
        job.config = config
        db.commit()
    else:
        raise HTTPException(400, detail={"code": "INVALID_STATE", "message": "当前状态不可取消"})
    AuditLogRepository(db).record(current_user.id, "JOB_CANCELLED",
                                  resource_type="conversion_job", resource_id=job_id)
    return _to_out(db.get(ConversionJob, job_id), db)


@router.delete("/{job_id}", status_code=204)
def delete_job(job_id: str, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    job = _owned_job(job_id, db, current_user.id)
    db.delete(job)
    db.commit()
    return None


def _owned_job(job_id: str, db: Session, owner_id: str) -> ConversionJob:
    from xpu_platform.db.repositories import ProjectRepository
    job = db.get(ConversionJob, job_id)
    if job is None:
        raise HTTPException(404, detail={"code": "JOB_NOT_FOUND", "message": "任务不存在"})
    project = ProjectRepository(db).get_owned(owner_id, job.project_id)
    if project is None:
        raise HTTPException(404, detail={"code": "JOB_NOT_FOUND", "message": "任务不存在或无权访问"})
    return job