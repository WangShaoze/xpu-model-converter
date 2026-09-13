# -*- coding: utf-8 -*-
"""仓储层(Phase 3 §56): 任务状态持久化。"""
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from xpu_platform.db.models.job import ConversionJob
from xpu_platform.db.models.stage import JobStage


class JobRepository:
    """conversion_jobs + job_stages 的读写。"""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create(
        self,
        project_id: str,
        source_model_id: str,
        config: Dict[str, Any],
        pipeline_version: str = "10-stage",
    ) -> ConversionJob:
        job = ConversionJob(
            project_id=project_id,
            source_model_id=source_model_id,
            status="CREATED",
            pipeline_version=pipeline_version,
            config=dict(config),
        )
        self._session.add(job)
        self._session.commit()
        self._session.refresh(job)
        return job

    def get(self, job_id: str) -> Optional[ConversionJob]:
        return self._session.get(ConversionJob, job_id)

    def list_by_project(self, project_id: str, limit: int = 100) -> List[ConversionJob]:
        stmt = (
            select(ConversionJob)
            .where(ConversionJob.project_id == project_id)
            .order_by(ConversionJob.created_at.desc())
            .limit(limit)
        )
        return list(self._session.scalars(stmt))

    def update_status(self, job_id: str, status: str, **fields: Any) -> Optional[ConversionJob]:
        job = self.get(job_id)
        if job is None:
            return None
        job.status = status
        for key, value in fields.items():
            setattr(job, key, value)
        self._session.commit()
        self._session.refresh(job)
        return job

    def add_stage(self, job_id: str, stage_name: str, stage_order: int) -> JobStage:
        stage = JobStage(job_id=job_id, stage_name=stage_name, stage_order=stage_order, status="PENDING")
        self._session.add(stage)
        self._session.commit()
        self._session.refresh(stage)
        return stage

    def update_stage(
        self,
        stage_id: str,
        *,
        status: Optional[str] = None,
        progress: Optional[int] = None,
        error_message: str = "",
        duration_ms: int = 0,
    ) -> Optional[JobStage]:
        stage = self._session.get(JobStage, stage_id)
        if stage is None:
            return None
        if status is not None:
            stage.status = status
        if progress is not None:
            stage.progress = progress
        if error_message:
            stage.error_message = error_message
        if duration_ms:
            stage.duration_ms = duration_ms
        self._session.commit()
        self._session.refresh(stage)
        return stage

    def list_stages(self, job_id: str) -> List[JobStage]:
        stmt = (
            select(JobStage)
            .where(JobStage.job_id == job_id)
            .order_by(JobStage.stage_order)
        )
        return list(self._session.scalars(stmt))
