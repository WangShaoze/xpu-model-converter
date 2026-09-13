# -*- coding: utf-8 -*-
"""Artifact / JobEvent / AuditLog 仓储(Phase 3 §56)。"""
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from xpu_platform.db.models.artifact import Artifact
from xpu_platform.db.models.audit import AuditLog
from xpu_platform.db.models.event import JobEvent


class ArtifactRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create(
        self,
        job_id: str,
        filename: str,
        storage_key: str,
        *,
        stage: str = "",
        artifact_type: str = "",
        size: int = 0,
        sha256: str = "",
        mime_type: str = "application/octet-stream",
    ) -> Artifact:
        artifact = Artifact(
            job_id=job_id, stage=stage, artifact_type=artifact_type, filename=filename,
            storage_key=storage_key, size=size, sha256=sha256, mime_type=mime_type,
        )
        self._session.add(artifact)
        self._session.commit()
        self._session.refresh(artifact)
        return artifact

    def get(self, artifact_id: str) -> Optional[Artifact]:
        return self._session.get(Artifact, artifact_id)

    def list_by_job(self, job_id: str) -> List[Artifact]:
        stmt = select(Artifact).where(Artifact.job_id == job_id).order_by(Artifact.created_at)
        return list(self._session.scalars(stmt))


class JobEventRepository:
    """事件持久化(§9/§56): 同一 Job 内 sequence 严格递增。"""

    def __init__(self, session: Session) -> None:
        self._session = session

    def next_sequence(self, job_id: str) -> int:
        stmt = (
            select(JobEvent.sequence)
            .where(JobEvent.job_id == job_id)
            .order_by(JobEvent.sequence.desc())
            .limit(1)
        )
        last = self._session.execute(stmt).scalar()
        return int(last) + 1 if last else 1

    def append(
        self,
        job_id: str,
        event: str,
        *,
        stage: str = "",
        message: str = "",
        progress: Optional[int] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> JobEvent:
        row = JobEvent(
            job_id=job_id, stage=stage, event=event, message=message, progress=progress,
            sequence=self.next_sequence(job_id), data=dict(data or {}),
        )
        self._session.add(row)
        self._session.commit()
        self._session.refresh(row)
        return row

    def list_after(self, job_id: str, sequence: int = 0) -> List[JobEvent]:
        stmt = (
            select(JobEvent)
            .where(JobEvent.job_id == job_id, JobEvent.sequence > sequence)
            .order_by(JobEvent.sequence)
        )
        return list(self._session.scalars(stmt))


class AuditLogRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def record(
        self,
        user_id: str,
        action: str,
        *,
        resource_type: str = "",
        resource_id: str = "",
        ip: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditLog:
        log = AuditLog(
            user_id=user_id, action=action, resource_type=resource_type,
            resource_id=resource_id, ip=ip, meta=dict(metadata or {}),
        )
        self._session.add(log)
        self._session.commit()
        self._session.refresh(log)
        return log

    def list_by_user(self, user_id: str, limit: int = 100) -> List[AuditLog]:
        stmt = (
            select(AuditLog)
            .where(AuditLog.user_id == user_id)
            .order_by(AuditLog.timestamp.desc())
            .limit(limit)
        )
        return list(self._session.scalars(stmt))
