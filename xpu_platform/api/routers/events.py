# -*- coding: utf-8 -*-
"""SSE 事件流(Phase 5 §28): GET /jobs/{job_id}/events。

事件持久化在 job_events(sequence 递增), 断线恢复用 ``Last-Event-ID`` 从该
sequence 之后回放。流在 Job 进入终态且无新事件后关闭。
"""
import asyncio
import json
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from xpu_platform.api.dependencies import get_current_user, get_db
from xpu_platform.db.models.job import ConversionJob
from xpu_platform.db.repositories import JobEventRepository

router = APIRouter(prefix="/jobs/{job_id}/events", tags=["jobs"])

_TERMINAL = {"SUCCESS", "FAILED", "CANCELLED"}
_POLL_SECONDS = 0.3


def _sse(event: str, sequence: int, data: dict) -> str:
    payload = {"sequence": sequence, "event": event, **data}
    return "event: {}\nid: {}\ndata: {}\n\n".format(event, sequence, json.dumps(payload, ensure_ascii=False))


def _job_status(db: Session, job_id: str) -> str:
    """纯列查询返回状态字符串, 避免触碰 ORM 实例触发过期刷新。"""
    row = db.query(ConversionJob.status).filter(ConversionJob.id == job_id).first()
    return row[0] if row else ""


async def _event_stream(job_id: str, session_factory, last_event_id: Optional[int]):
    """轮询 job_events, 从 last_event_id 之后回放; 每次轮询独立 session 免 detached。"""
    sequence = last_event_id or 0
    while True:
        db: Session = session_factory()
        try:
            repo = JobEventRepository(db)
            events = repo.list_after(job_id, sequence)
            for ev in events:
                yield _sse(ev.event, ev.sequence, {
                    "job_id": ev.job_id, "stage": ev.stage, "message": ev.message,
                    "progress": ev.progress, "data": ev.data,
                })
                sequence = ev.sequence
            status = _job_status(db, job_id)
            done = status in _TERMINAL and not repo.list_after(job_id, sequence)
        finally:
            db.close()
        if done:
            break
        await asyncio.sleep(_POLL_SECONDS)


@router.get("")
async def job_events(
    job_id: str,
    request: Request,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
    last_event_id: Optional[str] = Header(default=None, alias="Last-Event-ID"),
):
    del db
    del current_user
    session_factory = request.app.state.session_factory
    check_db: Session = session_factory()
    try:
        job = check_db.get(ConversionJob, job_id)
    finally:
        check_db.close()
    if job is None:
        raise HTTPException(404, detail={"code": "JOB_NOT_FOUND", "message": "任务不存在"})
    parsed: Optional[int] = None
    if last_event_id:
        try:
            parsed = int(last_event_id)
        except ValueError:
            parsed = None
    return StreamingResponse(
        _event_stream(job_id, session_factory, parsed),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )