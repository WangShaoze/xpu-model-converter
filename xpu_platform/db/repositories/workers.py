# -*- coding: utf-8 -*-
"""Worker 仓储(Phase 4 §23): 注册/心跳/就绪列表, 供 Scheduler 调度。"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from xpu_platform.db.models.worker import Worker


class WorkerRepository:
    """workers 表读写: 设备指纹声明 + 心跳保活 + 就绪查询。"""

    def __init__(self, session: Session) -> None:
        self._session = session

    def get(self, worker_id: str) -> Optional[Worker]:
        return self._session.get(Worker, worker_id)

    def upsert(self, fields: Dict[str, Any]) -> Worker:
        """按 worker_id 注册或刷新 Worker(设备指纹幂等写入)。"""
        worker = self.get(fields.get("worker_id", ""))
        if worker is None:
            worker = Worker(worker_id=fields.get("worker_id", ""))
            self._session.add(worker)
        for key, value in fields.items():
            if key == "worker_id":
                continue
            setattr(worker, key, value)
        worker.last_heartbeat = datetime.utcnow()
        self._session.commit()
        self._session.refresh(worker)
        return worker

    def heartbeat(self, worker_id: str, status: str = "READY") -> Optional[Worker]:
        """心跳: 刷新 last_heartbeat 与当前状态(READY/BUSY/OFFLINE)。"""
        worker = self.get(worker_id)
        if worker is None:
            return None
        worker.status = status
        worker.last_heartbeat = datetime.utcnow()
        self._session.commit()
        self._session.refresh(worker)
        return worker

    def mark_offline(self, worker_id: str) -> Optional[Worker]:
        return self.heartbeat(worker_id, status="OFFLINE")

    def list_ready(self, device_type: str = "xpu") -> List[Worker]:
        """§23: 只有 device_type=xpu 且 status=READY 的 Worker 可被调度。"""
        stmt = (
            select(Worker)
            .where(Worker.device_type == device_type, Worker.status == "READY")
            .order_by(Worker.worker_id)
        )
        return list(self._session.scalars(stmt))
