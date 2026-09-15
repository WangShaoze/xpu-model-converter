# -*- coding: utf-8 -*-
"""Worker 状态路由(Phase 5 / §23): 查询真实 XPU Worker 声明。"""
from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from xpu_platform.api.dependencies import get_current_user, get_db
from xpu_platform.api.schemas import WorkerOut
from xpu_platform.db.repositories import WorkerRepository

router = APIRouter(prefix="/workers", tags=["workers"])


@router.get("", response_model=List[WorkerOut])
def list_workers(device_type: str = "xpu", db: Session = Depends(get_db),
                 current_user=Depends(get_current_user)):
    del current_user  # 需登录, 不泄露 worker 详情的部署范围
    return [WorkerOut.model_validate(w) for w in WorkerRepository(db).list_ready(device_type)]