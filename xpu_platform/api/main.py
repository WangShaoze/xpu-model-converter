# -*- coding: utf-8 -*-
"""FastAPI 应用入口(Phase 5 §58): 统一前缀 /api/v1, 装配中间件与路由。

启动::

    uvicorn xpu_platform.api.main:app --reload
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import os
import time

from sqlalchemy import text

from xpu_platform.api.config import get_settings
from xpu_platform.api.middleware import (
    RequestIDMiddleware,
    http_exception_handler,
    unified_error_handler,
)
from xpu_platform.api.routers import (
    artifacts,
    auth,
    events,
    jobs,
    models,
    projects,
    workers,
)
from xpu_platform.db.models.base import Base
from xpu_platform.db.session import create_engine_from_url, create_session_factory
from xpu_platform.api.ratelimit import LoginGuard
from xpu_platform.worker.queue import InMemoryJobQueue, JobQueue

logger = logging.getLogger("xpu.api")
settings = get_settings()


def _run_alembic_upgrade(database_url: str) -> None:
    """对 Postgres 执行 alembic upgrade head(§15); 迁移脚本随镜像一起发布。"""
    from alembic import command
    from alembic.config import Config

    os.environ["DATABASE_URL"] = database_url  # migrations/env.py 从该变量取连接串
    cfg = Config()
    script_location = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "db", "migrations",
    )
    cfg.set_main_option("script_location", script_location)
    command.upgrade(cfg, "head")


def _wait_for_database(database_url: str, timeout: float = 30.0) -> None:
    """容器编排下 DB 可能晚于 API 就绪, 启动前短轮询等待。"""
    deadline = time.time() + timeout
    engine = create_engine_from_url(database_url)
    last_err = None
    while time.time() < deadline:
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return
        except Exception as err:  # 连接拒绝/初始化中
            last_err = err
            time.sleep(1.0)
    raise RuntimeError("等待数据库就绪超时: {}".format(last_err))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if settings.database_url.startswith("sqlite"):
        # 本地 SQLite 开发库: 未迁移时自动建表
        Base.metadata.create_all(create_engine_from_url(settings.database_url))
    else:
        # Docker Compose/生产 Postgres: 等待就绪后执行 Alembic 迁移(§15)
        _wait_for_database(settings.database_url)
        _run_alembic_upgrade(settings.database_url)
    yield


def build_job_queue() -> JobQueue:
    """按 QUEUE_BACKEND 构造队列: redis(Compose/生产) 或 memory(本地/测试)。

    Redis 模式下短轮询等待 Redis 就绪(容器启动顺序), 30s 不可达则快速失败。
    """
    if settings.queue_backend == "redis":
        import redis as _redis

        from xpu_platform.worker.queue import RedisJobQueue

        deadline = time.time() + 30.0
        last_err = None
        while time.time() < deadline:
            try:
                client = _redis.Redis.from_url(settings.redis_url, socket_connect_timeout=5)
                client.ping()
                return RedisJobQueue(client=client)
            except Exception as err:
                last_err = err
                time.sleep(1.0)
        raise RuntimeError("等待 Redis 就绪超时: {}".format(last_err))
    return InMemoryJobQueue()


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="模型转换 Web SaaS API(PyTorch/Paddle → ONNX → 昆仑芯 XPU → Docker 交付包)",
    docs_url="/docs",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.add_middleware(RequestIDMiddleware)

app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(Exception, unified_error_handler)

# 任务队列: 按 QUEUE_BACKEND 选择 redis(Docker Compose/生产) 或内存队列(本地/测试)
app.state.job_queue = build_job_queue()
# /auth/* 防暴力破解: IP 限流 + 账号失败锁定(多副本生产应换 Redis 实现)
app.state.login_guard = LoginGuard(
    login_rate_per_minute=settings.login_rate_per_minute,
    login_max_failures=settings.login_max_failures,
    login_lock_minutes=settings.login_lock_minutes,
    register_rate_per_hour=settings.register_rate_per_hour,
)
# 数据库会话工厂: API 端点(+SSE 回放)统一从这里取, 测试可用同构 factory 替换
app.state.session_factory = create_session_factory(settings.database_url)

api = settings.api_prefix
app.include_router(auth.router, prefix=api)
app.include_router(projects.router, prefix=api)
app.include_router(models.router, prefix=api)
app.include_router(jobs.router, prefix=api)
app.include_router(artifacts.router, prefix=api)
app.include_router(workers.router, prefix=api)
app.include_router(events.router, prefix=api)


@app.get(api + "/health")
def health():
    return {"status": "ok"}