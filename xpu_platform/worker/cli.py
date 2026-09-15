# -*- coding: utf-8 -*-
"""Worker CLI(Phase 4 §57): 启动 Worker 进程, 轮询队列执行转换 Job。

用法::

    xpu-worker --worker-id xpu-worker-01 --chip KUNLUNXIN --device-id 0
    xpu-worker --queue redis --redis-url redis://localhost:6379/0

Worker 只执行, 不处理登录/权限/HTTP(§22)。
"""
import argparse
import os
import socket
import time
from pathlib import Path
from typing import Any, Dict

from sqlalchemy import text

from xpu_platform.db.models.base import Base
from xpu_platform.db.session import create_engine_from_url, create_session_factory
from xpu_platform.worker.executor import WorkerRuntime
from xpu_platform.worker.queue import InMemoryJobQueue, RedisJobQueue
from xpu_platform.worker.scheduler import WorkerScheduler


def _wait_for_database(database_url: str, timeout: float = 30.0) -> None:
    """与 API 相同的 DB 就绪等待(Compose 中 worker depends_on api, 这里再兜底一次)。"""
    deadline = time.time() + timeout
    engine = create_engine_from_url(database_url)
    last_err = None
    while time.time() < deadline:
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return
        except Exception as err:
            last_err = err
            time.sleep(1.0)
    raise SystemExit("等待数据库就绪超时: {}".format(last_err))


def build_pipeline(source: str, config: Dict[str, Any]):
    """按 Job config 构造 ConversionPipeline(§22: Worker 内执行 Engine)。"""
    from xpu_converter.cli.common import DEFAULT_HARDWARE
    from xpu_converter.pipeline import ConversionPipeline

    shape = config.get("input_shape") or config.get("input_size")
    return ConversionPipeline(
        model_path=source,
        output_dir=config.get("output_dir", "./output"),
        model_type=config.get("model_type"),
        hardware=config.get("hardware") or DEFAULT_HARDWARE,
        precision=config.get("precision"),
        input_shape=shape,
        runtime=config.get("runtime") or "detection",
        version=config.get("version", "v1.0"),
        package_name=config.get("package_name"),
        validation_dataset=config.get("validation_dataset"),
        validation_enabled=bool(config.get("validation_enabled", True)),
        benchmark_iterations=int(config.get("benchmark_iterations", 50) or 50),
        export_docker=bool(config.get("export_docker", True)),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="xpu-worker", description="XPU 转换 Worker")
    parser.add_argument("--worker-id", default=os.environ.get("XPU_WORKER_ID", ""),
                        help="Worker 唯一标识(缺省: xpu-worker-<hostname>)")
    parser.add_argument("--hostname", default=socket.gethostname())
    parser.add_argument("--device-type", default="xpu", choices=["xpu", "cpu"])
    parser.add_argument("--chip", default=os.environ.get("XPU_CHIP", ""), help="如 KUNLUNXIN")
    parser.add_argument("--device-id", default=os.environ.get("XPU_DEVICE_ID", "0"))
    parser.add_argument("--sdk-version", default=os.environ.get("XPU_SDK_VERSION", ""))
    parser.add_argument("--driver-version", default=os.environ.get("XPU_DRIVER_VERSION", ""))
    parser.add_argument("--queue",
                        default=os.environ.get("XPU_QUEUE") or os.environ.get("QUEUE_BACKEND", "memory"),
                        choices=["memory", "redis"])
    parser.add_argument("--redis-url", default=os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"),
                        help="缺省沿用 create_session_factory 默认(本地 SQLite)")
    parser.add_argument("--workspace", default=os.environ.get("WORKSPACE_ROOT")
                        or os.environ.get("XPU_WORKSPACE", "./worker-workspace"))
    parser.add_argument("--storage-backend",
                        default=os.environ.get("STORAGE_BACKEND", "local"),
                        choices=["local", "minio", "s3"])
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-iterations", type=int, default=None,
                        help="处理 N 个 Job 后退出(测试用); 缺省持续运行")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    worker_id = args.worker_id or "xpu-worker-{}".format(args.hostname)
    worker = WorkerRuntime(
        worker_id=worker_id, hostname=args.hostname, device_type=args.device_type,
        chip=args.chip, device_id=args.device_id,
        sdk_version=args.sdk_version, driver_version=args.driver_version,
    )
    if args.queue == "redis":
        queue = _redis_queue(args.redis_url)
    else:
        queue = InMemoryJobQueue()

    session_factory = create_session_factory(args.database_url)
    # 本地 SQLite 开发库: 未迁移时自动建表; Postgres 由 API 侧 Alembic 迁移, Worker 仅等待就绪
    if not args.database_url or args.database_url.startswith("sqlite"):
        Base.metadata.create_all(create_engine_from_url(args.database_url))
    else:
        _wait_for_database(args.database_url)
    Path(args.workspace).mkdir(parents=True, exist_ok=True)

    # 产物存储: minio(Compose/生产) 或本地共享卷; 由 STORAGE_BACKEND 统一配置
    artifact_store = None
    if args.storage_backend in ("minio", "s3"):
        from xpu_platform.storage_factory import build_artifact_store

        artifact_store = build_artifact_store()

    scheduler = WorkerScheduler(
        queue=queue, session_factory=session_factory, pipeline_factory=build_pipeline,
        worker=worker, workspace_root=args.workspace, artifact_store=artifact_store,
        max_retries=args.max_retries,
    )
    print("Worker 启动: id={} chip={} device={} queue={} workspace={}".format(
        worker_id, args.chip or "(auto)", args.device_id, args.queue, args.workspace))
    processed = scheduler.run(poll_interval=args.poll_interval, max_iterations=args.max_iterations)
    print("Worker 退出: 共处理 {} 个 Job".format(processed))
    return 0


def _redis_queue(redis_url: str) -> RedisJobQueue:
    try:
        import redis as _redis
    except ImportError as err:
        raise SystemExit(
            "使用 redis 队列需要安装依赖: pip install 'redis>=5.0'(或改用 --queue memory)"
        ) from err
    return RedisJobQueue(client=_redis.Redis.from_url(redis_url))


if __name__ == "__main__":
    raise SystemExit(main())
