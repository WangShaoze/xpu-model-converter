# -*- coding: utf-8 -*-
"""Worker 调度器(Phase 4 §57 第四轮: Redis + Worker + Scheduler)。

Scheduler 职责: 从 Queue 取 Job → 交给 :func:`run_job` 执行 → ack/requeue,
并维护 Worker 心跳与重试。API 不直接执行 Pipeline(§21)。

重试语义(§31): 执行器抛异常(而非 Job 自身失败)时, 先重置 Job 为 QUEUED
再 requeue, 避免乐观锁 claim 因状态 RUNNING 卡死; 超过 max_retries 则标记 FAILED。
"""
from typing import Any, Callable, Dict, Optional

from xpu_platform.db.repositories import JobRepository, WorkerRepository
from xpu_platform.worker.executor import WorkerRuntime, run_job
from xpu_platform.worker.queue import JobQueue


class WorkerScheduler:
    """单 Worker 调度循环: 轮询队列 → 执行 → ack/requeue → 心跳。"""

    def __init__(
        self,
        queue: JobQueue,
        session_factory,
        pipeline_factory: Callable[[str, Dict[str, Any]], Any],
        worker: WorkerRuntime,
        workspace_root: str,
        artifact_store=None,
        max_retries: int = 2,
    ) -> None:
        self.queue = queue
        self.session_factory = session_factory
        self.pipeline_factory = pipeline_factory
        self.worker = worker
        self.workspace_root = workspace_root
        self.artifact_store = artifact_store
        self.max_retries = max_retries
        self._retries: Dict[str, int] = {}

    # ---- 心跳 ----
    def heartbeat(self, status: str) -> None:
        db = self.session_factory()
        try:
            WorkerRepository(db).upsert(self.worker.to_row())
            WorkerRepository(db).heartbeat(self.worker.worker_id, status)
        finally:
            db.close()

    def register(self) -> None:
        """启动时注册设备指纹(§23)并置 READY。"""
        db = self.session_factory()
        try:
            WorkerRepository(db).upsert(self.worker.to_row())
        finally:
            db.close()

    # ---- 单次轮询 ----
    def poll_once(self) -> Optional[Dict[str, Any]]:
        """取一个 Job 并执行。无任务返回 None; 返回执行摘要。"""
        job_id = self.queue.dequeue()
        if job_id is None:
            return None
        self.worker.status = "BUSY"
        self.heartbeat("BUSY")
        try:
            result = run_job(
                job_id, self.session_factory, self.pipeline_factory, self.worker,
                self.workspace_root, self.artifact_store,
            )
            self.queue.ack(job_id)
            self._retries.pop(job_id, None)
            return result
        except Exception as err:  # 执行器崩溃(非 Job 自身失败)
            attempts = self._retries.get(job_id, 0) + 1
            if attempts <= self.max_retries:
                self._retries[job_id] = attempts
                self._reset_for_retry(job_id, str(err))
                self.queue.requeue(job_id)
                return {"status": "RETRY", "attempt": attempts, "error": str(err)}
            self._retries.pop(job_id, None)
            self._mark_failed(job_id, str(err))
            self.queue.ack(job_id)
            return {"status": "FAILED", "error": str(err)}
        finally:
            self.worker.status = "READY"
            self.heartbeat("READY")

    # ---- 长循环 ----
    def run(self, poll_interval: float = 1.0, max_iterations: Optional[int] = None) -> int:
        """轮询执行。

        ``max_iterations`` 缺省时无限运行(队列空则 sleep); 显式给出时, 处理满
        N 个或队列为空即退出(供 CLI 测试/一次性执行)。
        """
        import time

        self.register()
        processed = 0
        while max_iterations is None or processed < max_iterations:
            try:
                result = self.poll_once()
            except KeyboardInterrupt:
                break
            if result is not None:
                processed += 1
            elif max_iterations is not None:
                break  # 显式次数: 队列空即结束, 避免空转
            else:
                time.sleep(poll_interval)
        return processed

    # ---- 内部: 重试 / 失败落库 ----
    def _reset_for_retry(self, job_id: str, error: str) -> None:
        db = self.session_factory()
        try:
            JobRepository(db).update_status(job_id, "QUEUED", error_message=error)
        finally:
            db.close()

    def _mark_failed(self, job_id: str, error: str) -> None:
        db = self.session_factory()
        try:
            JobRepository(db).update_status(job_id, "FAILED", error_message=error)
        finally:
            db.close()
