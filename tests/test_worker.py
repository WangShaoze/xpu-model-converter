# -*- coding: utf-8 -*-
"""Phase 4 Worker/Scheduler 测试: 成功链路 / 崩溃重试 / 重试上限失败 / 乐观锁抢占 / 心跳。"""
import tempfile
import unittest
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xpu_platform.db.models import Base, ConversionJob, JobEvent, Model, Project, User, Worker
from xpu_platform.db.repositories import JobRepository, WorkerRepository
from xpu_platform.worker import InMemoryJobQueue, WorkerRuntime, WorkerScheduler


def _make_session():
    engine = create_engine("sqlite+pysqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False)()


def _make_job(session, source_path: str) -> ConversionJob:
    user = User(username="alice", email="a@x.com", password_hash="h")
    session.add(user)
    session.commit()
    project = Project(owner_id=user.id, name="demo", description="d")
    session.add(project)
    session.commit()
    model = Model(project_id=project.id, name="best.pt", filename="best.pt",
                  framework="pytorch", model_type="yolov10", status="READY",
                  storage_key=source_path, sha256="ab" * 32)
    session.add(model)
    session.commit()
    return JobRepository(session).create(project.id, model.id, config={"task": "detection"})


class _OkResult:
    """与 run_pipeline 兼容的最小成功结果: 10 步全 ok。"""

    model_path = "best.pt"
    onnx_path = ""
    optimized_onnx_path = ""
    artifact = None
    package_zip = ""
    accuracy = None
    benchmark = None
    degraded = False
    steps = [{"index": i, "title": "s{}".format(i), "ok": True, "detail": "ok"} for i in range(1, 11)]


class _OkPipeline:
    STEP_TITLES = ["s{}".format(i) for i in range(1, 11)]

    def __init__(self):
        self.result = _OkResult()

    def run(self):
        return self.result


class WorkerRuntimeTest(unittest.TestCase):
    def test_heartbeat_upsert_and_ready_filter(self):
        session = _make_session()
        repo = WorkerRepository(session)
        worker = WorkerRuntime(worker_id="xpu-worker-01", hostname="node1", chip="KUNLUNXIN")
        repo.upsert(worker.to_row())
        repo.upsert(WorkerRuntime(worker_id="cpu-worker-01", device_type="cpu").to_row())
        ready = [w.worker_id for w in repo.list_ready()]
        self.assertIn("xpu-worker-01", ready)
        self.assertNotIn("cpu-worker-01", ready)
        # 心跳 BUSY 后不再可调度
        repo.heartbeat("xpu-worker-01", "BUSY")
        self.assertNotIn("xpu-worker-01", [w.worker_id for w in repo.list_ready()])
        repo.mark_offline("xpu-worker-01")
        self.assertEqual(repo.get("xpu-worker-01").status, "OFFLINE")
        session.close()


class SchedulerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tmp.name) / "ws"
        self.workspace.mkdir()
        self.source = Path(self.tmp.name) / "best.pt"
        self.source.write_bytes(b"fake-weight-bytes")
        self.session_factory = None
        self.engine = create_engine("sqlite+pysqlite:///:memory:", connect_args={"check_same_thread": False})
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(bind=self.engine, autoflush=False)
        self.queue = InMemoryJobQueue()
        self.worker = WorkerRuntime(worker_id="xpu-worker-01", hostname="node1", chip="KUNLUNXIN")

    def tearDown(self):
        self.engine.dispose()
        self.tmp.cleanup()

    def _scheduler(self, pipeline_factory=None, max_retries=2):
        return WorkerScheduler(
            queue=self.queue, session_factory=self.session_factory,
            pipeline_factory=pipeline_factory or (lambda src, cfg: _OkPipeline()),
            worker=self.worker, workspace_root=str(self.workspace), max_retries=max_retries,
        )

    def test_success_path_persists_stages_and_events(self):
        db = self.session_factory()
        job = _make_job(db, str(self.source))
        db.close()
        self.queue.enqueue(job.id)

        result = self._scheduler().poll_once()
        self.assertEqual(result["status"], "SUCCESS")

        db = self.session_factory()
        row = db.get(ConversionJob, job.id)
        self.assertEqual(row.status, "SUCCESS")
        self.assertEqual(row.worker_id, "xpu-worker-01")
        stages = JobRepository(db).list_stages(job.id)
        self.assertEqual(len(stages), 10)
        self.assertTrue(all(s.status == "SUCCESS" for s in stages))
        events = [e.event for e in db.query(JobEvent).filter(JobEvent.job_id == job.id).order_by(JobEvent.sequence)]
        self.assertIn("job_finished", events)
        # Worker 心跳已落库且回到 READY
        self.assertEqual(db.get(Worker, "xpu-worker-01").status, "READY")
        db.close()
        # 队列已 ack
        self.assertEqual(self.queue.pending(), 0)

    def test_no_job_returns_none(self):
        self.assertIsNone(self._scheduler().poll_once())

    def test_crash_retries_then_succeeds(self):
        calls = {"n": 0}

        def factory(src, cfg):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("boom first time")
            return _OkPipeline()

        db = self.session_factory()
        job = _make_job(db, str(self.source))
        db.close()
        self.queue.enqueue(job.id)

        scheduler = self._scheduler(factory)
        first = scheduler.poll_once()
        self.assertEqual(first["status"], "RETRY")
        # 崩溃后 Job 已重置为 QUEUED 且被 requeue
        db = self.session_factory()
        self.assertEqual(db.get(ConversionJob, job.id).status, "QUEUED")
        db.close()
        second = scheduler.poll_once()
        self.assertEqual(second["status"], "SUCCESS")

    def test_crash_over_max_retries_marks_failed(self):
        def factory(src, cfg):
            raise RuntimeError("always boom")

        db = self.session_factory()
        job = _make_job(db, str(self.source))
        db.close()
        self.queue.enqueue(job.id)

        scheduler = self._scheduler(factory, max_retries=1)
        self.assertEqual(scheduler.poll_once()["status"], "RETRY")
        self.assertEqual(scheduler.poll_once()["status"], "FAILED")
        db = self.session_factory()
        row = db.get(ConversionJob, job.id)
        self.assertEqual(row.status, "FAILED")
        self.assertIn("always boom", row.error_message)
        db.close()
        self.assertEqual(self.queue.pending(), 0)

    def test_claim_is_exclusive(self):
        db = self.session_factory()
        job = _make_job(db, str(self.source))
        repo = JobRepository(db)
        first = repo.claim(job.id, from_status=["CREATED", "QUEUED", "FAILED"])
        second = repo.claim(job.id, from_status=["CREATED", "QUEUED", "FAILED"])
        self.assertIsNotNone(first)
        self.assertIsNone(second)  # 已被置 RUNNING, 第二个 Worker 抢不到
        db.close()


class _FakeRedis:
    """最小 redis-py 替身: 内部存 bytes, 复现真实客户端的 bytes 返回。"""

    def __init__(self):
        self.lists = {}
        self.sets = {}

    def lpush(self, key, value):
        self.lists.setdefault(key, []).insert(0, value.encode() if isinstance(value, str) else value)

    def rpop(self, key):
        values = self.lists.get(key)
        return values.pop() if values else None

    def brpop(self, keys, timeout=0):
        for key in keys:
            value = self.rpop(key)
            if value is not None:
                return key, value
        return None

    def sadd(self, key, *members):
        bucket = self.sets.setdefault(key, set())
        for member in members:
            bucket.add(member)
        return 1

    def srem(self, key, *members):
        bucket = self.sets.get(key, set())
        for member in members:
            bucket.discard(member)
        return 1

    def llen(self, key):
        return len(self.lists.get(key, []))


class RedisJobQueueTest(unittest.TestCase):
    def test_dequeue_decodes_bytes_job_id(self):
        # Phase 7: redis-py 默认返回 bytes, 不转 str 会导致 Postgres bytea 与 varchar 比较报错
        from xpu_platform.worker.queue import RedisJobQueue

        fake = _FakeRedis()
        queue = RedisJobQueue(client=fake)
        queue.enqueue("job-uuid-0123")

        job_id = queue.dequeue()
        self.assertEqual(job_id, "job-uuid-0123")
        self.assertIsInstance(job_id, str)
        # 处理集合中也应是 str(ack/requeue 后续按 str 操作)
        self.assertIn("job-uuid-0123", fake.sets[queue.processing_key])

    def test_blocking_dequeue_also_returns_str(self):
        from xpu_platform.worker.queue import RedisJobQueue

        fake = _FakeRedis()
        queue = RedisJobQueue(client=fake)
        queue.enqueue("job-uuid-4567")
        self.assertEqual(queue.dequeue(timeout=1), "job-uuid-4567")

    def test_empty_queue_returns_none(self):
        from xpu_platform.worker.queue import RedisJobQueue

        self.assertIsNone(RedisJobQueue(client=_FakeRedis()).dequeue())
        self.assertIsNone(RedisJobQueue(client=_FakeRedis()).dequeue(timeout=0))


if __name__ == "__main__":
    unittest.main()
