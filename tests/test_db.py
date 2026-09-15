# -*- coding: utf-8 -*-
"""Phase 3 PostgreSQL 持久化测试: 模型建表 / 仓储 CRUD / 事件递增(用 SQLite 内存库)。"""
import unittest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xpu_platform.db.models import (
    Artifact,
    AuditLog,
    Base,
    ConversionJob,
    JobEvent,
    JobStage,
    Model,
    Project,
    User,
    Worker,
)
from xpu_platform.db.repositories import (
    ArtifactRepository,
    AuditLogRepository,
    JobEventRepository,
    JobRepository,
)


def _make_session():
    engine = create_engine("sqlite+pysqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False)()


class DbModelTest(unittest.TestCase):
    def setUp(self):
        self.session = _make_session()

    def tearDown(self):
        self.session.close()

    def test_user_project_model_job_roundtrip(self):
        user = User(username="alice", email="a@x.com", password_hash="argon2hash")
        self.session.add(user)
        self.session.commit()
        project = Project(owner_id=user.id, name="demo", description="d")
        self.session.add(project)
        self.session.commit()
        model = Model(project_id=project.id, name="best.pt", filename="best.pt",
                      framework="pytorch", model_type="yolov10", status="READY",
                      storage_key="models/{}/best.pt".format(project.id), sha256="ab" * 32)
        self.session.add(model)
        self.session.commit()
        job = ConversionJob(project_id=project.id, source_model_id=model.id,
                            status="CREATED", config={"task": "detection", "input_size": [640, 640]})
        self.session.add(job)
        self.session.commit()
        self.session.refresh(job)
        self.assertEqual(job.status, "CREATED")
        self.assertEqual(job.config["input_size"], [640, 640])
        self.assertEqual(job.id, job.id)

    def test_job_stage_and_artifact(self):
        user = User(username="bob", email="b@x.com", password_hash="h")
        self.session.add(user)
        self.session.commit()
        project = Project(owner_id=user.id, name="p")
        self.session.add(project)
        self.session.commit()
        job = ConversionJob(project_id=project.id, source_model_id="m1")
        self.session.add(job)
        self.session.commit()
        stage = JobStage(job_id=job.id, stage_name="onnx_export", stage_order=3,
                         status="SUCCESS", progress=100, metrics={"nodes": 42})
        self.session.add(stage)
        self.session.commit()
        artifact = Artifact(job_id=job.id, stage="03-onnx", artifact_type="onnx",
                            filename="model.onnx", storage_key="jobs/{}/stages/03-onnx/model.onnx".format(job.id),
                            size=123, sha256="cd" * 32, mime_type="application/onnx")
        self.session.add(artifact)
        self.session.commit()
        self.assertEqual(self.session.get(JobStage, stage.id).metrics["nodes"], 42)
        self.assertEqual(self.session.get(Artifact, artifact.id).storage_key.split("/")[0], "jobs")

    def test_worker_and_audit(self):
        worker = Worker(worker_id="xpu-worker-01", hostname="node1", device_type="xpu",
                        chip="KUNLUNXIN", device_id="0", status="READY")
        self.session.add(worker)
        self.session.commit()
        log = AuditLog(user_id="u1", action="JOB_CREATED", resource_type="conversion_job",
                       resource_id="j1", ip="127.0.0.1", meta={"model": "yolov10"})
        self.session.add(log)
        self.session.commit()
        self.assertEqual(self.session.get(Worker, "xpu-worker-01").chip, "KUNLUNXIN")
        self.assertEqual(self.session.get(AuditLog, log.id).meta["model"], "yolov10")


class RepositoryTest(unittest.TestCase):
    def setUp(self):
        self.session = _make_session()

    def tearDown(self):
        self.session.close()

    def test_job_repository_lifecycle(self):
        repo = JobRepository(self.session)
        job = repo.create(project_id="p1", source_model_id="m1",
                          config={"task": "detection"}, pipeline_version="10-stage")
        self.assertEqual(job.status, "CREATED")
        repo.update_status(job.id, "QUEUED", queued_at=None)
        repo.update_status(job.id, "RUNNING")
        job = repo.get(job.id)
        self.assertEqual(job.status, "RUNNING")
        stage = repo.add_stage(job.id, "inspect", 1)
        repo.update_stage(stage.id, status="SUCCESS", progress=100, duration_ms=50)
        self.assertEqual(repo.list_stages(job.id)[0].status, "SUCCESS")
        self.assertEqual(repo.list_by_project("p1")[0].id, job.id)

    def test_event_sequence_strictly_increasing(self):
        repo = JobEventRepository(self.session)
        e1 = repo.append("j1", "stage_started", stage="inspect")
        e2 = repo.append("j1", "stage_finished", stage="inspect", progress=100)
        e3 = repo.append("j1", "log", message="hello")
        self.assertEqual([e1.sequence, e2.sequence, e3.sequence], [1, 2, 3])
        # 断线恢复: 从 sequence 之后回放
        after = repo.list_after("j1", sequence=1)
        self.assertEqual([e.sequence for e in after], [2, 3])

    def test_artifact_and_audit_repository(self):
        arts = ArtifactRepository(self.session)
        a = arts.create("j1", "model.onnx", "jobs/j1/stages/03-onnx/model.onnx",
                        stage="03-onnx", artifact_type="onnx", size=10, sha256="ef" * 32)
        self.assertEqual(arts.get(a.id).filename, "model.onnx")
        self.assertEqual(len(arts.list_by_job("j1")), 1)

        audit = AuditLogRepository(self.session)
        rec = audit.record("u1", "MODEL_UPLOADED", resource_type="model", resource_id="m1", ip="10.0.0.1")
        self.assertEqual(rec.action, "MODEL_UPLOADED")
        self.assertEqual(len(audit.list_by_user("u1")), 1)


if __name__ == "__main__":
    unittest.main()
