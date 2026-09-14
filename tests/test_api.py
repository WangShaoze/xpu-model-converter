# -*- coding: utf-8 -*-
"""Phase 5 FastAPI 测试: 认证 / 项目 / 模型上传 / Job 入队 / SSE / Artifact / 鉴权拒绝。"""
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xpu_platform.api.config import Settings
from xpu_platform.api.dependencies import get_db
from xpu_platform.api.main import app
from xpu_platform.db.models import Base, ConversionJob
from xpu_platform.db.repositories import JobEventRepository


def _make_settings(workspace: str) -> Settings:
    return Settings(
        database_url="sqlite+pysqlite:///:memory:",
        workspace_root=workspace,
        max_upload_mb=1,
        jwt_secret="test-secret" * 2,
    )


class ApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.workspace = Path(cls.tmp.name) / "ws"
        cls.workspace.mkdir()
        cls.settings_patcher = mock.patch(
            "xpu_platform.api.config.get_settings", lambda: _make_settings(str(cls.workspace)))
        cls.settings_patcher.start()

    @classmethod
    def tearDownClass(cls):
        cls.settings_patcher.stop()
        app.dependency_overrides.clear()
        cls.tmp.cleanup()

    def setUp(self):
        # 临时文件库: 多 session 共享同一磁盘库, 避免内存库连接隔离/过期刷新问题
        self.db_path = Path(self.tmp.name) / "test-{}.db".format(uuid.uuid4().hex[:8])
        self.db_path.unlink(missing_ok=True)
        self.engine = create_engine(
            "sqlite+pysqlite:///{}".format(self.db_path),
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(self.engine)
        sf = sessionmaker(bind=self.engine, autoflush=False)
        app.state.session_factory = sf

        def override_get_db():
            db = sf()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = override_get_db
        self.client = TestClient(app)
        self.token = self._register()
        self.auth = {"Authorization": "Bearer " + self.token}

    def _register(self, username="alice", password="password123") -> str:
        r = self.client.post("/api/v1/auth/register", json={
            "username": username, "email": "{}@x.com".format(username), "password": password})
        self.assertEqual(r.status_code, 201, r.text)
        return r.json()["access_token"]

    # ---- 认证 ----
    def test_login_and_me(self):
        r = self.client.post("/api/v1/auth/login", json={
            "username": "alice", "password": "password123"})
        self.assertEqual(r.status_code, 200)
        me = self.client.get("/api/v1/auth/me", headers=self.auth)
        self.assertEqual(me.status_code, 200)
        self.assertEqual(me.json()["username"], "alice")

    def test_unauthorized_rejected(self):
        r = self.client.get("/api/v1/auth/me")
        self.assertEqual(r.status_code, 401)
        r = self.client.get("/api/v1/projects")
        self.assertEqual(r.status_code, 401)

    # ---- 项目 + 模型 ----
    def test_project_and_model_flow(self):
        proj = self.client.post("/api/v1/projects", json={"name": "p1", "description": "d"},
                                headers=self.auth)
        self.assertEqual(proj.status_code, 201)
        pid = proj.json()["id"]

        files = {"file": ("best.pt", b"fake-weight-bytes", "application/octet-stream")}
        r = self.client.post("/api/v1/projects/{}/models".format(pid), files=files, headers=self.auth)
        self.assertEqual(r.status_code, 201, r.text)
        self.assertEqual(r.json()["status"], "READY")
        self.assertTrue(r.json()["sha256"])

        # 扩展名白名单拒绝
        bad = {"file": ("malware.exe", b"MZ", "application/octet-stream")}
        rb = self.client.post("/api/v1/projects/{}/models".format(pid), files=bad, headers=self.auth)
        self.assertEqual(rb.status_code, 400)

        # 越权访问他人项目
        self._register("bob")
        bob = self.client.post("/api/v1/auth/login", json={"username": "bob", "password": "password123"})
        bob_auth = {"Authorization": "Bearer " + bob.json()["access_token"]}
        rb2 = self.client.get("/api/v1/projects/{}".format(pid), headers=bob_auth)
        self.assertEqual(rb2.status_code, 404)

    # ---- Job ----
    def test_create_job_enqueues_and_cancel(self):
        pid = self.client.post("/api/v1/projects", json={"name": "p"}, headers=self.auth).json()["id"]
        src = self.workspace / "source"
        src.mkdir()
        (src / "best.pt").write_bytes(b"fake")
        model = self.client.post(
            "/api/v1/projects/{}/models".format(pid),
            files={"file": ("best.pt", b"fake-weight-bytes", "application/octet-stream")},
            headers=self.auth).json()
        job = self.client.post("/api/v1/jobs", json={
            "project_id": pid, "model_id": model["id"],
            "config": {"model_type": "yolov10", "task": "detection"}},
            headers=self.auth)
        self.assertEqual(job.status_code, 201, job.text)
        jid = job.json()["id"]
        self.assertEqual(job.json()["status"], "QUEUED")
        # 已入队
        self.assertEqual(app.state.job_queue.pending(), 1)

        detail = self.client.get("/api/v1/jobs/{}".format(jid), headers=self.auth)
        self.assertEqual(detail.status_code, 200)
        cancel = self.client.post("/api/v1/jobs/{}/cancel".format(jid), headers=self.auth)
        self.assertEqual(cancel.json()["status"], "CANCELLED")

    def test_create_job_requires_ready_model(self):
        pid = self.client.post("/api/v1/projects", json={"name": "p"}, headers=self.auth).json()["id"]
        r = self.client.post("/api/v1/jobs", json={
            "project_id": pid, "model_id": "nonexistent", "config": {}}, headers=self.auth)
        self.assertEqual(r.status_code, 404)

    # ---- SSE 事件流 ----
    def test_sse_events_with_last_event_id(self):
        db = sessionmaker(bind=self.engine, autoflush=False)()
        # 构造终态 Job, 预置事件
        job = ConversionJob(project_id="p-x", source_model_id="m-x", status="SUCCESS", config={})
        db.add(job)
        db.commit()
        jid = job.id  # session 存活时缓存, 避免 close 后访问过期 ORM 属性
        repo = JobEventRepository(db)
        repo.append(jid, "stage_started", stage="inspect", progress=10)
        repo.append(jid, "stage_finished", stage="inspect", progress=100)
        db.close()

        with self.client.stream("GET", "/api/v1/jobs/{}/events".format(jid), headers=self.auth) as resp:
            self.assertEqual(resp.status_code, 200)
            body = "".join(resp.iter_text())
        self.assertIn("event: stage_started", body)
        self.assertIn("event: stage_finished", body)
        # 带 Last-Event-ID 断线恢复: 跳过后台刚发的第一条
        with self.client.stream(
                "GET", "/api/v1/jobs/{}/events".format(jid),
                headers={**self.auth, "Last-Event-ID": "1"}) as resp:
            body2 = "".join(resp.iter_text())
        self.assertIn("id: 2", body2)


if __name__ == "__main__":
    unittest.main()