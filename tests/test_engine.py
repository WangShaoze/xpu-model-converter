# -*- coding: utf-8 -*-
"""第2轮 Engine 四抽象测试(ConversionContext / ArtifactStore / Stage / EventSink)。"""
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from xpu_converter.engine import (
    CollectingEventSink,
    ConversionContext,
    ConversionJob,
    InvalidArtifactKey,
    InvalidTransitionError,
    JobStage,
    JobStatus,
    LocalArtifactStore,
    MinioArtifactStore,
    StageStatus,
    create_artifact_store,
    guess_mime,
    new_context,
)
from xpu_converter.engine.converter import run_pipeline
from xpu_converter.engine.events import artifact_created, job_finished, stage_finished


class _FakeResult:
    def __init__(self, model_path="best.pt"):
        self.model_path = model_path
        self.onnx_path = ""
        self.optimized_onnx_path = ""
        self.artifact = None
        self.package_zip = ""
        self.accuracy = None
        self.benchmark = None
        self.degraded = False
        self.steps = [
            {"index": 1, "title": "模型识别", "ok": True, "detail": "inspect"},
            {"index": 2, "title": "模型加载", "ok": True, "detail": "load"},
            {"index": 3, "title": "PyTorch → ONNX", "ok": False, "detail": "export failed"},
        ]


class _FakePipeline:
    STEP_TITLES = ["模型识别", "模型加载", "PyTorch → ONNX", "ONNX Graph Check",
                   "XPU Operator Analysis", "Graph Optimization", "Backend Artifact Build",
                   "Accuracy Validation", "Performance Benchmark", "Docker Package"]

    def __init__(self):
        self.result = _FakeResult()

    def run(self):
        return self.result


class EventSinkTest(unittest.TestCase):
    def test_collecting_sink_buffers_and_echo(self):
        sink = CollectingEventSink()
        sink.emit(artifact_created("j1", "compile", artifact_type="paddle", filename="model.pdmodel"))
        sink.emit(stage_finished("j1", "compile", "SUCCESS"))
        self.assertEqual(len(sink.events), 2)
        self.assertEqual(sink.events[0].event, "artifact_created")
        self.assertEqual(sink.events[1].data["status"], "SUCCESS")

    def test_emit_typed_swallows_errors(self):
        class BrokenSink(CollectingEventSink):
            def emit(self, event):
                raise RuntimeError("downstream down")

        sink = BrokenSink()
        # emit_typed 不应抛错而阻断主流程
        sink.emit_typed(stage_finished("j1", "compile", "SUCCESS"))
        self.assertEqual(len(sink.events), 0)


class StageStateMachineTest(unittest.TestCase):
    def test_stage_lifecycle(self):
        stage = JobStage(name="compile", index=6).start()
        self.assertEqual(stage.status, StageStatus.RUNNING)
        stage.finish(StageStatus.SUCCESS)
        self.assertEqual(stage.status, StageStatus.SUCCESS)
        self.assertEqual(stage.progress, 100)
        self.assertEqual(stage.to_dict()["name"], "compile")

    def test_job_success_and_failure(self):
        job = ConversionJob(job_id="job_1").queue().run()
        self.assertEqual(job.status, JobStatus.RUNNING)
        job.succeed()
        self.assertEqual(job.status, JobStatus.SUCCESS)
        self.assertIsNotNone(job.finished_at)

        job2 = ConversionJob(job_id="job_2").queue().run().fail("boom")
        self.assertEqual(job2.status, JobStatus.FAILED)
        self.assertEqual(job2.payload["error_message"], "boom")

    def test_job_illegal_transitions_raise(self):
        # SUCCESS -> RUNNING / QUEUED 禁止
        job = ConversionJob(job_id="j1").queue().run().succeed()
        with self.assertRaises(InvalidTransitionError):
            job.run()
        with self.assertRaises(InvalidTransitionError):
            job.queue()
        # CANCELLED -> RUNNING 禁止
        cancelled = ConversionJob(job_id="j2").queue().run().request_cancel().cancel()
        self.assertEqual(cancelled.status, JobStatus.CANCELLED)
        with self.assertRaises(InvalidTransitionError):
            cancelled.run()
        # CREATED 直接结束 禁止
        with self.assertRaises(InvalidTransitionError):
            ConversionJob(job_id="j3").succeed()
        # 未请求取消直接 CANCELLED 禁止
        with self.assertRaises(InvalidTransitionError):
            ConversionJob(job_id="j4").queue().run().cancel()

    def test_job_cancel_flow(self):
        job = ConversionJob(job_id="j5").queue().run().request_cancel()
        self.assertEqual(job.status, JobStatus.CANCEL_REQUESTED)
        job.cancel()
        self.assertEqual(job.status, JobStatus.CANCELLED)
        self.assertIsNotNone(job.finished_at)

    def test_stage_illegal_transitions_raise(self):
        # PENDING 直接结束 禁止
        with self.assertRaises(InvalidTransitionError):
            JobStage(name="s1", index=1).finish(StageStatus.SUCCESS)
        stage = JobStage(name="s2", index=2).start()
        # RUNNING 二次 start 禁止
        with self.assertRaises(InvalidTransitionError):
            stage.start()
        stage.finish(StageStatus.SUCCESS)
        # SUCCESS -> FAILED 禁止
        with self.assertRaises(InvalidTransitionError):
            stage.finish(StageStatus.FAILED)

    def test_stage_cancel(self):
        stage = JobStage(name="s3", index=3).start().cancel()
        self.assertEqual(stage.status, StageStatus.CANCELLED)
        self.assertEqual(stage.to_dict()["status"], "CANCELLED")


class ArtifactStoreTest(unittest.TestCase):
    def test_local_store_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalArtifactStore(str(Path(tmp) / "store"))
            source = Path(tmp) / "model.onnx"
            source.write_text("binary-ish")
            key = store.put("compile/artifact/model.onnx", str(source))
            self.assertTrue(store.exists("compile/artifact/model.onnx"))
            self.assertTrue(Path(key).is_file())
            got = store.get("compile/artifact/model.onnx")
            self.assertEqual(Path(got).read_text(encoding="utf-8"), "binary-ish")
            store.delete("compile/artifact/model.onnx")
            self.assertFalse(store.exists("compile/artifact/model.onnx"))

    def test_nested_key_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalArtifactStore(str(Path(tmp) / "store"))
            source = Path(tmp) / "model.pt"
            source.write_text("weights")
            key = "jobs/job_1/stages/03-onnx/model.onnx"
            store.put(key, str(source))
            # 层级保留, 而不是 Path(key).name 扁平化
            self.assertTrue((Path(tmp) / "store" / "jobs" / "job_1" / "stages" / "03-onnx" / "model.onnx").is_file())
            self.assertEqual(Path(store.get(key)).name, "model.onnx")
            # list 记录完整 key
            self.assertEqual([a.key for a in store.list()], [key])

    def test_traversal_and_absolute_keys_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalArtifactStore(str(Path(tmp) / "store"))
            source = Path(tmp) / "ok.txt"
            source.write_text("x")
            for bad in ("../evil", "a/../../evil", "/etc/passwd", "C:/windows", "a/.."):
                with self.assertRaises(InvalidArtifactKey, msg=bad):
                    store.put(bad, str(source))
                with self.assertRaises(InvalidArtifactKey, msg=bad):
                    store.get(bad)
            # 穿越 key 不得写出 store 目录之外
            self.assertFalse((Path(tmp) / "evil").exists())


class EventMetadataTest(unittest.TestCase):
    def test_event_has_id_timestamp_sequence(self):
        ev = stage_finished("job_1", "compile", "SUCCESS")
        self.assertTrue(ev.event_id)
        self.assertTrue(ev.timestamp)
        self.assertIn("event_id", ev.to_dict())
        self.assertIn("timestamp", ev.to_dict())
        self.assertIn("sequence", ev.to_dict())

    def test_sequence_strictly_increasing_per_job(self):
        sink = CollectingEventSink()
        for i in range(5):
            sink.emit(stage_finished("job_a", "s{}".format(i), "SUCCESS"))
        for i in range(2):
            sink.emit(stage_finished("job_b", "s{}".format(i), "SUCCESS"))
        seq_a = [e.sequence for e in sink.events if e.job_id == "job_a"]
        seq_b = [e.sequence for e in sink.events if e.job_id == "job_b"]
        self.assertEqual(seq_a, [1, 2, 3, 4, 5])
        self.assertEqual(seq_b, [1, 2])

    def test_console_sink_stamps_sequence(self):
        from xpu_converter.engine import ConsoleEventSink
        sink = ConsoleEventSink()
        sink.emit(stage_finished("job_1", "compile", "SUCCESS"))
        sink.emit(job_finished("job_1", "SUCCESS"))
        self.assertEqual([e.sequence for e in sink.events], [1, 2])


class ContextTest(unittest.TestCase):
    def test_new_context_prepares_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = new_context("job_1", tmp, "best.pt",
                              config={"target_backend": "paddle-xpu"})
            self.assertTrue(ctx.artifacts_dir.is_dir())
            self.assertTrue(ctx.logs_dir.is_dir())
            self.assertTrue(ctx.reports_dir.is_dir())
            self.assertEqual(ctx.config["target_backend"], "paddle-xpu")


class RunPipelineTest(unittest.TestCase):
    def test_run_pipeline_emits_events_and_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = ConversionContext(
                job_id="job_1", workspace=Path(tmp), source_model=Path(tmp) / "best.pt",
                event_sink=CollectingEventSink(),
                artifact_store=LocalArtifactStore(str(Path(tmp) / "artifacts")),
            ).prepare()
            job, manifest, collected = run_pipeline(ctx, _FakePipeline())
            self.assertEqual(job.status, JobStatus.FAILED)  # 用例里步骤3失败
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(len(collected), 3 * 2 + 1)  # 3 步 × (started+finished) + job_finished
            self.assertTrue(any(e.event == "job_finished" for e in collected))
            self.assertEqual(manifest["stages"][2]["status"], "FAILED")


def _install_fake_minio():
    """注入 fake ``minio`` 模块, 使 MinioArtifactStore 可不依赖真实客户端测试。"""
    saved = {name: sys.modules.get(name) for name in ("minio", "minio.error")}
    minio_mod = types.ModuleType("minio")
    error_mod = types.ModuleType("minio.error")

    class S3Error(Exception):
        def __init__(self, code="", message=""):
            super().__init__(code, message)
            self.code = code
            self.message = message

    class _FakeMinioClient:
        def __init__(self, *args, **kwargs):
            pass

        def bucket_exists(self, bucket):
            return True

        def make_bucket(self, bucket):
            return None

    minio_mod.Minio = _FakeMinioClient
    error_mod.S3Error = S3Error
    sys.modules["minio"] = minio_mod
    sys.modules["minio.error"] = error_mod
    return saved


def _restore_minio(saved):
    for name, mod in saved.items():
        if mod is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = mod


class MinioArtifactStoreTest(unittest.TestCase):
    def setUp(self):
        self._saved = _install_fake_minio()
        self.addCleanup(_restore_minio, self._saved)

    def _store(self, tmp, **kw):
        store = MinioArtifactStore(
            endpoint="localhost:9000", access_key="ak", secret_key="sk",
            bucket="models", cache_dir=str(Path(tmp) / "cache"), **kw)
        store.client = mock.Mock()
        store.client.bucket_exists.return_value = True
        return store

    def test_put_preserves_hierarchical_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            source = Path(tmp) / "model.pt"
            source.write_text("weights")
            key = "jobs/job_1/stages/03-onnx/model.onnx"
            self.assertEqual(store.put(key, str(source)), key)
            store.client.fput_object.assert_called_once_with(
                "models", key, str(source), content_type="application/onnx")

    def test_get_downloads_into_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            store.client.stat_object.return_value = object()
            store.client.fget_object.return_value = None
            local = store.get("jobs/j1/package.zip")
            self.assertTrue(local.startswith(str(Path(tmp) / "cache")))
            store.client.fget_object.assert_called_once_with("models", "jobs/j1/package.zip", local)

    def test_exists_missing_returns_false(self):
        from minio.error import S3Error
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            store.client.stat_object.side_effect = S3Error(code="NoSuchKey")
            self.assertFalse(store.exists("jobs/j1/missing.onnx"))
            store.client.stat_object.side_effect = None
            store.client.stat_object.return_value = object()
            self.assertTrue(store.exists("jobs/j1/present.onnx"))

    def test_delete_ignores_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            store.client.remove_object.side_effect = RuntimeError("NoSuchKey")
            store.delete("jobs/j1/x.onnx")  # 不应抛错
            store.client.remove_object.assert_called_once_with("models", "jobs/j1/x.onnx")

    def test_presigned_upload_and_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            store.client.presigned_put_object.return_value = "https://minio/put"
            store.client.presigned_get_object.return_value = "https://minio/get"
            self.assertEqual(store.presigned_upload("jobs/j1/model.pt"), "https://minio/put")
            self.assertEqual(store.presigned_download("jobs/j1/model.pt"), "https://minio/get")
            store.client.presigned_put_object.assert_called_once_with("models", "jobs/j1/model.pt", expires=3600)

    def test_minio_rejects_traversal_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            for bad in ("../evil", "/etc/passwd"):
                with self.assertRaises(InvalidArtifactKey):
                    store.put(bad, str(Path(tmp) / "model.pt"))
                with self.assertRaises(InvalidArtifactKey):
                    store.presigned_download(bad)
            store.client.fput_object.assert_not_called()


class FactoryTest(unittest.TestCase):
    def test_create_local_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = create_artifact_store("local", store_dir=str(Path(tmp) / "s"))
            self.assertIsInstance(store, LocalArtifactStore)
            self.assertTrue(store.exists("anything") in (True, False))  # 抽象方法已实现

    def test_create_minio_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._saved = _install_fake_minio()
            self.addCleanup(_restore_minio, self._saved)
            store = create_artifact_store(
                "minio", endpoint="localhost:9000", access_key="ak", secret_key="sk",
                bucket="models", cache_dir=str(Path(tmp) / "c"))
            self.assertIsInstance(store, MinioArtifactStore)
            self.assertEqual(store.bucket, "models")

    def test_unknown_backend_raises(self):
        with self.assertRaises(ValueError):
            create_artifact_store("nfs")

    def test_guess_mime(self):
        self.assertEqual(guess_mime("a.onnx"), "application/onnx")
        self.assertEqual(guess_mime("b.zip"), "application/zip")
        self.assertEqual(guess_mime("c.bin"), "application/octet-stream")


if __name__ == "__main__":
    unittest.main()