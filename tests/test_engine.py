# -*- coding: utf-8 -*-
"""第2轮 Engine 四抽象测试(ConversionContext / ArtifactStore / Stage / EventSink)。"""
import tempfile
import unittest
from pathlib import Path

from xpu_converter.engine import (
    CollectingEventSink,
    ConversionContext,
    ConversionJob,
    JobStage,
    JobStatus,
    LocalArtifactStore,
    StageStatus,
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


if __name__ == "__main__":
    unittest.main()