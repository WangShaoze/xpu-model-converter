# -*- coding: utf-8 -*-
"""把转换流水线接入 ConversionContext(第2轮 Web SaaS 架构, §38/§40)。

作用: 同一份 ``ConversionPipeline`` 既被 CLI 用, 也能被 Worker 用——本模块把
流水线的 10 个步骤翻译成 JobStage 状态、落盘每阶段产物并发出结构化事件, 产出
一份可审计的 conversion-manifest, 供前端 Job Detail 页直接渲染。
"""
from typing import Any, Dict, List

from xpu_converter.engine import events
from xpu_converter.engine.context import ConversionContext
from xpu_converter.engine.stage import ConversionJob, JobStage, StageStatus

# 阶段 key / 产物类型(与 Pipeline STEP_TITLES 顺序一一对应)
_STAGE_SPECS = [
    ("inspect", "metadata"),
    ("load", "model_info"),
    ("onnx_export", "onnx"),
    ("graph_check", "graph_report"),
    ("capability", "capability_report"),
    ("optimize", "onnx"),
    ("compile", "artifact"),
    ("validate", "accuracy_report"),
    ("benchmark", "benchmark_report"),
    ("package", "package"),
]

_FALLBACK_TITLES = [
    "模型识别", "模型加载", "PyTorch → ONNX", "ONNX Graph Check", "XPU Operator Analysis",
    "Graph Optimization", "Backend Artifact Build", "Accuracy Validation",
    "Performance Benchmark", "Docker Package",
]


def _stage_files(result, index: int) -> List[str]:
    """把第 index 步骤对应的产物文件从 result 上取出。"""
    artifact = result.artifact
    mapping = {
        2: [result.onnx_path] if result.onnx_path else [],
        5: [result.optimized_onnx_path] if result.optimized_onnx_path else [],
        6: list(artifact.artifact_files) if artifact else [],
        9: [result.package_zip] if result.package_zip else [],
    }
    return mapping.get(index, [])


def run_pipeline(context: ConversionContext, pipeline):
    """在上下文中执行流水线, 产出 ``(job, manifest, events)``。

    ``pipeline`` 为 :class:`xpu_converter.pipeline.ConversionPipeline` 实例。
    """
    job = ConversionJob(job_id=context.job_id, payload=dict(context.config))
    job.queue().run()

    result = None
    error = ""
    try:
        result = pipeline.run()
    except Exception as err:  # 失败步骤由 pipeline 自己在 result.steps 或异常中表达
        error = str(err)
        result = getattr(pipeline, "result", None)

    steps = list(getattr(result, "steps", []) or [])
    titles = getattr(pipeline, "STEP_TITLES", None) or _FALLBACK_TITLES

    # 第 1 遍: 先落盘各阶段产物, 收集 storage key 供第 2 遍回填
    saved: Dict[str, str] = {}
    for index, (_, _type) in enumerate(_STAGE_SPECS):
        for path in _stage_files(result, index):
            if not path:
                continue
            try:
                key = "{}/{}".format(_STAGE_SPECS[index][0], str(path).rsplit("/", 1)[-1])
                self_key = context.artifact_store.put(key, str(path))
                context.save_artifact(_STAGE_SPECS[index][0], str(path), _STAGE_SPECS[index][1])
                saved[_type] = self_key or str(path)
            except Exception as err:  # 产物落盘失败不阻断流程
                context.emit(events.log_event(context.job_id, _STAGE_SPECS[index][0], "保存产物失败: {}".format(err)))

    # 第 2 遍: 生成 JobStage 状态并发出事件
    job.stages = []
    for index, step in enumerate(steps, start=1):
        name = titles[index - 1] if index - 1 < len(titles) else "Step {}".format(index)
        _, _type = _STAGE_SPECS[index - 1]
        stage = JobStage(name=name, index=index).start()
        stage.finish(
            StageStatus.SUCCESS if step.get("ok") else StageStatus.FAILED,
            error=step.get("detail", ""),
        )
        stage.output_artifact = ",".join(_stage_files(result, index - 1))
        job.stages.append(stage)
        job.current_stage = name
        context.emit(events.stage_started(context.job_id, name))
        context.emit(events.stage_finished(context.job_id, name, stage.status.value))

    reports = {}
    if getattr(result, "accuracy", None) is not None:
        reports["accuracy"] = result.accuracy.to_dict()
    if getattr(result, "benchmark", None) is not None:
        reports["benchmark"] = result.benchmark.to_dict()

    all_ok = bool(steps) and all(step.get("ok") for step in steps)
    manifest = {
        "job_id": context.job_id,
        "status": ("success" if all_ok else "failed") if not error else "failed",
        "model": getattr(result, "model_path", ""),
        "error": error,
        "stages": [stage.to_dict() for stage in job.stages],
        "reports": reports,
        "artifacts": saved,
        "degraded": bool(getattr(result, "degraded", False)),
    }

    if all_ok and not error:
        job.succeed()
    else:
        job.fail(error or "存在失败步骤")
    context.emit(events.job_finished(context.job_id, job.status.value))
    collected = getattr(context.event_sink, "events", None)
    return job, manifest, list(collected or [])