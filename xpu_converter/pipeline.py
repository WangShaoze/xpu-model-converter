# -*- coding: utf-8 -*-
"""端到端转换流水线(建设目标 §2 / §7)。

一条命令跑完 10 个步骤, 并以 ``[01] 标题 ... OK`` 的形式输出:

    [01] 模型识别        [02] 模型加载        [03] PyTorch → ONNX
    [04] ONNX Graph Check[05] XPU Operator Analysis
    [06] Graph Optimization  [07] Kunlun Compile
    [08] Accuracy Validation [09] Performance Benchmark
    [10] Docker Package
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from xpu_converter.backend.base import BackendArtifact, OperatorAnalysis
from xpu_converter.config import BuildManifest, HardwareConfig, ModelConfig
from xpu_converter.errors import XpuConverterError
from xpu_converter.exporter.docker_exporter import DockerExporter
from xpu_converter.frontend.base import BaseModelAdapter, FrontendModel
from xpu_converter.ir import onnx as onnx_ir
from xpu_converter.logging_utils import StepReporter, get_logger
from xpu_converter.optimizer.pipeline import OptimizationPipeline, OptimizationReport
from xpu_converter.registry.backend_registry import create_backend, resolve_hardware_config
from xpu_converter.registry.model_registry import create_adapter, detect_model_type
from xpu_converter.rewrite.registry import RewriteResult, default_registry
from xpu_converter.validator.accuracy import AccuracyReport, AccuracyValidator, load_dataset_inputs
from xpu_converter.validator.benchmark import BenchmarkResult, BenchmarkRunner

logger = get_logger(__name__)

STEP_TITLES = [
    "模型识别",
    "模型加载",
    "PyTorch → ONNX",
    "ONNX Graph Check",
    "XPU Operator Analysis",
    "Graph Optimization",
    "Kunlun Compile",
    "Accuracy Validation",
    "Performance Benchmark",
    "Docker Package",
]


@dataclass
class ConversionResult:
    """一次完整转换的结果。"""

    model_path: str = ""
    model_type: str = ""
    framework: str = "pytorch"
    task: str = "detection"
    hardware: str = "kunlun"
    precision: str = "fp16"
    input_shape: List[int] = field(default_factory=list)
    output_dir: str = ""
    onnx_path: str = ""
    optimized_onnx_path: str = ""
    package_zip: str = ""
    package_dir: str = ""
    frontend_model: Optional[FrontendModel] = None
    operator_analysis: Optional[OperatorAnalysis] = None
    optimization_report: Optional[OptimizationReport] = None
    rewrite_result: Optional[RewriteResult] = None
    artifact: Optional[BackendArtifact] = None
    accuracy: Optional[AccuracyReport] = None
    benchmark: Optional[BenchmarkResult] = None
    steps: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def degraded(self) -> bool:
        return bool(self.artifact and self.artifact.degraded)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_path": self.model_path,
            "model_type": self.model_type,
            "framework": self.framework,
            "task": self.task,
            "hardware": self.hardware,
            "precision": self.precision,
            "input_shape": list(self.input_shape),
            "output_dir": self.output_dir,
            "onnx_path": self.onnx_path,
            "optimized_onnx_path": self.optimized_onnx_path,
            "package_zip": self.package_zip,
            "package_dir": self.package_dir,
            "degraded": self.degraded,
            "operator_analysis": self.operator_analysis.to_dict() if self.operator_analysis else {},
            "optimization": self.optimization_report.to_dict() if self.optimization_report else {},
            "rewrite": self.rewrite_result.to_dict() if self.rewrite_result else {},
            "artifact": self.artifact.to_dict() if self.artifact else {},
            "accuracy": self.accuracy.to_dict() if self.accuracy else {},
            "benchmark": self.benchmark.to_dict() if self.benchmark else {},
            "steps": list(self.steps),
            "warnings": list(self.warnings),
        }


class ConversionPipeline:
    """模型 → XPU → Docker 交付包的完整流水线。"""

    def __init__(
        self,
        model_path: str,
        output_dir: str = "./output",
        model_type: Optional[str] = None,
        hardware: str = "kunlun",
        precision: Optional[str] = None,
        input_shape: Optional[List[int]] = None,
        runtime: str = "detection",
        version: str = "v1.0",
        package_name: Optional[str] = None,
        optimization_level: Optional[int] = None,
        device: Optional[str] = None,
        sdk_adapter: Optional[str] = None,
        validation_dataset: Optional[str] = None,
        validation_enabled: bool = True,
        benchmark_iterations: int = 50,
        export_docker: bool = True,
        allow_degraded: bool = True,
        optimizer_level_override: Optional[int] = None,
        reporter: Optional[StepReporter] = None,
    ) -> None:
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.model_type = detect_model_type(model_path, model_type)
        self.hardware = hardware
        self.precision = precision
        self.input_shape = list(input_shape) if input_shape else None
        self.runtime = runtime
        self.version = version
        self.package_name = package_name
        self.optimization_level = optimization_level
        self.device = device
        self.sdk_adapter = sdk_adapter
        self.validation_dataset = validation_dataset
        self.validation_enabled = bool(validation_enabled)
        self.benchmark_iterations = int(benchmark_iterations)
        self.export_docker = bool(export_docker)
        self.allow_degraded = bool(allow_degraded)
        self.reporter = reporter or StepReporter(total=len(STEP_TITLES))
        self.log = logger

        self.result = ConversionResult(
            model_path=str(model_path),
            model_type=self.model_type,
            hardware=hardware,
            output_dir=str(self.output_dir),
        )

    # ------------------------------------------------------------------ 主流程
    def run(self) -> ConversionResult:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._run_step(0, self._step_inspect)
        self._run_step(1, self._step_load)
        self._run_step(2, self._step_export_onnx)
        self._run_step(3, self._step_graph_check)
        self._run_step(4, self._step_analyze_operators)
        self._run_step(5, self._step_optimize)
        self._run_step(6, self._step_compile)
        self._run_step(7, self._step_validate)
        self._run_step(8, self._step_benchmark)
        self._run_step(9, self._step_package)
        return self.result

    def _run_step(self, index: int, action) -> None:
        title = STEP_TITLES[index]
        try:
            detail = action()
        except XpuConverterError as err:
            self.reporter.step(title, ok=False, detail=str(err))
            self.result.steps.append({"index": index + 1, "title": title, "ok": False, "detail": str(err)})
            raise
        except Exception as err:  # 非预期异常同样按步骤失败上报
            message = "{}: {}".format(type(err).__name__, err)
            self.reporter.step(title, ok=False, detail=message)
            self.result.steps.append({"index": index + 1, "title": title, "ok": False, "detail": message})
            raise
        self.reporter.step(title, ok=True, detail=detail or "")
        self.result.steps.append({"index": index + 1, "title": title, "ok": True, "detail": detail or ""})

    # ------------------------------------------------------------------ 步骤 01
    def _step_inspect(self) -> str:
        self.adapter = self._create_adapter()
        frontend = self.adapter.inspect(self.model_path)
        self.result.frontend_model = frontend
        self.result.framework = frontend.framework
        self.result.task = frontend.task
        self.result.input_shape = list(frontend.input_shape)
        return "{} / {} / {} 类, 输入 {}".format(
            frontend.framework, frontend.model_type, frontend.num_classes, frontend.input_shape
        )

    def _create_adapter(self) -> BaseModelAdapter:
        overrides: Dict[str, Any] = {}
        if self.input_shape:
            overrides["input_shape"] = self.input_shape
        adapter = create_adapter(self.model_type, overrides)
        return adapter

    # ------------------------------------------------------------------ 步骤 02
    def _step_load(self) -> str:
        self.model = self.adapter.load_model(self.model_path)
        name = type(self.model).__name__
        return "已加载模型对象: {}".format(name)

    # ------------------------------------------------------------------ 步骤 03
    def _step_export_onnx(self) -> str:
        onnx_dir = self.output_dir / "onnx"
        onnx_dir.mkdir(parents=True, exist_ok=True)
        target = str(onnx_dir / "{}_raw.onnx".format(self.adapter.config.model_type))
        exported = self.adapter.export_onnx(
            self.model_path, target,
            input_shape=self.input_shape or self.adapter.input_shape(),
            opset=self.adapter.opset(),
            dynamic=False,
        )
        self.result.onnx_path = str(exported)
        size = os.path.getsize(exported) / (1024.0 * 1024.0)
        return "ONNX: {} ({:.2f} MB)".format(Path(exported).name, size)

    # ------------------------------------------------------------------ 步骤 04
    def _step_graph_check(self) -> str:
        graph = onnx_ir.load(self.result.onnx_path)
        graph.infer_shapes()
        self.graph = graph
        checks = onnx_ir.check_model(graph.raw) if hasattr(onnx_ir, "check_model") else None
        raw_nodes = len(graph.nodes)
        return "节点 {} 个, opset {}, 输入/输出 {} / {}".format(
            raw_nodes, graph.opset, len(graph.inputs), len(graph.outputs)
        ) + ("" if checks is None else ", checker: {}".format(checks))

    # ------------------------------------------------------------------ 步骤 05
    def _step_analyze_operators(self) -> str:
        self.backend = self._create_backend()
        analysis = self.backend.analyze(self.graph)
        self.result.operator_analysis = analysis
        detail = analysis.summary()
        if analysis.unsupported_ops:
            self.result.warnings.append(
                "存在昆仑 XPU 不支持的算子: {}".format(", ".join(analysis.unsupported_ops))
            )
        return detail

    def _create_backend(self):
        overrides: Dict[str, Any] = {}
        if self.precision:
            overrides["precision"] = self.precision
        if self.device:
            overrides["device"] = self.device
        if self.sdk_adapter:
            overrides["sdk_adapter"] = self.sdk_adapter
        if self.optimization_level is not None:
            overrides["optimization_level"] = self.optimization_level
        self.hardware_config: HardwareConfig = resolve_hardware_config(self.hardware, overrides)
        self.result.precision = self.hardware_config.precision
        return create_backend(self.hardware, self.hardware_config)

    # ------------------------------------------------------------------ 步骤 06
    def _step_optimize(self) -> str:
        level = self.hardware_config.optimization_level
        report = OptimizationPipeline.default(level).run(self.graph)
        rewrite = default_registry().apply(self.graph)
        self.result.optimization_report = report
        self.result.rewrite_result = rewrite

        optimized_dir = self.output_dir / "onnx"
        optimized_dir.mkdir(parents=True, exist_ok=True)
        optimized = str(optimized_dir / "{}_optimized.onnx".format(self.adapter.config.model_type))
        onnx_ir.save(self.graph, optimized)
        self.result.optimized_onnx_path = optimized

        changed = [item.name for item in report.passes if item.changed]
        detail = "passes: {}, 改写: {} 处".format(
            ", ".join(changed) or "无变更",
            rewrite.applied_count,
        )
        if rewrite.notes:
            for note in rewrite.notes:
                self.reporter.info(note)
        return detail

    # ------------------------------------------------------------------ 步骤 07
    def _step_compile(self) -> str:
        from xpu_converter.backend.kunlun.graph_builder import XpuGraphBuilder

        xpu_dir = self.output_dir / "xpu"
        xpu_dir.mkdir(parents=True, exist_ok=True)
        builder = XpuGraphBuilder(self.hardware)
        input_shapes = {}
        if self.input_shape and self.graph.inputs:
            input_shapes[self.graph.inputs[0].name] = list(self.input_shape)
        xpu_graph = builder.build(
            self.graph,
            precision=self.hardware_config.precision,
            workdir=str(xpu_dir),
            input_shapes=input_shapes,
        )
        for note in xpu_graph.notes:
            self.reporter.info(note)
        artifact = self.backend.compile(
            xpu_graph, str(xpu_dir / "model.xpu"), config=None
        )
        self.result.artifact = artifact
        for note in artifact.notes:
            self.result.warnings.append(note)
            self.reporter.info(note)
        return "产物 {} (adapter={}, 精度={})".format(
            Path(artifact.model_path).name, artifact.sdk_adapter, artifact.precision
        )

    # ------------------------------------------------------------------ 步骤 08
    def _step_validate(self) -> str:
        from xpu_converter.backend.kunlun.runtime import OnnxRuntimeSession

        artifact = self.result.artifact
        reference = OnnxRuntimeSession(self.result.onnx_path, device="cpu")
        target = self.backend.create_runtime(artifact)

        input_spec = {}
        for tensor in self.graph.inputs:
            input_spec[tensor.name] = list(tensor.shape)
        samples, synthetic, notes = load_dataset_inputs(
            self.validation_dataset, input_spec, max_samples=4
        )
        for note in notes:
            self.reporter.info(note)
            self.result.warnings.append(note)

        report = AccuracyValidator(fail_on_mismatch=False).validate(
            reference, target, samples,
            onnx_session=OnnxRuntimeSession(self.result.optimized_onnx_path, device="cpu"),
            synthetic=synthetic, notes=notes,
        )
        self.result.accuracy = report
        reference.close()
        target.close()
        return "{} | {}".format(report.summary(), "通过" if report.passed else "未通过")

    # ------------------------------------------------------------------ 步骤 09
    def _step_benchmark(self) -> str:
        artifact = self.result.artifact
        session = self.backend.create_runtime(artifact)
        runner = BenchmarkRunner(iterations=self.benchmark_iterations, warmup=max(1, self.benchmark_iterations // 10))
        result = runner.run(session, device=self.hardware_config.device)
        session.close()
        self.result.benchmark = result
        for note in result.notes:
            self.reporter.info(note)
            self.result.warnings.append(note)
        return result.summary()

    # ------------------------------------------------------------------ 步骤 10
    def _step_package(self) -> str:
        if not self.export_docker:
            self.reporter.info("已按参数跳过 Docker 交付包导出")
            return "已跳过"
        manifest = self._build_manifest()
        exporter = DockerExporter(
            manifest=manifest,
            model_config=self.adapter.config,
            hardware_config=self.hardware_config,
            runtime=self.runtime,
            # 交付包 packages/ 内的依赖轮子来源由 DockerExporter 从 hardware_config.docker 读取
        )
        # 交付包目录与 ZIP 直接落在输出目录顶层, 与建设目标 §13 的 dist/ 结构一致
        package_root = self.output_dir
        zip_path = exporter.export(
            model_path=self.result.artifact.model_path,
            output_dir=str(package_root),
            artifact=self.result.artifact,
            onnx_path=self.result.optimized_onnx_path,
            accuracy=self.result.accuracy,
            benchmark=self.result.benchmark,
        )
        self.result.package_zip = zip_path
        self.result.package_dir = str(package_root / manifest.package_name)
        return Path(zip_path).name

    def _build_manifest(self) -> BuildManifest:
        name = self.package_name or self.adapter.config.model_type
        return BuildManifest.from_dict({
            "name": name,
            "version": self.version,
            "task": self.result.task,
            "source": {
                "framework": self.result.framework,
                "model_type": self.adapter.config.model_type,
                "file": os.path.basename(self.model_path),
            },
            "input": {
                "shape": list(self.result.input_shape),
                "dtype": self.adapter.config.input_dtype,
            },
            "target": {
                "hardware": self.hardware,
                "precision": self.hardware_config.precision,
                "batch_size": 1,
            },
            "runtime": {"type": self.runtime, "port": int(self.hardware_config.runtime.get("port", 58025))},
            "validation": {
                "enabled": self.validation_enabled,
                "dataset": self.validation_dataset or "",
            },
            "package": {"docker": True, "name": "{}_dockerimg_{}".format(name, self.version)},
        })
