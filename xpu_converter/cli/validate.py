# -*- coding: utf-8 -*-
"""``xpu-converter validate``(建设目标 §13 / §17): 基准 vs 目标产物精度校验。"""
import tempfile
from pathlib import Path

from xpu_converter.cli.common import (
    artifact_from_model,
    backend_for,
    hardware_config_from,
    input_shape_from,
    print_kv,
    reference_session,
    save_json,
    target_session,
)
from xpu_converter.errors import ConfigError, ValidationError
from xpu_converter.validator.accuracy import AccuracyValidator, load_dataset_inputs
from xpu_converter.validator.benchmark import BenchmarkRunner


def cmd_validate(args) -> int:
    """``validate --source best.pt --target model.xpu --dataset ./testdata``。"""
    target_path = Path(args.target)
    if not target_path.is_file():
        raise ConfigError("待验证产物不存在: {}".format(target_path))

    config = hardware_config_from(args)
    backend = backend_for(args, config)
    shape = input_shape_from(getattr(args, "input_shape", None))

    with tempfile.TemporaryDirectory(prefix="xpu_validate_") as tmp:
        reference = reference_session(args.source, args.model_type, shape, Path(tmp))
        input_spec = {name: shape or list(reference.input_shapes().get(name) or [1, 3, 640, 640])
                      for name in reference.input_names}
        samples, synthetic, notes = load_dataset_inputs(
            getattr(args, "dataset", None), input_spec, max_samples=int(getattr(args, "max_samples", 4) or 4)
        )
        for note in notes:
            print("  - {}".format(note))

        artifact = artifact_from_model(str(target_path), config.precision, config.target_chip)
        target = target_session(backend, artifact, config)

        report = AccuracyValidator(fail_on_mismatch=not getattr(args, "report_only", False)).validate(
            reference, target, samples, synthetic=synthetic, notes=notes
        )
        reference.close()
        target.close()

    print_kv("精度校验", {
        "基准": args.source,
        "目标": str(target_path),
        "目标格式": artifact.artifact_format,
        "样本数": report.sample_count,
        "合成输入": report.synthetic_inputs,
        "结论": "通过" if report.passed else "未通过",
        "汇总": report.summary(),
    })
    if report.pairs:
        print_kv("逐路结果", {
            name: "{} ({})".format("OK" if detail.get("ok") else "FAIL",
                                   detail.get("pair", name))
            for name, detail in report.pairs.items()
        })
    for note in report.notes:
        print("  ! {}".format(note))

    if getattr(args, "benchmark", False):
        session = target_session(backend, artifact, config)
        result = BenchmarkRunner(iterations=int(getattr(args, "iterations", 50) or 50)).run(
            session,
            sample=samples[0] if samples else None,
            device=config.device,
            hardware=backend.describe() if hasattr(backend, "describe") else {},
            model=Path(target_path).name,
            precision=config.precision,
        )
        session.close()
        print_kv("性能基准", dict(result.latency_ms, **{"吞吐(FPS)": result.throughput_fps}))
        print_kv("硬件指纹", {
            "chip": result.chip or "(缺失)",
            "指纹": result.hardware_fingerprint,
            "可用": result.available,
            "可信": result.credible,
            "batch": result.batch,
        })
        for note in result.notes:
            print("  ! {}".format(note))

    output = getattr(args, "output", None)
    if output:
        path = Path(output)
        target_file = path / "accuracy_report.json" if path.suffix.lower() != ".json" else path
        save_json(report.to_dict(), target_file)
        print("已写出: {}".format(target_file))

    if not report.passed and not getattr(args, "report_only", False):
        raise ValidationError("精度校验未通过: {}".format(report.summary()))
    return 0 if report.passed else 1
