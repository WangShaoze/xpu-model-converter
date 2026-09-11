# -*- coding: utf-8 -*-
"""``xpu-converter`` 命令行入口(建设目标 §13)。

::

    xpu-converter inspect  best.pt
    xpu-converter convert  --model best.pt --model-type yolov10 --hardware kunlun \\
                           --input-shape 1,3,640,640 --precision fp16 --output ./output
    xpu-converter export-onnx --model best.pt --model-type yolov10 --output ./output/onnx
    xpu-converter analyze  model.onnx --device kunlun
    xpu-converter compile  --model model.onnx --device kunlun --precision fp16
    xpu-converter validate --source best.pt --target model.xpu --dataset ./testdata
    xpu-converter package  --model model.xpu --model-type yolov10 --output ./package
    xpu-converter build    --model best.pt --model-type yolov10 --device kunlun --precision fp16
"""
import argparse
import sys
from typing import List, Optional

from xpu_converter.cli.common import add_hardware_args, add_model_args
from xpu_converter.cli.compile import cmd_compile
from xpu_converter.cli.convert import cmd_build, cmd_convert, cmd_export_onnx
from xpu_converter.cli.inspect import (
    cmd_analyze,
    cmd_inspect,
    cmd_list_capabilities,
    cmd_list_models,
)
from xpu_converter.cli.package import cmd_package
from xpu_converter.cli.validate import cmd_validate
from xpu_converter.errors import XpuConverterError
from xpu_converter.logging_utils import setup_logging
from xpu_converter.version import CONVERTER_VERSION, RUNTIME_API_VERSION


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xpu-converter",
        description="模型转换平台: best.pt → XPU 模型 → Docker 交付包",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
               "  xpu-converter convert --model ./best.pt --model-type yolov10 \\\n"
               "      --hardware kunlun --input-shape 1,3,640,640 --precision fp16 --output ./output\n",
    )
    parser.add_argument("--version", action="version",
                        version="xpu-converter {} (runtime api {})".format(CONVERTER_VERSION, RUNTIME_API_VERSION))
    parser.add_argument("-v", "--verbose", action="store_true", help="输出调试日志")
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    # ---------------------------------------------------------------- inspect
    inspect_parser = subparsers.add_parser("inspect", help="识别模型并查看元信息")
    add_model_args(inspect_parser, positional=True)
    inspect_parser.add_argument("--input-shape", dest="input_shape", default=None, help="覆盖输入形状, 如 1,3,640,640")
    add_hardware_args(inspect_parser, with_precision=False)
    inspect_parser.add_argument("--json", default=None, help="把识别结果写到指定 JSON 文件")
    inspect_parser.set_defaults(handler=cmd_inspect)

    # ---------------------------------------------------------------- convert
    convert_parser = subparsers.add_parser("convert", help="一键完成: 模型 → ONNX → XPU → 交付包(10 步)")
    _add_pipeline_args(convert_parser, default_output="./output")
    convert_parser.set_defaults(handler=cmd_convert)

    # ------------------------------------------------------------------ build
    build_parser = subparsers.add_parser("build", help="与 convert 等价的一键入口, 默认输出到 dist/")
    _add_pipeline_args(build_parser, default_output="./dist", with_docker_flag=True)
    build_parser.set_defaults(handler=cmd_build)

    # ------------------------------------------------------------ export-onnx
    onnx_parser = subparsers.add_parser("export-onnx", help="只导出 ONNX 中间模型")
    add_model_args(onnx_parser)
    onnx_parser.add_argument("--output", default=None, help="输出 ONNX 文件或目录, 默认 ./output/onnx")
    onnx_parser.add_argument("--input-shape", dest="input_shape", default=None, help="覆盖输入形状, 如 1,3,640,640")
    onnx_parser.add_argument("--opset", type=int, default=None, help="ONNX opset 版本")
    onnx_parser.add_argument("--dynamic", action="store_true", help="导出动态 batch(默认静态 shape)")
    onnx_parser.set_defaults(handler=cmd_export_onnx)

    # ---------------------------------------------------------------- analyze
    analyze_parser = subparsers.add_parser("analyze", help="分析 ONNX 算子在目标后端的支持情况")
    add_model_args(analyze_parser, positional=True)
    add_hardware_args(analyze_parser)
    analyze_parser.add_argument("--json", default=None, help="把分析结果写到指定 JSON 文件")
    analyze_parser.set_defaults(handler=cmd_analyze)

    # ---------------------------------------------------------------- compile
    compile_parser = subparsers.add_parser("compile", help="把 ONNX 编译为 XPU 产物")
    add_model_args(compile_parser)
    add_hardware_args(compile_parser)
    compile_parser.add_argument("--output", default=None, help="编译产物输出目录, 默认 ./output/xpu")
    compile_parser.add_argument("--input-shape", dest="input_shape", default=None, help="固化输入形状, 如 1,3,640,640")
    compile_parser.add_argument("--json", default=None, help="把编译报告写到指定 JSON 文件")
    compile_parser.set_defaults(handler=cmd_compile)

    # --------------------------------------------------------------- validate
    validate_parser = subparsers.add_parser("validate", help="基准模型与 XPU 产物的精度校验")
    validate_parser.add_argument("--source", required=True, help="基准模型: .pt 框架模型或 .onnx")
    validate_parser.add_argument("--target", required=True, help="待验证产物: model.xpu 或 .onnx")
    validate_parser.add_argument("--model-type", dest="model_type", default=None, help="source 的模型类型")
    validate_parser.add_argument("--dataset", default=None, help="校验数据集目录(内含 .npy), 缺省用确定性随机输入")
    validate_parser.add_argument("--input-shape", dest="input_shape", default=None, help="覆盖输入形状")
    validate_parser.add_argument("--max-samples", dest="max_samples", type=int, default=4, help="最大校验样本数")
    add_hardware_args(validate_parser)
    validate_parser.add_argument("--benchmark", action="store_true", help="同时执行性能基准测试")
    validate_parser.add_argument("--iterations", type=int, default=50, help="基准测试迭代次数")
    validate_parser.add_argument("--report-only", dest="report_only", action="store_true",
                                 help="精度不一致时只报告不报错")
    validate_parser.add_argument("--output", default=None, help="精度报告输出文件或目录")
    validate_parser.set_defaults(handler=cmd_validate)

    # ---------------------------------------------------------------- package
    package_parser = subparsers.add_parser("package", help="由已有 XPU 产物生成 Docker 交付包")
    add_model_args(package_parser)
    add_hardware_args(package_parser)
    package_parser.add_argument("--output", default=None, help="交付包输出目录, 默认 ./package")
    package_parser.add_argument("--input-shape", dest="input_shape", default=None, help="覆盖输入形状")
    package_parser.add_argument("--version", default=None, help="交付包版本, 默认 v1.0")
    package_parser.add_argument("--package-name", dest="package_name", default=None, help="包名, 默认取模型类型")
    package_parser.add_argument("--runtime", default=None, help="Runtime 类型, 默认 detection")
    package_parser.add_argument("--image-tar", dest="image_tar", default=None,
                                help="包内已有的镜像 tar 文件名(README 据此描述部署方式)")
    package_parser.add_argument("--dev-package", dest="dev_package", action="store_true",
                                help="允许把 degraded 占位产物打入 Docker 交付包(仅供开发联调, 默认禁止)")
    package_parser.set_defaults(handler=cmd_package)

    # ------------------------------------------------------------ list-models
    models_parser = subparsers.add_parser("list-models", help="列出模型的生命周期状态(stable/experimental/planned)")
    models_parser.add_argument("--status", default=None,
                               choices=("stable", "experimental", "deprecated", "planned"),
                               help="只显示指定状态的模型")
    models_parser.add_argument("--json", default=None, help="把列表写到指定 JSON 文件")
    models_parser.set_defaults(handler=cmd_list_models)

    # ------------------------------------------------------ list-capabilities
    caps_parser = subparsers.add_parser("list-capabilities", help="打印算子能力表与目标硬件能力指纹")
    add_hardware_args(caps_parser, with_precision=False)
    caps_parser.add_argument("--json", default=None, help="把能力表写到指定 JSON 文件")
    caps_parser.set_defaults(handler=cmd_list_capabilities)

    return parser


def _add_pipeline_args(parser: argparse.ArgumentParser, default_output: str,
                       with_docker_flag: bool = False) -> None:
    # 默认值统一留空, 便于区分"用户显式指定"与"未指定", 从而正确套用 --manifest(§14)
    add_model_args(parser, required=False)
    add_hardware_args(parser)
    parser.add_argument("--manifest", default=None,
                        help="转换 Manifest YAML(建设目标 §14), 提供默认的模型/精度/形状等, 命令行优先")
    parser.add_argument("--output", default=None, help="输出目录, 默认 {}".format(default_output))
    parser.add_argument("--input-shape", dest="input_shape", default=None, help="输入形状, 默认 1,3,640,640")
    parser.add_argument("--version", default=None, help="交付包版本, 默认 v1.0")
    parser.add_argument("--package-name", dest="package_name", default=None, help="包名, 默认取模型类型")
    parser.add_argument("--runtime", default=None, help="Runtime 类型, 默认 detection")
    parser.add_argument("--optimization-level", dest="optimization_level", type=int, default=None,
                        help="图优化级别 0/1/2, 默认取硬件配置")
    parser.add_argument("--validation-dataset", dest="validation_dataset", default=None,
                        help="精度校验数据集目录(内含 .npy)")
    parser.add_argument("--no-validate", dest="no_validate", action="store_true", help="跳过精度校验")
    parser.add_argument("--benchmark-iterations", dest="benchmark_iterations", type=int, default=50,
                        help="性能基准迭代次数, 默认 50")
    parser.add_argument("--no-docker", dest="no_docker", action="store_true", help="不生成 Docker 交付包")
    if with_docker_flag:
        parser.add_argument("--docker", dest="docker_flag", action="store_true",
                            help="生成 Docker 交付包(默认开启, 与 --no-docker 互斥)")
    parser.add_argument("--allow-degraded", dest="allow_degraded", action="store_true",
                        help="缺少真实昆仑后端时允许生成降级占位产物(仅供开发联调, 默认禁止)")
    parser.add_argument("--dev-package", dest="dev_package", action="store_true",
                        help="允许把 degraded 占位产物打入 Docker 交付包(仅供开发联调, 默认禁止)")
    parser.add_argument("--json", default=None, help="把转换结果写到指定 JSON 文件")


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as err:  # --help / 参数错误: 规范化为返回码, 便于嵌入调用
        return int(err.code or 0)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0

    setup_logging("DEBUG" if getattr(args, "verbose", False) else "INFO")
    handler = getattr(args, "handler", None)
    if handler is None:  # pragma: no cover - 参数表固定, 不会走到
        parser.error("未知命令: {}".format(args.command))
    try:
        return int(handler(args) or 0)
    except XpuConverterError as err:
        print("[FAIL] {}".format(err), file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover
        print("已中断", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
