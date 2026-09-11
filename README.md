# xpu-model-converter

模型转换平台：把客户给的框架模型（`best.pt`）转成**昆仑芯 XPU 编译产物**，并自动封装为
可直接交付的 **Docker 部署包**（`<算法名>-dockerimg_<版本>.zip`）。

交付包结构与客户真实交付包（`nwai-aj-falldowndetect-gxdwnngdj-dockerimg_v1.0`、
`nwai_dwt_szh_gx_nngjfznxj_a-dockerimg_v1.4`）保持**完全一致**：顶层为
`Dockerfile` / `build.sh` / `readme.txt` / `packages/` 加一个**算法同名目录**（内含 Runtime
代码、模型产物、`confidence.json`、`runtime.yaml`、`start.sh`），接口说明书 `.docx` 与
`testimage.jpg` 可选随包分发。

## 快速开始

```bash
pip install -e ".[torch,onnx]"

xpu-converter convert \
    --model ./best.pt \
    --model-type yolov10 \
    --hardware kunlun \
    --input-shape 1,3,640,640 \
    --precision fp16 \
    --output ./output
```

输出：

```
[01] 模型识别 OK
[02] 模型加载 OK
[03] PyTorch → ONNX OK
[04] ONNX Graph Check OK
[05] XPU Operator Analysis OK
[06] Graph Optimization OK
[07] Kunlun Compile OK
[08] Accuracy Validation OK
[09] Performance Benchmark OK
[10] Docker Package OK
```

最终得到 `./output/yolov10-dockerimg_v1.0.zip`。

## 命令行

| 命令 | 作用 |
| --- | --- |
| `xpu-converter inspect best.pt` | 识别模型并查看元信息 |
| `xpu-converter convert --model ...` | 一键跑完 10 步（含 Docker 交付包） |
| `xpu-converter build --model ...` | 与 `convert` 等价，默认输出到 `dist/` |
| `xpu-converter export-onnx --model ...` | 只导出 ONNX 中间模型 |
| `xpu-converter analyze model.onnx --device kunlun` | 分析算子在目标后端的支持情况 |
| `xpu-converter compile --model model.onnx --precision fp16` | 只做 XPU 编译 |
| `xpu-converter validate --source best.pt --target model.xpu` | 基准 vs 产物精度校验 |
| `xpu-converter package --model model.xpu` | 由已有产物生成 Docker 交付包 |

`convert` / `build` 支持 `--manifest`（见 [`examples/yolov10n.yaml`](examples/yolov10n.yaml)），
由一份 YAML 驱动整次转换，命令行参数优先。

## 四层模型生命周期

```
Framework Model (.pt/.pdmodel)
        │  Frontend  (xpu_converter/frontend)
        ▼
Intermediate Model (.onnx)
        │  Optimizer / Rewrite  (xpu_converter/optimizer, xpu_converter/rewrite)
        ▼
Compiled Model (model.xpu)
        │  Backend  (xpu_converter/backend/kunlun)
        ▼
Deployment Package (<算法名>-dockerimg_<版本>.zip)
           Exporter  (xpu_converter/exporter)
```

## 目录结构

```
xpu_converter/
├── cli/            # 命令行入口(inspect/convert/export-onnx/analyze/compile/validate/package/build)
├── frontend/       # 模型适配器: pytorch/{yolov8,yolov9,yolov10,yolov11}, paddle/{ppocr,paddledetection}
├── ir/             # 与框架无关的中间表示(tensor/node/graph) + ONNX 互转
├── optimizer/      # 图优化: shape inference / 常量折叠 / Conv+BN 融合 / 图简化
├── rewrite/        # 算子改写: Silu、Mish、Upsample、NMS→CPU
├── backend/kunlun/ # 昆仑芯后端(SDK 解耦): operator_registry / graph_builder / compiler / runtime
├── validator/      # 精度校验(tensor_compare/accuracy) 与性能基准(benchmark)
├── exporter/       # Docker 交付包导出: template(Jinja2) / manifest / docker_exporter
├── registry/       # 模型与后端注册表
└── pipeline.py     # 10 步端到端流水线

runtime/            # 公共 Runtime(与模型无关, 所有 YOLO 共用一套, 交付包内以 nwai_* 命名)
├── common/         # nwai_config/nwai_logging/nwai_backend/minio+kafka 上报/send_log_webserver/DevicePool/gunicorn
└── detection/      # 检测服务: /predict /predict_image /health /setflag

templates/docker/kunlun/   # Dockerfile / build.sh / readme.txt / start.sh 模板
configs/models/            # 各模型配置(yolov8/9/10/11)
configs/hardware/          # 硬件后端配置(kunlun)
examples/                  # Manifest 与一键脚本示例
tests/                     # 单元测试与端到端测试
```

## 交付包结构

结构与客户真实交付包一致：顶层固定 4 项 + 1 个算法同名目录，说明书与测试图为可选资产。

```
<算法名>-dockerimg_<版本>/
├── Dockerfile            # FROM 基础镜像 + ENV 注入 + COPY 算法目录
├── build.sh              # install|restart|stop|delete|help
├── readme.txt            # 组件名称 / 手动安装 / 自动部署 / 测试 / 相关信息
├── packages/             # 额外依赖 wheel(无依赖时含 README.txt)
├── testimage.jpg         # 可选, 来自 --assets-dir
├── <接口说明书>.docx       # 可选, 来自 --assets-dir
└── <算法名>/              # ★ 算法同名目录: 代码 + 模型产物 + 配置平铺
    ├── nwai_*.py         # 公共 Runtime(与模型无关, 所有模型共用)
    ├── send_log_webserver.py
    ├── send_log_settings.json
    ├── runtime.yaml      # Runtime 运行参数(端口/输入输出契约/类别表)
    ├── confidence.json   # 类别表(model_id/export_id/name/confidence/is_export)
    ├── model.xpu         # 编译产物(Paddle 静态图为 .pdmodel + .pdiparams)
    ├── metadata.json     # 转换/编译元数据 + 来源环境指纹
    ├── artifact.json     # 产物来源与编译信息
    ├── model.yaml        # 本次转换的 Manifest(如何产生这个模型)
    └── start.sh          # 容器入口: 校验产物 + 启动 Runtime
```

`Dockerfile` 以 `COPY <算法名>/ /usr/local/<算法名>/` 引入算法目录，
`WORKDIR /usr/local/<算法名>` 后由 `CMD ["/bin/bash", "start.sh"]` 启动。

`packages/` 的依赖轮子来源由 `configs/deployment/kunlun_docker.yaml` 的
`docker.packages_dir` 指定（留空则写入占位 `README.txt`），Dockerfile 内以
`pip install /tmp/packages/*.whl --no-deps` 安装，无需联网。

接口说明书 `.docx` 与 `testimage.jpg` 由 `docker.assets_dir`（或 CLI `--assets-dir`）
指定的目录提供，导出时复制到包顶层；目录缺失或未命中时跳过并打印告警，不影响打包。

## 昆仑芯 Backend 的 SDK 解耦约定

在 XPU 型号、SDK/XTCL 版本确定之前，**不假设任何具体厂商 API**。`KunlunBackend.compile()`
只依赖 `XpuGraph`（已固化静态 shape 的待编译图），真实 SDK 调用被隔离在
`backend/kunlun/compiler.py` 的 `KunlunSdkAdapter` 子类中：

| `--sdk-adapter` | 行为 |
| --- | --- |
| `auto` | 按优先级自动探测（默认） |
| `xpuctl` | 调用 XPU Toolkit / XTCL 编译 ONNX |
| `paddle` | 复用镜像内 Paddle Inference 的 XPU 能力 |
| `stub` | SDK 未就绪时的占位产物，标记 `degraded`，**禁止对外交付** |

探测不到 SDK 时会降级为 `stub`，并把 `degraded=true` 贯穿 `metadata.json` / `Dockerfile` /
`readme.txt` / `start.sh`，避免占位件被误当成正式交付物。拿到昆仑 SDK 信息后，只需补全
对应适配器的 `compile` / `open_session`，上层流水线与交付层无需改动。

## V1 范围

PyTorch / YOLOv10 / Detection / NCHW / 静态 shape / batch 1 / FP32→FP16 / 昆仑 XPU /
Raw Detection 输出（**NMS 在 CPU**）/ Flask+Gunicorn / `/predict` + `/health` /
Docker ZIP / PyTorch-ONNX-XPU 三路校验。

第二阶段：YOLOv8 / YOLOv9 / YOLO11（配置已就绪）。第三阶段：PaddleOCR / PaddleDetection。

## 测试

```bash
python -m unittest discover -s tests -t .
```

无 torch / ultralytics 环境下，测试使用**合成 ONNX 图**验证优化、改写、编译、校验、
打包全链路（数值等价性由 onnxruntime 保证）。
