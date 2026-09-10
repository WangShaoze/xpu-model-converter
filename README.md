# xpu-model-converter

模型转换平台：把客户给的框架模型（`best.pt`）转成**昆仑芯 XPU 编译产物**，并自动封装为
可直接交付的 **Docker 部署包**（`<name>_dockerimg_v1.0.zip`）。

交付包结构与 `yolov9t_dockerimg_v1.0.zip` 保持**交付层兼容**，并补齐 `manifest.json`
作为唯一事实来源，消灭 `manifest ≠ Dockerfile ≠ README ≠ 实际文件` 的不一致问题。

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

最终得到 `./output/yolov10_dockerimg_v1.0.zip`。

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
Deployment Package (<name>_dockerimg_v1.0.zip)
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

runtime/            # 公共 Runtime(与模型无关, 所有 YOLO 共用一套)
├── common/         # 配置/日志/minio+kafka 上报/send_log_webserver/DevicePool/gunicorn
└── detection/      # 检测服务: /predict /predict_image /health /setflag

templates/docker/kunlun/   # Dockerfile / build.sh / install.conf / readme.txt / start.sh 模板
configs/models/            # 各模型配置(yolov8/9/10/11)
configs/hardware/          # 硬件后端配置(kunlun)
examples/                  # Manifest 与一键脚本示例
tests/                     # 单元测试与端到端测试
```

## 交付包结构

```
<name>_dockerimg_v1.0/
├── Dockerfile            # 全部由 manifest 渲染, 不硬编码模型名/端口
├── build.sh              # install|stop|restart|delete|help
├── install.conf          # 现场可覆盖的部署参数
├── readme.txt            # 与包内实际文件一致(镜像 tar 存在与否自动切换描述)
├── start.sh              # 容器入口: 校验产物 + 启动 Runtime
├── manifest.json         # ★ 唯一事实来源: 包名/镜像名/端口/文件清单/校验和
├── packages/             # 额外依赖 wheel(无依赖时含 README.txt)
├── runtime.tgz           # ★ 公共 Runtime(与模型无关)
├── model/
│   ├── model.xpu         # 编译产物
│   ├── model.yaml        # 本次转换的 Manifest
│   └── metadata.json     # 转换/编译元数据
└── config/
    ├── confidence.json   # 类别表(model_id/export_id/name/confidence/is_export)
    └── runtime.yaml      # Runtime 运行参数
```

`packages/` 的依赖轮子来源由 `configs/hardware/kunlun.yaml` 的 `docker.packages_dir`
指定（留空则写入占位 `README.txt`），Dockerfile 内以
`pip install /tmp/packages/*.whl --no-deps` 安装，无需联网。

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

探测不到 SDK 时会降级为 `stub`，并把 `degraded=true` 贯穿 `manifest.json` / `Dockerfile` /
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
