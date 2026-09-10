# 模型转换平台自身的运行镜像(建设目标 §4: 仓库根 Dockerfile)
#
# 用途: 在无 GPU 的构建机上完成 best.pt -> ONNX -> (XPU 产物) -> Docker 交付包。
# 说明: 昆仑 SDK 不在本镜像内, sdk_adapter 默认探测失败时降级为 stub 占位产物;
#       真实的 XPU 编译请在装有昆仑 SDK 的环境中执行, 或把 SDK 挂载/装进本镜像。
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    XPU_CONVERTER_HOME=/opt/xpu-model-converter

WORKDIR /opt/xpu-model-converter

# ONNX / 校验 / 模板渲染等基础依赖
RUN pip install --no-cache-dir \
        "numpy>=1.21" "PyYAML>=5.4" "Jinja2>=3.0" \
        "onnx>=1.12" "onnxruntime>=1.13"

# PyTorch(CPU 版) 用于 PyTorch 前端导出 ONNX
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch>=1.13"

COPY pyproject.toml README.md ./
COPY xpu_converter/ ./xpu_converter/
COPY runtime/ ./runtime/
COPY templates/ ./templates/
COPY configs/ ./configs/
COPY examples/ ./examples/

RUN pip install --no-cache-dir -e .

# 默认: 挂载待转换模型到 /work, 在其上执行转换
WORKDIR /work

ENTRYPOINT ["xpu-converter"]
CMD ["--help"]
