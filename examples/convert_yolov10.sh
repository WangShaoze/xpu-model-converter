#!/usr/bin/env bash
# 一条命令完成: best.pt -> ONNX -> XPU -> Docker 交付包(建设目标 §2)
#
# 最终产出: ./output/yolov10n_dockerimg_v1.0.zip
set -euo pipefail

xpu-converter convert \
    --model ./best.pt \
    --model-type yolov10 \
    --hardware kunlun \
    --input-shape 1,3,640,640 \
    --precision fp16 \
    --output ./output

# 等价写法(使用 Manifest, 见建设目标 §14):
# xpu-converter convert --manifest ./yolov10n.yaml --output ./output

# 只做单步时可以按需拆分:
# xpu-converter inspect   best.pt
# xpu-converter export-onnx --model best.pt --model-type yolov10 --output ./output/onnx
# xpu-converter analyze   ./output/onnx/yolov10_raw.onnx --device kunlun
# xpu-converter compile   --model ./output/onnx/yolov10_optimized.onnx --device kunlun --precision fp16
# xpu-converter validate  --source best.pt --target ./output/xpu/model.xpu --dataset ./testdata
# xpu-converter package   --model ./output/xpu/model.xpu --model-type yolov10 --output ./package
