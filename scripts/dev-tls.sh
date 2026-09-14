#!/usr/bin/env bash
# 开发环境 TLS(HTTPS)自签证书生成脚本(§46 传输层加密)。
#
# 用途: 本地以 HTTPS 启动 FastAPI, 验证密码加密登录全链路。
# 注意: 自签证书浏览器会提示"不安全", 仅限本机开发; 生产必须使用
#       CA 签发证书(见 deploy/Caddyfile, Caddy 自动申请 Let's Encrypt)。
#
# 用法:
#   bash scripts/dev-tls.sh
#   uv run uvicorn xpu_platform.api.main:app \
#     --host 0.0.0.0 --port 8443 \
#     --ssl-keyfile dev-certs/key.pem --ssl-certfile dev-certs/cert.pem
#
#   前端指向 HTTPS 网关:
#   NEXT_PUBLIC_API_BASE_URL=https://localhost:8443 npm run dev
set -euo pipefail

OUT_DIR="dev-certs"
mkdir -p "${OUT_DIR}"

if command -v openssl >/dev/null 2>&1; then
  openssl req -x509 -newkey rsa:2048 -nodes \
    -keyout "${OUT_DIR}/key.pem" \
    -out "${OUT_DIR}/cert.pem" \
    -days 365 \
    -subj "/CN=localhost" \
    -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"
  chmod 600 "${OUT_DIR}/key.pem"
  echo "已生成自签证书: ${OUT_DIR}/cert.pem / ${OUT_DIR}/key.pem"
else
  echo "未找到 openssl, 请安装后重试" >&2
  exit 1
fi
