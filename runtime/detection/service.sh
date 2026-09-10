#!/bin/bash
# =====================================================================
# 算法服务启动脚本(公共 Runtime, 交付产物 ②)
# 启动顺序: 先拉起日志上报服务(send_log_webserver, 端口20000), 再启动算法服务
# 设置 NWAI_FLASK_ONLY=1 时使用 flask 单进程模式(调试/无 gunicorn 场景)
# 本脚本与具体模型无关, 所有 YOLO 模型共用同一套 Runtime
# 注: 交付包内由 start.sh(入口脚本)调用本脚本; 单独使用 Runtime 时可直接执行本脚本
# =====================================================================
cd "$(dirname "$0")" || exit 1
export RUNTIME_HOME="$(pwd)"

nohup python send_log_webserver.py > /dev/null 2>&1 &

if [ -n "${NWAI_FLASK_ONLY+x}" ]; then
    exec python runtime_server.py
else
    exec python runtime_gunicorn.py
fi
