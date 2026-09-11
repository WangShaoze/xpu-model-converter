# -*- coding: utf-8 -*-
"""检测任务运行时配置与 gunicorn 钩子。

配置优先级: 环境变量(docker run -e) > 算法目录下 ``runtime.yaml`` > 内置默认值。
"""
import os

import nwai_config
from nwai_devices import DevicePool
from nwai_tools import GetDateTime, WriteLog

CONFIG = nwai_config.load_runtime_config()
_RUNTIME = CONFIG.get("runtime") or {}
_POST = CONFIG.get("postprocess") or {}
_PRE = CONFIG.get("preprocess") or {}
_INPUT = CONFIG.get("input") or {}
_ALGORITHM = nwai_config.algorithm_info(CONFIG)

MODEL_NAME = str(CONFIG.get("name") or "model")
MODEL_VERSION = str(CONFIG.get("version") or "v1.0")
TASK = str(CONFIG.get("task") or "detection")

WEB_PORT = int(os.environ.get("WEB_PORT") or _RUNTIME.get("port") or 58025)
WORKERS = int(os.environ.get("workers") or os.environ.get("WORKERS") or _RUNTIME.get("workers") or 1)
DEVICE = str(os.environ.get("DEVICE") or _RUNTIME.get("device") or "auto")
HOST_IP = str(os.environ.get("HOST_IP") or _RUNTIME.get("host_ip") or "127.0.0.1")

CONF_THRES = float(os.environ.get("CONF_THRES") or _POST.get("conf_thres") or 0.25)
IOU_THRES = float(os.environ.get("IOU_THRES") or _POST.get("iou_thres") or 0.45)
MAX_DET = int(_POST.get("max_det") or 300)
OUTPUT_LAYOUT = str(_POST.get("output_layout") or "auto")
# 输出契约: 由转换器对真实 ONNX 探测后写入 runtime.yaml 的 output 段。
# 缺失时解码器会显式报错, 不再用 shape 启发式猜测(ChatGPT 修改意见 §16)。
OUTPUT_CONTRACT = dict(CONFIG.get("output") or _POST.get("output") or {})

INPUT_SHAPE = [int(item) for item in (_INPUT.get("shape") or [1, 3, 640, 640])]
INPUT_LAYOUT = str(_INPUT.get("layout") or "NCHW")
PREPROCESS = dict(_PRE)
NUM_CLASSES = int(CONFIG.get("num_classes") or 80)

RESULT_DIR = str(_RUNTIME.get("result_dir") or "/tmp/nwai_result")
LOG_DIR = os.environ.get("NWAI_LOG_DIR", "/tmp/nwai_log")
PROCESS_LOG_FILE = os.path.join(LOG_DIR, "process.log")
SETFLAG_FILE = os.path.join(LOG_DIR, "setflag.json")

ALGORITHM = _ALGORITHM["algorithm"]
ALGORITHM_NAME = _ALGORITHM["algorithm_name"]
ALGORITHM_VERSION = _ALGORITHM["algorithm_version"]
ALGORITHM_TYPE = _ALGORITHM["algorithm_type"]
ALGORITHM_APP_TYPE = _ALGORITHM["algorithm_app_type"]
COMPONENT_CODE = _ALGORITHM["component_code"]
TEMPLATE_VERSION = _ALGORITHM["template_version"]

# gunicorn 各 worker 的 XPU 卡分配池
DEVICE_POOL = DevicePool(device_mode=DEVICE)
_WORKER_DEVICES = {}


def os_makedirs():
    for path in (LOG_DIR, RESULT_DIR):
        try:
            os.makedirs(path, exist_ok=True)
        except Exception:
            pass


def log(message):
    WriteLog(PROCESS_LOG_FILE, "[{}] {}".format(GetDateTime()[0], message))


def worker_device_key(worker):
    """worker 的唯一标识(优先 pid, 兼顾 pre_fork 阶段无 pid 的情况)。"""
    return getattr(worker, "pid", None) or getattr(worker, "age", 0)


# --------------------------------------------------------------------- gunicorn 钩子
def on_starting(server):
    """master 启动前: 探测设备并打印部署信息。"""
    os_makedirs()
    log("starting {} v{} workers={} device={} xpu_cards={} port={}".format(
        MODEL_NAME, MODEL_VERSION, WORKERS, DEVICE, DEVICE_POOL.device_count, WEB_PORT))


def pre_fork(server, worker):
    """为即将 fork 的 worker 预分配一张卡。"""
    device = DEVICE_POOL.acquire()
    _WORKER_DEVICES[worker_device_key(worker)] = device
    log("pre_fork worker={} device={}".format(worker_device_key(worker), device))


def post_fork(server, worker):
    """worker 内设置设备环境变量(引擎在首次请求时按此加载)。"""
    device = _WORKER_DEVICES.get(worker_device_key(worker))
    if device is not None:
        os.environ["GPU_ID"] = str(device)
        os.environ["DEVICE"] = "xpu"
    else:
        os.environ["DEVICE"] = "cpu"
    log("post_fork pid={} device={}".format(getattr(worker, "pid", "-"), device))


def child_exit(server, worker):
    """worker 退出: 归还设备。"""
    key = worker_device_key(worker)
    DEVICE_POOL.release(_WORKER_DEVICES.pop(key, None))


def get_gunicorn_options():
    """返回 gunicorn 选项(含钩子)。"""
    workers = DEVICE_POOL.worker_count(WORKERS)
    return {
        "app_uri": "nwai_webserver:app",
        "bind": "0.0.0.0:{}".format(WEB_PORT),
        "workers": workers,
        "worker_class": "sync",
        "timeout": 120,
        "graceful_timeout": 30,
        "keepalive": 5,
        "preload_app": False,
        "accesslog": os.path.join(LOG_DIR, "access.log"),
        "errorlog": os.path.join(LOG_DIR, "error.log"),
        "loglevel": os.environ.get("LOG_LEVEL", "info"),
        "on_starting": on_starting,
        "pre_fork": pre_fork,
        "post_fork": post_fork,
        "child_exit": child_exit,
    }
