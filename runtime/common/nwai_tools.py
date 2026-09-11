# -*- coding: utf-8 -*-
"""公共工具: 客户响应结构、时间、日志上报开关。

本文件属于 **公共 Runtime**(建设目标 §5/§10), 所有模型/任务共用, 与具体模型无关。
"""
import datetime
import json
import os

import numpy as np

try:  # 容器内已安装 pytz; 本地缺失时退化为系统时区
    import pytz

    LOCAL_TIMEZONE = pytz.timezone("Asia/Shanghai")
except ImportError:  # pragma: no cover
    LOCAL_TIMEZONE = None

CUR_DIR = os.path.abspath(os.path.dirname(__file__))
SEND_LOG_SETTINGS_FILE = os.path.join(CUR_DIR, "send_log_settings.json")
# 容器内本地日志转发服务端口(转投 MINIO/KAFKA)
SEND_LOG_WEBSERVER_PORT = 20000
SEND_LOG_WEBSERVER_URL = "http://127.0.0.1:{}/log".format(SEND_LOG_WEBSERVER_PORT)

# 客户规范返回码
RETURN_CODE_OK = 0
RETURN_CODE_PARAM_ERROR = -1
RETURN_CODE_NO_RESULT = -2
RETURN_CODE_INTERNAL_ERROR = -3
RETURN_CODE_UNSUPPORTED = -7


class JsonEncoder(json.JSONEncoder):
    """把 numpy 类型转换为可 JSON 序列化对象。"""

    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(JsonEncoder, self).default(obj)


def ExceptionMessage(err):
    """异常信息文本化。"""
    return "{}: {}".format(type(err).__name__, err)


def GetResultInfo(algorithm="", algorithm_name="", algorithm_version="", algorithm_type="",
                  algorithm_app_type="", code=RETURN_CODE_OK, message="", data=None, task_id="",
                  algorithm_app_type_value="", component_code="", template_version=""):
    """构造与"南网人工智能组件"规范一致的响应体。"""
    return {
        "task_id": task_id,
        "algorithm_app_type_value": algorithm_app_type_value,
        "component_code": component_code,
        "template_version": template_version,
        "algorithm": algorithm,
        "algorithm_name": algorithm_name,
        "algorithm_version": algorithm_version,
        "algorithm_type": algorithm_type,
        "algorithm_app_type": algorithm_app_type,
        "code": code,
        "message": message,
        "data": data,
    }


def Jsonify(result_info):
    """响应体 -> utf-8 bytes。"""
    return json.dumps(result_info, ensure_ascii=False, indent=4, cls=JsonEncoder).encode("utf-8")


def WriteLog(process_log_file_path, algorithm_log_str, max_log_size=300):
    """追加写日志文件, 超过 ``max_log_size`` MB 时先删除。"""
    try:
        if os.path.exists(process_log_file_path) and \
                os.path.getsize(process_log_file_path) / (1024 ** 2) > max_log_size:
            os.remove(process_log_file_path)
        if not isinstance(algorithm_log_str, str):
            algorithm_log_str = json.dumps(algorithm_log_str, ensure_ascii=False, cls=JsonEncoder)
        with open(process_log_file_path, "a+", encoding="utf-8") as fa:
            fa.write(algorithm_log_str + "\n")
    except Exception:
        return


def GetDateTime():
    """返回 ``(当前时间, 紧凑时间戳, 年, 月, 日)``, 统一使用东八区。"""
    date = datetime.datetime.now()
    if LOCAL_TIMEZONE is not None:
        date = date.astimezone(LOCAL_TIMEZONE)
    current_datetime = str(date)[:-13]
    today_strftime = date.strftime("%Y%m%d%H%M%S")
    return current_datetime, today_strftime, str(date.year), str(date.month), str(date.day)


def LogDetailFormat(algorithm="", task_id="", detail="", message="", code=RETURN_CODE_OK):
    """统一日志明细结构。"""
    current_datetime, today_strftime, year, month, day = GetDateTime()
    return {
        "algorithm": algorithm,
        "task_id": task_id,
        "code": code,
        "message": message,
        "detail": detail,
        "date": current_datetime,
        "timestamp": today_strftime,
        "year": year,
        "month": month,
        "day": day,
    }


def SendLogEnabled():
    """日志上报总开关(``send_log_settings.json`` 中 ``SEND_LOG_ENABLED=false`` 时关闭)。"""
    try:
        with open(SEND_LOG_SETTINGS_FILE, "r", encoding="utf-8") as fr:
            return bool(json.load(fr).get("SEND_LOG_ENABLED", True))
    except Exception:
        return False


def ApplicationJsonMode(url, log_str, timeout=0.01):
    """通过容器本地 send_log_webserver 转投日志到 MINIO/KAFKA。

    返回 ``(是否成功, 说明)``; 上报失败 **不影响** 识别服务。
    """
    if not SendLogEnabled():
        return True, "send log disabled"
    try:
        import urllib3

        urllib3.disable_warnings()
        http = urllib3.PoolManager(timeout=timeout)
        # 必须 encode, 否则中文会报错
        response = http.request(method="POST", url=url,
                                headers={"Content-Type": "application/json"},
                                body=log_str.encode())
        if getattr(response, "status", 200) >= 400:
            return False, "send log http status {}".format(response.status)
    except Exception as err:
        return False, "[Error] send log failed. err_message:{}".format(ExceptionMessage(err))
    return True, "normal"
