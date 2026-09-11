# -*- coding: utf-8 -*-
"""Docker 交付包端到端验收(ChatGPT 修改意见 §46)。

真实验收链路::

    best.pt → Docker ZIP → docker build → docker run → curl /health → curl /predict

这是最终交付的验收测试, 依赖真实 XPU 机器与 docker, 因此**默认 skip**; 只有显式
提供环境变量时才执行:

    XPU_E2E=1                     打开 E2E
    XPU_E2E_PACKAGE=<dir>         已由 `xpu-converter build` 生成的交付包目录
    XPU_E2E_IMAGE=<tag>           可选, 构建出的镜像 tag(默认 xpu-e2e:latest)
    XPU_E2E_PORT=<port>           可选, 宿主机映射端口(默认 58025)
    XPU_E2E_DOCKER_ARGS="..."     可选, 追加给 `docker run` 的参数(如 --device)
"""
import base64
import json
import os
import shutil
import subprocess
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

# 1x1 PNG, 仅用于打通 /predict 请求链路
TINY_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)

E2E_ENABLED = os.environ.get("XPU_E2E", "") == "1"
PACKAGE_DIR = os.environ.get("XPU_E2E_PACKAGE", "")
IMAGE_TAG = os.environ.get("XPU_E2E_IMAGE", "xpu-e2e:latest")
HOST_PORT = int(os.environ.get("XPU_E2E_PORT", "58025"))
EXTRA_RUN_ARGS = os.environ.get("XPU_E2E_DOCKER_ARGS", "")


def _e2e_ready() -> bool:
    return bool(E2E_ENABLED and PACKAGE_DIR and Path(PACKAGE_DIR).is_dir()
                and shutil.which("docker"))


@unittest.skipUnless(_e2e_ready(), "需要 XPU_E2E=1 + XPU_E2E_PACKAGE + docker")
class DockerEndToEndTest(unittest.TestCase):
    """构建镜像 → 启动容器 → 健康检查 → 推理一次 → 清理。"""

    container = ""

    def setUp(self) -> None:
        self.container = "xpu-e2e-{}".format(int(time.time()))
        self._docker("build", "-t", IMAGE_TAG, ".")

    def tearDown(self) -> None:
        if self.container:
            subprocess.run(["docker", "rm", "-f", self.container],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

    def test_health_and_predict(self) -> None:
        args = ["run", "-d", "--name", self.container, "-p", "{}:{}".format(HOST_PORT, HOST_PORT)]
        if EXTRA_RUN_ARGS:
            args.extend(EXTRA_RUN_ARGS.split())
        args.append(IMAGE_TAG)
        self._docker(*args)

        self._wait_health()
        payload = json.dumps({"base64": TINY_PNG_BASE64, "code": "e2e"}).encode("utf-8")
        request = urllib.request.Request(
            self._url("/predict"), data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8"))
        self.assertEqual(response.status, 200)
        self.assertEqual(str(body.get("code")), "0", "predict 返回非成功码: {}".format(body))

    # ------------------------------------------------------------------ 内部
    def _docker(self, *args) -> str:
        result = subprocess.run(
            ["docker", *args], cwd=PACKAGE_DIR,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, "docker {} 失败:\n{}".format(args[0], result.stdout))
        return result.stdout.strip()

    def _url(self, path: str) -> str:
        return "http://127.0.0.1:{}{}".format(HOST_PORT, path)

    def _wait_health(self, timeout: float = 300.0, interval: float = 5.0) -> None:
        deadline = time.time() + timeout
        last_error = ""
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(self._url("/health"), timeout=10) as response:
                    if response.status == 200:
                        return
            except (urllib.error.URLError, OSError) as err:
                last_error = str(err)
            time.sleep(interval)
        logs = subprocess.run(["docker", "logs", self.container], stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, check=False).stdout
        self.fail("容器健康检查超时({}s): {}\n--- docker logs ---\n{}".format(timeout, last_error, logs))
