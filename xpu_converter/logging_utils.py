# -*- coding: utf-8 -*-
"""日志与流水线步骤输出。"""
import logging
import sys

LOGGER_NAME = "xpu_converter"


def get_logger(name: str = LOGGER_NAME) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def setup_logging(level: str = "INFO") -> logging.Logger:
    logger = get_logger()
    logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    return logger


class StepReporter:
    """按 "[01] 标题 .... OK" 的形式输出流水线步骤（见建设目标 §2）。"""

    def __init__(self, total: int = 0, stream=None):
        self.total = total
        self.index = 0
        self._stream = stream
        self.records = []

    @property
    def stream(self):
        """延迟解析输出流, 保证 stdout 重定向(CLI / 测试)能够生效。"""
        return self._stream or sys.stdout

    def step(self, title: str, ok: bool = True, detail: str = "", skipped: bool = False) -> None:
        self.index += 1
        padded = "[{:02d}]".format(self.index)
        if skipped:
            status = "SKIPPED"
        else:
            status = "OK" if ok else "FAIL"
        line = "{} {} {}".format(padded, title, status)
        print(line, file=self.stream)
        if detail:
            for raw in str(detail).splitlines():
                print("     {}".format(raw), file=self.stream)
        self.records.append({
            "index": self.index, "title": title, "ok": bool(ok),
            "skipped": bool(skipped), "detail": detail,
        })

    def info(self, text: str) -> None:
        print("     {}".format(text), file=self.stream)

    @property
    def all_ok(self) -> bool:
        return all(r["ok"] for r in self.records)
