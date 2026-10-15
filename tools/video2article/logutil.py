"""本模块的日志设施。

**为什么不直接复用一览成文的 logging_config.py**：

原版 `configure_logging()` 会 `logging.getLogger()`（root）上摘掉全部 handler 再装
自己的 —— 因为它假设自己是**整个进程**里唯一的日志来源。

但本模块是**寄生在 GPT-SoVITS WebUI 里的一个 tab**：webui.py 自己往 root 挂了
Gradio 的输出捕获，`asr`、`slicer2` 等子进程也共用 root。如果这里再动 root，
表现就是「点一下成文按钮，整个 WebUI 的日志/进度条全没了」，而且极难定位。

所以这里只给**自己的** logger 挂 handler，root 一个都不碰。
需要看日志的地方有两个：
- `logs/video2article.log`
- 本 tab 界面上的日志框（靠 handler 里注册的 sink，见 `attach_sink`）
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List

LOGGER_NAME = "video_to_article"

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
_DATE_FORMAT = "%H:%M:%S"

# 界面日志框用的 sink 列表（每项是一个单参数 callable，接收格式化后的字符串）
_SINKS: List[Callable[[str], None]] = []

_configured = False


def _log_file() -> Path:
    """日志文件路径：GPT-SoVITS 根目录下的 logs/video2article.log。"""
    from .paths import LOGS_DIR

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    return LOGS_DIR / "video2article.log"


class _SinkHandler(logging.Handler):
    """把日志转发给界面上的文本框。"""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            text = self.format(record)
        except Exception:
            return
        for sink in list(_SINKS):
            try:
                sink(text)
            except Exception:
                # sink 是 UI 回调，抛异常绝不能影响主流程
                pass


def configure_logging(name: str = LOGGER_NAME) -> logging.Logger:
    """取得本模块的 logger（幂等）。

    刻意**不**碰 root logger，理由见模块 docstring。
    """
    global _configured
    logger = logging.getLogger(name)
    if _configured:
        return logger

    logger.setLevel(logging.INFO)
    # 不往 root 冒泡，否则 webui.py 的 root handler 会重复打一遍
    logger.propagate = False

    fmt = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    try:
        file_handler = logging.FileHandler(_log_file(), encoding="utf-8", errors="replace")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    except OSError:
        # 日志文件打不开不该让功能不可用（例如只读目录 / 便携包被占用）
        pass

    sink_handler = _SinkHandler()
    sink_handler.setFormatter(fmt)
    logger.addHandler(sink_handler)

    _configured = True
    return logger


def attach_sink(sink: Callable[[str], None]) -> None:
    """注册一个日志回调（WebUI 用）。返回的函数用于取消注册。"""
    _SINKS.append(sink)

    def _remove() -> None:
        try:
            _SINKS.remove(sink)
        except ValueError:
            pass

    return _remove


def detach_all_sinks() -> None:
    """清空所有 sink（测试用 / 界面重建时用）。"""
    _SINKS.clear()


def get_logger(suffix: str = "") -> logging.Logger:
    """子模块专用：``get_logger("asr")`` → logger ``video2article.asr``。"""
    return configure_logging(f"{LOGGER_NAME}.{suffix}" if suffix else LOGGER_NAME)
