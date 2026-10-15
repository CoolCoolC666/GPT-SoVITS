"""把一览成文的两个云端 ASR 引擎移植进 GPT-SoVITS。

一次性脚本（跑完即弃）：只做「复制 + 改导入 + 换 ffmpeg 定位」，
不碰任何协议逻辑——原实现踩过的坑都写在注释里，重写等于重新踩一遍。

用法：runtime/python.exe _port_asr.py
"""

from __future__ import annotations

import re
from pathlib import Path

SRC = Path(r"E:\000~\YilanChengWen-src\src\video_to_article\media")
DST = Path(r"E:\AI_Models\GPT-SoVITS-v2pro-20261006nvidia50x0\tools\video2article")

HEADER = '''"""{title}

=========================== 移植说明 ===========================

本文件由 CoolCoolC666/video-to-article（MIT）的
src/video_to_article/media/{origin} 移植而来，只改了四处：

1. 相对导入改为本模块内的绝对导入（logging_config -> logutil 等）
2. ffmpeg/ffprobe 定位改为复用 media.py（优先 GPT-SoVITS 自带的
   runtime/ffmpeg.exe），原版硬编码的一览成文安装路径已删除
3. 日志不再劫持 root logger
4. 原 docstring 里的「本 fork 改动」章节已删除（那是另一个仓库的说明）

**协议逻辑一行未改。** 原实现踩过的坑都写在下面各处注释里
（字段名是 orderId 不是 taskId、状态是 content.orderInfo.status 不是
taskStatus、orderResult 是 JSON 字符串、上传必须 query string + raw body
而不是 multipart……）。改动前请先读那些注释。
===============================================================
"""

'''

FFMPEG_NEW = (
    'def _find_ffmpeg() -> Optional[str]:\n'
    '    """复用 media.py 的定位逻辑（GPT-SoVITS 自带 runtime/ffmpeg.exe 优先）。"""\n'
    '    from .media import find_ffmpeg\n'
    '\n'
    '    return find_ffmpeg()\n'
)

FFPROBE_NEW = (
    'def _find_ffprobe() -> Optional[str]:\n'
    '    """复用 media.py 的定位逻辑。"""\n'
    '    from .media import find_ffprobe\n'
    '\n'
    '    return find_ffprobe()\n'
)


def strip_leading_docstring(text: str) -> str:
    """去掉原文件开头的模块 docstring（只处理紧贴文件头的第一个三引号块）。"""
    if not text.startswith('"""'):
        return text
    end = text.find('"""', 3)
    if end == -1:
        return text
    return text[end + 3:].lstrip("\n")


def port(origin: str, name: str, title: str) -> None:
    src = SRC / origin
    text = src.read_text(encoding="utf-8")
    original_lines = len(text.splitlines())

    # 1) 导入改写
    text = text.replace(
        "from ..logging_config import configure_logging",
        "from .logutil import configure_logging",
    )
    text = text.replace(
        "from ..text_utils import format_time, import_required",
        "from .text_utils import format_time, import_required",
    )
    text = text.replace(
        "from ..text_utils import format_time",
        "from .text_utils import format_time",
    )
    text = text.replace(
        "from ..paths import APP_ROOT",
        "from .paths import CUSTOM_POST_HEADERS_FILE",
    )
    text = text.replace(
        "return Path(APP_ROOT) / HEADERS_FILENAME",
        "return Path(CUSTOM_POST_HEADERS_FILE)",
    )

    # 2) ffmpeg 候选表整体换成 media.py 委托
    text = re.sub(r"_FFMPEG_PATHS = \[.*?\]\n", "", text, flags=re.S)
    text = re.sub(r"_FFPROBE_PATHS = \[.*?\]\n", "", text, flags=re.S)
    text = re.sub(
        r"def _find_ffmpeg\(\) -> Optional\[str\]:.*?\n(?=\n\ndef |\n\n#)",
        FFMPEG_NEW,
        text,
        flags=re.S,
    )
    text = re.sub(
        r"def _find_ffprobe\(\) -> Optional\[str\]:.*?\n(?=\n\ndef |\n\n#)",
        FFPROBE_NEW,
        text,
        flags=re.S,
    )

    out = HEADER.format(title=title, origin=origin) + strip_leading_docstring(text)
    DST.mkdir(parents=True, exist_ok=True)
    (DST / name).write_text(out, encoding="utf-8")

    leftovers = [
        line
        for line in out.splitlines()
        if ".." in line.split("#")[0][:40] and ("import" in line)
    ]
    print(f"{name}: {original_lines} -> {len(out.splitlines())} 行")
    if leftovers:
        print("  !! 残留相对导入：")
        for line in leftovers:
            print("     ", line)
    for token in ("_FFMPEG_PATHS", "_FFPROBE_PATHS", "APP_ROOT"):
        if token in out:
            print(f"  !! 仍存在 {token}")


def main() -> None:
    port(
        "xf_asr.py",
        "asr_xf.py",
        "讯飞听见云端 ASR（长语音转写 REST API）。",
    )
    port(
        "custom_post_asr.py",
        "asr_custom_post.py",
        "自定义 POST 云端 ASR（OpenAI 兼容 / DashScope 异步）。",
    )


if __name__ == "__main__":
    main()
