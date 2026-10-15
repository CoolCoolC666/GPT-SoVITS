"""video2article — 视频/音频转文章的流水线（移植自「一览成文 YilanChengWen」）。

给 GPT-SoVITS 补上「读完之后写」的一环：

    链接 / 本地音视频
        → （yt-dlp 下载，可选）
        → ffmpeg 抽音
        → 云端 ASR（讯飞听见 / custom_post）
        → LLM 成文（prompts/articles 模板）
        → Markdown 文章落盘

移植来源：CoolCoolC666/video-to-article（MIT，本仓库同一作者）。
移植时**刻意**没动三个东西，原因见各文件顶部注释：

- 日志不再劫持 root logger（一览成文原版会，本仓库 webui.py 已经在用 root）
- 配置改为只读外链，不复制任何含密钥的 config.json
- ffmpeg 优先用 GPT-SoVITS 自带的 runtime/ffmpeg.exe
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
