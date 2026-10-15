"""ffmpeg 定位与音频抽取。

**为什么 ffmpeg 查找顺序和一览成文不一样**：一览成文找的是它自己 `ffmpeg/` 目录
下的副本；本模块寄生在 GPT-SoVITS 里，而 GPT-SoVITS 的便携包**已经自带**
`runtime/ffmpeg.exe` 和 `runtime/ffprobe.exe`。优先用它，等于零新增依赖。

**为什么抽取成 16kHz 单声道**：两个云端引擎都对输入有偏好，且体积直接决定上传耗时：
- 讯飞长语音：单文件 ≤500MB，抽成 16k 单声道 mp3 通常能压到原视频的 1~2%
- custom_post（OpenAI 兼容风格）：单请求 ≤500 秒 / 50MB，必须先压
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from .logutil import get_logger
from .paths import APP_ROOT, OUTPUT_DIR, RUNTIME_DIR

logger = get_logger("media")

# ffmpeg 候选：GPT-SoVITS 自带 → 项目根 → PATH
_FFMPEG_CANDIDATES = (
    RUNTIME_DIR / "ffmpeg.exe",
    APP_ROOT / "ffmpeg.exe",
)
_FFPROBE_CANDIDATES = (
    RUNTIME_DIR / "ffprobe.exe",
    APP_ROOT / "ffprobe.exe",
)


class MediaError(RuntimeError):
    """媒体处理失败（ffmpeg 缺失 / 抽音失败 / 探测失败）。"""


def find_ffmpeg() -> Optional[str]:
    for p in _FFMPEG_CANDIDATES:
        if p.is_file():
            return str(p)
    return shutil.which("ffmpeg")


def find_ffprobe() -> Optional[str]:
    for p in _FFPROBE_CANDIDATES:
        if p.is_file():
            return str(p)
    return shutil.which("ffprobe")


def probe_duration(path: str | Path) -> float:
    """探测时长（秒）。失败返回 -1.0，**不抛异常**——调用方据此决定要不要分片。"""
    ffprobe = find_ffprobe()
    if not ffprobe:
        logger.warning("ffprobe 找不到，跳过时长探测")
        return -1.0
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "json", str(path)],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            logger.warning("ffprobe 失败 (code=%s)，无法探测时长", result.returncode)
            return -1.0
        return float(json.loads(result.stdout)["format"]["duration"])
    except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError, ValueError, OSError) as e:
        logger.warning("ffprobe 异常: %s", e)
        return -1.0


def _run_ffmpeg(args: list[str]) -> None:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise MediaError(
            "找不到 ffmpeg。GPT-SoVITS 便携包自带 runtime/ffmpeg.exe，"
            "请确认该文件存在；或安装系统版 ffmpeg 并加入 PATH。"
        )
    cmd = [ffmpeg, "-y", "-loglevel", "error", *args]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired as e:
        raise MediaError("ffmpeg 执行超时（超过 30 分钟）") from e
    if result.returncode != 0:
        raise MediaError(f"ffmpeg 失败 (code={result.returncode}): {result.stderr[:400]}")


def extract_audio(
    media_path: str | Path,
    out_dir: str | Path | None = None,
    sample_rate: int = 16000,
) -> Path:
    """从视频/音频里抽成人声轨音频，返回生成的 mp3 路径。

    已经是纯音频且体积不大时**仍然转码**：两个云端引擎都吃 mp3/wav，
    统一转一遍可以避免「mkv 里是 aac、云端不收」这类问题。
    """
    media_path = Path(media_path)
    if not media_path.is_file():
        raise MediaError(f"文件不存在: {media_path}")

    out_dir = Path(out_dir) if out_dir else OUTPUT_DIR / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{media_path.stem}.mp3"

    logger.info("抽取音频: %s → %s", media_path.name, out_path.name)
    _run_ffmpeg([
        "-i", str(media_path),
        "-vn",                       # 丢掉视频轨
        "-ac", "1",                  # 单声道
        "-ar", str(sample_rate),     # 16kHz：ASR 通用最佳采样率
        "-b:a", "64k",
        "-c:a", "libmp3lame",
        str(out_path),
    ])

    if not out_path.is_file() or out_path.stat().st_size < 1024:
        raise MediaError(f"ffmpeg 没有产出有效音频: {out_path}")
    logger.info("音频抽取完成 (%.2f MB)", out_path.stat().st_size / 1024 / 1024)
    return out_path


def format_size(num_bytes: int) -> str:
    mb = num_bytes / 1024 / 1024
    return f"{mb:.2f}MB" if mb < 1024 else f"{mb / 1024:.2f}GB"
