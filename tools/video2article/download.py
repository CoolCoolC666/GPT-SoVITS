"""视频下载（yt-dlp 薄封装）。

只做一件事：把一个视频链接变成一个本地文件（默认直接下**音频**，
因为后面链路只吃音频）。

**为什么默认只要音频**：下游是 ASR → 成文，拿视频流纯属浪费带宽和时间；
真需要留存视频的场合，用户可以自己在 UI 上关掉这个开关。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .logutil import get_logger
from .paths import OUTPUT_DIR
from .text_utils import sanitize_filename, sanitize_path_component

logger = get_logger("download")


class DownloadError(RuntimeError):
    """下载失败。消息里带上 yt-dlp 的原始报错，便于判断是网络还是平台限制。"""


def is_url(text: str) -> bool:
    t = str(text or "").strip().lower()
    return t.startswith(("http://", "https://"))


def download_media(
    url: str,
    out_dir: str | Path | None = None,
    audio_only: bool = True,
    cookies_from_browser: str = "",
    cookies_file: str = "",
    max_duration: int = 0,
) -> Path:
    """下载链接里的媒体，返回本地文件路径。

    Args:
        url: 视频链接
        out_dir: 输出目录，默认 ``output/video2article/download``
        audio_only: True 只要音频（m4a/mp3），False 下载最佳视频
        cookies_from_browser: 从浏览器读 cookies（如 "firefox" / "edge"）
        cookies_file: cookies.txt 路径
        max_duration: >0 时丢弃超过该秒数的媒体
    """
    url = str(url or "").strip()
    if not url:
        raise DownloadError("链接为空")
    if not is_url(url):
        raise DownloadError(f"不是有效的链接: {url}")

    out_dir = Path(out_dir) if out_dir else OUTPUT_DIR / "download"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import yt_dlp
    except ImportError as e:
        raise DownloadError(
            "未安装 yt-dlp，请运行: runtime/python.exe -m pip install yt-dlp"
        ) from e

    opts = {
        "outtmpl": str(out_dir / "%(title).60B.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "windowsfilenames": True,
        "restrictfilenames": False,
        "retries": 3,
        "fragment_retries": 3,
    }
    if audio_only:
        opts["format"] = "bestaudio/best"
        opts["postprocessors"] = [
            {"key": "FFmpegExtractAudio", "preferredcodec": "m4a", "preferredquality": "128"}
        ]
    if cookies_from_browser:
        opts["cookiesfrombrowser"] = (cookies_from_browser,)
    elif cookies_file:
        opts["cookiefile"] = str(cookies_file)
    if max_duration and max_duration > 0:
        # 用 yt-dlp 自带的 duration_limit，而不是手写 match_filter：
        # 手写那条要用 f-string 拼 lambda，超长媒体要在下载**开始前**就被拒，
        # 而 duration_limit 是官方语义（超限直接跳过，不落半截文件）。
        opts["duration_limit"] = int(max_duration)

    logger.info("开始下载: %s", url)
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if info is None:
                raise DownloadError("yt-dlp 没有返回结果（可能是链接失效或需要登录）")
            if "entries" in info:
                entries = [e for e in info["entries"] if e]
                if not entries:
                    raise DownloadError("链接里没有可下载的条目")
                info = entries[0]
    except DownloadError:
        raise
    except Exception as e:
        raise DownloadError(f"下载失败: {e}") from e

    path = _locate_downloaded(out_dir)
    if path is None:
        raise DownloadError(f"下载流程结束但没在 {out_dir} 找到文件")

    title = str(info.get("title") or "")[:80]
    logger.info("下载完成: %s (%s)", path.name, title or "未知标题")
    return path


def _locate_downloaded(out_dir: Path) -> Optional[Path]:
    """yt-dlp 不会把最终文件名告诉我们（后处理会改扩展名），所以取目录里最新的媒体文件。"""
    candidates = [
        p for p in out_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".m4a", ".mp3", ".wav", ".webm", ".mp4", ".mkv", ".opus"}
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def build_output_stem(title: str) -> str:
    """用视频标题生成一个安全的文件名主干。"""
    return sanitize_path_component(sanitize_filename(title or "untitled", max_length=60))
