"""流水线编排：链接/音视频 → 音频 → 转写 → 成文 → Markdown。

刻意做成**同步函数 + 进度回调**，而不是内部再起线程：
WebUI 那边有自己的 worker 线程模型（webui.py 里各 tab 都有 output_info 约定），
这里再套一层线程只会让「日志从哪来」变复杂。需要并发时在调用方做。
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from . import config_bridge
from .download import DownloadError, download_media, is_url
from .llm_client import LLMError, generate_article
from .logutil import get_logger
from .media import MediaError, extract_audio, format_size, probe_duration
from .paths import LOCAL_MEDIA_EXTENSIONS, OUTPUT_DIR, ensure_runtime_dirs, use_local_temp
from .prompts_store import list_article_prompts
from .text_utils import format_time, sanitize_filename

logger = get_logger("pipeline")

ProgressFn = Callable[[str], None]


class PipelineError(RuntimeError):
    """流水线失败。消息面向用户，必须指出是哪一环、怎么修。"""


@dataclass
class Result:
    """一次成稿的产物。"""

    ok: bool = False
    source: str = ""
    audio_path: Optional[Path] = None
    transcript_path: Optional[Path] = None
    article_path: Optional[Path] = None
    transcript_chars: int = 0
    article_chars: int = 0
    asr_engine: str = ""
    prompt_name: str = ""
    elapsed: float = 0.0
    error: str = ""
    detail: list = field(default_factory=list)


def _noop(_msg: str) -> None:
    pass


def list_templates() -> list:
    """界面上给「成稿模板」下拉用。"""
    return list_article_prompts()


def selfcheck() -> dict:
    """离线自检：不发网络请求，只看本机具备哪些条件。

    界面上的「检查配置」按钮直接调它。**默认不发任何请求**——
    探测远端配额这类动作很贵，不能当健康检查用。
    """
    ensure_runtime_dirs()
    from .media import find_ffmpeg, find_ffprobe

    cfg_info = config_bridge.describe()
    return {
        "ffmpeg": find_ffmpeg() or "(未找到)",
        "ffprobe": find_ffprobe() or "(未找到)",
        "templates": list_templates(),
        "llm_ready": cfg_info["llm_ready"],
        "llm_detail": cfg_info["llm_detail"],
        "asr_ready": cfg_info["asr_ready"],
        "asr_engine": cfg_info["asr_engine"],
        "asr_detail": cfg_info["asr_detail"],
        "config_path": cfg_info["config_path"],
    }


def run(
    source: str,
    prompt_name: str = "general_article",
    asr_engine: str = "",
    keep_audio: bool = False,
    download_audio_only: bool = True,
    max_duration: int = 0,
    progress: ProgressFn = _noop,
) -> Result:
    """跑完整条链路。

    Args:
        source: 本地音视频路径，或视频链接
        prompt_name: 成稿模板名（见 ``list_templates()``）
        asr_engine: 覆盖配置里的引擎（"xf_asr" / "custom_post"），留空用配置值
        keep_audio: 保留抽出的音频（默认跑完即删）
        download_audio_only: 下载链接时只要音频
        max_duration: 下载时长上限（秒），0 = 不限
        progress: 进度回调，每步打一行
    """
    started = time.time()
    res = Result(source=source, prompt_name=prompt_name)
    out_dir = OUTPUT_DIR

    def stage(msg: str) -> None:
        logger.info(msg)
        progress(msg)

    try:
        ensure_runtime_dirs()

        # ---------- 前置检查：配置不齐就别开跑 ----------
        transcribe_cfg = config_bridge.get_section("transcribe")
        engine = (asr_engine or transcribe_cfg.get("asr_engine") or "").strip()
        res.asr_engine = engine
        if engine not in ("xf_asr", "custom_post", "minimax_asr"):
            raise PipelineError(
                f"不支持的 ASR 引擎: {engine or '(空)'}。\n"
                f"本模块只接云端引擎：xf_asr（讯飞听见）或 custom_post。"
                f"请改 config.json 的 transcribe.asr_engine。"
            )
        # 旧名归一：minimax_asr → custom_post（保持与一览成文一致）
        if engine == "minimax_asr":
            engine = "custom_post"
            res.asr_engine = engine

        llm_cfg = config_bridge.get_section("llm")
        llm_ok, llm_why = config_bridge.llm_ready()
        if not llm_ok:
            raise PipelineError(f"大模型未就绪：{llm_why}")

        asr_ok, _name, asr_why = config_bridge.asr_ready()
        if not asr_ok:
            raise PipelineError(f"语音识别未就绪：{asr_why}")

        # ---------- 1. 取到本地媒体 ----------
        media_path: Path
        if is_url(source):
            stage(f"[1/5] 下载链接 → 音频…")
            youtube_cfg = config_bridge.get_section("youtube")
            media_path = download_media(
                source,
                audio_only=download_audio_only,
                cookies_from_browser=str(youtube_cfg.get("cookies_from_browser") or ""),
                cookies_file=str(youtube_cfg.get("cookies_file") or ""),
                max_duration=max_duration,
            )
        else:
            media_path = Path(source.strip())
            if not media_path.is_file():
                raise PipelineError(f"文件不存在: {media_path}")
            if media_path.suffix.lower() not in LOCAL_MEDIA_EXTENSIONS:
                stage(
                    f"提示：{media_path.suffix} 不在常见音视频扩展名里，仍按音频处理，"
                    f"若报错请确认文件类型"
                )
            stage(f"[1/5] 使用本地文件: {media_path.name}")

        # ---------- 2. 抽音 ----------
        stage("[2/5] 抽取音轨…")
        audio_path = extract_audio(media_path, out_dir / "audio")
        res.audio_path = audio_path
        duration = probe_duration(audio_path)
        if duration > 0:
            stage(f"      音频时长 {format_time(duration)}，体积 {format_size(audio_path.stat().st_size)}")
        else:
            stage(f"      音频体积 {format_size(audio_path.stat().st_size)}（探测不到时长）")

        # ---------- 3. 转写 ----------
        stage(f"[3/5] 云端转写（{res.asr_engine}）…")
        transcript = _transcribe(audio_path, engine, stage)
        if not transcript.strip():
            raise PipelineError(
                "转写结果为空。常见原因：音频里没有人声、Mock 模式开着、"
                "或云端引擎返回了空内容。"
            )
        res.transcript_chars = len(transcript)

        stem = sanitize_filename(media_path.stem, max_length=60)
        job_dir = out_dir / stem
        job_dir.mkdir(parents=True, exist_ok=True)
        transcript_path = job_dir / "transcript.md"
        transcript_path.write_text(
            f"<!-- 来源: {media_path.name} | 引擎: {res.asr_engine} -->\n\n{transcript.strip()}\n",
            encoding="utf-8",
        )
        res.transcript_path = transcript_path
        stage(f"      转写完成，{res.transcript_chars} 字符 → {transcript_path.name}")

        # ---------- 4. 成文 ----------
        stage(f"[4/5] 大模型成文（模板：{prompt_name}）…")
        article = generate_article(transcript, llm_cfg, prompt_name)
        res.article_chars = len(article)

        article_path = job_dir / "article.md"
        article_path.write_text(article.strip() + "\n", encoding="utf-8")
        res.article_path = article_path
        stage(f"      成稿完成，{res.article_chars} 字符 → {article_path.name}")

        # ---------- 5. 收尾 ----------
        if not keep_audio and is_url(source):
            # 只清下载目录里那份中间产物，本地文件一律不动
            try:
                media_path.unlink(missing_ok=True)
            except OSError as e:
                stage(f"      （中间音频删除失败，不影响结果: {e}）")

        res.ok = True
        res.elapsed = time.time() - started
        stage(f"[5/5] 完成，耗时 {format_time(res.elapsed)}")
        return res

    except (PipelineError, MediaError, DownloadError, LLMError) as e:
        res.error = str(e)
        res.elapsed = time.time() - started
        logger.error("流水线失败: %s", e)
        progress(f"失败：{e}")
        return res
    except Exception as e:  # noqa: BLE001 - 兜底，别让 WebUI 崩掉
        res.error = f"未预期的错误: {e}"
        res.elapsed = time.time() - started
        logger.error("流水线异常:\n%s", traceback.format_exc())
        progress(res.error)
        return res


def _transcribe(audio_path: Path, engine: str, progress: ProgressFn) -> str:
    """按引擎名分发到具体实现。

    整个调用被 `use_local_temp()` 包住：引擎内部的音频切片会落盘到项目内
    TEMP（E 盘），而不是系统 TEMP（C 盘）。详见 paths.use_local_temp 的说明。
    """
    with use_local_temp():
        if engine == "xf_asr":
            from .asr_xf import transcribe_audio_with_xf_asr

            return transcribe_audio_with_xf_asr(
                str(audio_path), config_bridge.get_section("transcribe", "xf_asr")
            )

        if engine == "custom_post":
            from .asr_custom_post import transcribe_audio_with_custom_post

            cfg = config_bridge.get_section("transcribe", "custom_post") or config_bridge.get_section(
                "transcribe", "minimax_asr"
            )
            return transcribe_audio_with_custom_post(str(audio_path), cfg)

    raise PipelineError(f"未实现的引擎: {engine}")


def transcribe_only(audio_or_video: str, asr_engine: str = "", progress: ProgressFn = _noop) -> Result:
    """只转写不成文（已有稿子、只想补转写时用）。"""
    started = time.time()
    res = Result(source=audio_or_video)
    try:
        transcribe_cfg = config_bridge.get_section("transcribe")
        engine = (asr_engine or transcribe_cfg.get("asr_engine") or "").strip()
        if engine == "minimax_asr":
            engine = "custom_post"
        res.asr_engine = engine

        src = Path(audio_or_video.strip())
        if not src.is_file():
            raise PipelineError(f"文件不存在: {src}")

        progress("[1/2] 抽取音轨…")
        audio_path = src if src.suffix.lower() in (".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg") \
            else extract_audio(src, OUTPUT_DIR / "audio")
        res.audio_path = audio_path

        progress(f"[2/2] 云端转写（{engine}）…")
        transcript = _transcribe(audio_path, engine, progress)
        res.transcript_chars = len(transcript)

        job_dir = OUTPUT_DIR / sanitize_filename(src.stem, max_length=60)
        job_dir.mkdir(parents=True, exist_ok=True)
        res.transcript_path = job_dir / "transcript.md"
        res.transcript_path.write_text(transcript.strip() + "\n", encoding="utf-8")

        res.ok = True
        res.elapsed = time.time() - started
        progress(f"完成，{res.transcript_chars} 字符，耗时 {format_time(res.elapsed)}")
        return res
    except Exception as e:
        res.error = str(e)
        logger.error("仅转写失败: %s", e)
        return res
