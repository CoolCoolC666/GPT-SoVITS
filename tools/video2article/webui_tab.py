"""WebUI 的「视频转文章」tab。

## 为什么要单独一个文件

`webui.py` 有 1983 行，且是上游同步维护的文件。这里的一切 UI 逻辑都收在
本文件，`webui.py` 里只留一次 `mount(i18n)` 调用（一个 import + 一个函数调用）。
好处是：以后要跟进上游更新，冲突面只有一个函数调用点。

## 为什么不自己再套线程

`pipeline.run()` 是阻塞的同步函数，直接挂在 Gradio 的 `.click()` 上即可——
`app.queue()` 已经把回调放进 worker 线程。再套一层 `threading.Thread`
只会让「日志从哪个线程来」变复杂，而且丢异常。
"""

from __future__ import annotations

import traceback

import gradio as gr

from . import config_bridge, pipeline
from .logutil import get_logger
from .paths import OUTPUT_DIR

logger = get_logger("webui")

TAB_LABEL = "3-视频转文章（一览成文）"


def _fmt_selfcheck(info: dict) -> str:
    ok = lambda b: "✅" if b else "⚠️"  # noqa: E731
    return (
        f"**ffmpeg**：`{info['ffmpeg']}`\n\n"
        f"**ffprobe**：`{info['ffprobe']}`\n\n"
        f"**配置文件**：`{info['config_path']}`\n\n"
        f"{ok(info['asr_ready'])} **语音识别**：{info['asr_engine']} — {info['asr_detail']}\n\n"
        f"{ok(info['llm_ready'])} **大模型**：{info['llm_detail']}\n\n"
        f"**可用成稿模板**：{', '.join(info['templates']) or '(无)'}"
    )


def mount(i18n=None) -> None:
    """在调用方的 `with gr.Tabs():` 上下文里建出这个 tab。"""
    label = i18n(TAB_LABEL) if i18n else TAB_LABEL

    with gr.TabItem(label):
        gr.Markdown(
            "把**视频/音频/链接**转成文字，并用大模型整理成结构化文章。\n\n"
            "链路：下载（yt-dlp）→ 抽音（ffmpeg）→ 云端转写（讯飞听见 / 自定义 POST）"
            "→ 大模型成文 → Markdown。\n\n"
            "能力移植自 [一览成文 YilanChengWen](https://github.com/CoolCoolC666/video-to-article)"
            "（MIT），成稿模板沿用其 `prompts/` 资产。"
        )

        with gr.Accordion(label="配置状态", open=True):
            status_md = gr.Markdown(value=_fmt_selfcheck(pipeline.selfcheck()))
            with gr.Row():
                refresh_btn = gr.Button(value="🔄 重新检查配置", variant="secondary", size="sm")
                open_out_btn = gr.Button(value="📂 打开输出目录", variant="secondary", size="sm")

        with gr.Accordion(label="成文设置", open=True):
            with gr.Row():
                template_dropdown = gr.Dropdown(
                    choices=pipeline.list_templates() or ["general_article"],
                    value=(pipeline.list_templates() or ["general_article"])[0],
                    label="成稿模板（决定文章结构与语气）",
                    scale=3,
                )
                engine_dropdown = gr.Dropdown(
                    choices=[
                        "跟随配置",
                        "xf_asr（讯飞听见）",
                        "custom_post（自定义 POST）",
                    ],
                    value="跟随配置",
                    label="转写引擎",
                    scale=2,
                )
            with gr.Row():
                keep_audio = gr.Checkbox(value=False, label="保留中间音频文件")
                download_audio_only = gr.Checkbox(
                    value=True, label="下载链接时只取音频（更省流量）"
                )
                max_duration = gr.Number(
                    value=0, label="下载时长上限（分钟，0 = 不限）", precision=0
                )

        with gr.Accordion(label="输入", open=True):
            source_input = gr.Textbox(
                label="本地音视频路径，或视频链接（B站 / YouTube / 抖音 等）",
                placeholder=r"例：D:\ Videos\lecture.mp4   或   https://www.bilibili.com/video/...",
                lines=2,
            )
            with gr.Row():
                run_btn = gr.Button(value="📝 转写并成文", variant="primary", scale=2)
                transcribe_btn = gr.Button(value="🎙 仅转写（不调用大模型）", variant="secondary", scale=1)

        with gr.Accordion(label="结果", open=True):
            log_box = gr.Textbox(label="运行日志", lines=14, max_lines=30, show_copy_button=True)
            article_box = gr.Markdown(label="文章预览")

        # ---------- 事件绑定 ----------
        def _refresh():
            return _fmt_selfcheck(pipeline.selfcheck())

        def _engine_value():
            picked = engine_dropdown.value or "跟随配置"
            return "" if picked == "跟随配置" else picked.split("（")[0].strip()

        def _run(only_transcribe: bool):
            source = (source_input.value or "").strip()
            if not source:
                msg = "请先填写本地文件路径或视频链接"
                return msg, ""

            engine = _engine_value()
            limit_min = float(max_duration.value or 0)

            logs: list = []

            def progress(line: str) -> None:
                logs.append(line)

            try:
                if only_transcribe:
                    res = pipeline.transcribe_only(source, asr_engine=engine, progress=progress)
                else:
                    res = pipeline.run(
                        source,
                        prompt_name=template_dropdown.value,
                        asr_engine=engine,
                        keep_audio=bool(keep_audio.value),
                        download_audio_only=bool(download_audio_only.value),
                        max_duration=int(limit_min * 60),
                        progress=progress,
                    )
            except Exception:  # noqa: BLE001
                detail = traceback.format_exc()
                logger.error(detail)
                return f"未预期的异常：\n{detail}", ""

            if not res.ok:
                return "\n".join(logs + [f"❌ 失败：{res.error}"]), ""

            footer = (
                f"\n\n---\n\n✅ 完成，耗时 {res.elapsed:.0f}s　"
                f"转写 {res.transcript_chars} 字符　成稿 {res.article_chars} 字符\n\n"
                f"文件位置：`{res.article_path}`"
            )
            article = ""
            if res.article_path and not only_transcribe:
                try:
                    article = res.article_path.read_text(encoding="utf-8")
                except OSError as e:
                    article = f"（成稿已保存，但读取失败：{e}）"
            elif res.transcript_path:
                try:
                    article = res.transcript_path.read_text(encoding="utf-8")
                except OSError:
                    article = ""

            return "\n".join(logs) + footer, article

        refresh_btn.click(_refresh, outputs=status_md)
        run_btn.click(lambda: _run(False), outputs=[log_box, article_box])
        transcribe_btn.click(lambda: _run(True), outputs=[log_box, article_box])
        open_out_btn.click(
            lambda: f"产物输出目录：\n`{OUTPUT_DIR}`\n\n（目录已存在时会按「文件名/」建子目录存放 transcript.md 与 article.md）",
            outputs=log_box,
        )
