"""独立预览窗口：只跑「视频转文章」这一个 tab。

**为什么需要它**：`webui.py` 是个近 2000 行、启动要吃 GPU 的重程序，
为了验证一个新 tab 而重启它不划算（也会打断正在用的人）。本脚本用同一个
`mount()` 函数另开一个轻量窗口，验证的仍然是**真正要交付的那份界面代码**。

用法::

    runtime/python.exe tools/video2article/scripts/run_tab_preview.py

启动后浏览器会打开 http://127.0.0.1:9880/
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    import gradio as gr

    from tools.video2article.webui_tab import mount

    with gr.Blocks(title="视频转文章（独立预览）", analytics_enabled=False) as app:
        gr.Markdown("# 视频转文章 · 独立预览\n\n这是从 `webui.py` 里抽出来的同一个 tab，"
                    "用于单独验收，不必重启完整的 GPT-SoVITS WebUI。")
        mount()

    print("打开 http://127.0.0.1:9880/ 查看")
    app.queue().launch(
        server_name="127.0.0.1",
        server_port=9880,
        inbrowser=True,
        share=False,
        show_api=False,
    )


if __name__ == "__main__":
    main()
