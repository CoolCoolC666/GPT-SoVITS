# -*- coding:utf-8 -*-
"""自定义 POST 云端 ASR —— 「0c-语音识别工具」的引擎之一。

支持两类协议（配置里的 ``api_style`` 决定）：

- ``openai_compat``：multipart 上传音频本体（MiniMax STT、多数自建网关）
- ``dashscope_async``：提交任务→轮询→下载结果（阿里百炼 Qwen3-ASR 等）

命令行契约与本地引擎一致：

    python tools/asr/custom_post_asr.py -i <输入目录> -o <输出目录> -s <尺寸> -l <语言> -p <精度>

引擎实现复用 ``tools/video2article/asr_custom_post.py``（移植自一览成文）。
"""

import argparse
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tools.asr import cloud_common  # noqa: E402
from tools.asr import cloud_config  # noqa: E402


def transcribe_one(path: Path, engine_cfg: dict, language: str) -> str:
    from tools.video2article.asr_custom_post import transcribe_audio_with_custom_post

    cfg = dict(engine_cfg)
    if language:
        cfg["language"] = language
    return transcribe_audio_with_custom_post(str(path), cfg)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", type=str, required=True, help="存放音频文件的目录")
    parser.add_argument("-o", "--output_folder", type=str, required=True, help="输出目录")
    parser.add_argument("-s", "--model_size", type=str, default="cloud", help="云端引擎无尺寸概念，忽略")
    parser.add_argument("-l", "--language", type=str, default="", help="语言（zh/en/ja/ko/auto）")
    parser.add_argument("-p", "--precision", type=str, default="float32", help="云端引擎无精度概念，忽略")
    cmd = parser.parse_args()

    engine_cfg = cloud_config.get_engine_config("custom_post")
    if not engine_cfg:
        print("未找到自定义 POST 配置，请在「0c-语音识别工具 → ⚙ 设置」里配置或从一览成文导入。")
        return 2

    state = cloud_config.readiness()
    if not state["custom_post"]["ready"]:
        print(f"自定义 POST 未就绪：{state['custom_post']['detail']}")
        return 2

    print(f"自定义 POST | {state['custom_post']['detail']}")
    print(f"api_style={engine_cfg.get('api_style')}  model={engine_cfg.get('model') or '(由服务端决定)'}")

    def _one(path: Path) -> str:
        return transcribe_one(path, engine_cfg, cmd.language)

    try:
        out = cloud_common.transcribe_folder(
            input_folder=cmd.input_folder,
            output_folder=cmd.output_folder,
            transcribe_one=_one,
            language=cmd.language or engine_cfg.get("language") or "auto",
            concurrency=int(state.get("concurrency") or 4),
        )
    except Exception:
        print(traceback.format_exc())
        return 1

    print(f"完成，标注文件：{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
