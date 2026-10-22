# -*- coding:utf-8 -*-
"""讯飞听见云端 ASR —— 「0c-语音识别工具」的引擎之一。

命令行契约与本地引擎完全一致（见 tools/asr/funasr_asr.py）：

    python tools/asr/xfyun_asr.py -i <输入目录> -o <输出目录> -s <尺寸> -l <语言> -p <精度>

产出与本地引擎同格式的标注文件：``{输出目录}/{输入目录名}.list``，
每行 ``文件路径|说话人|语言|文本``。

引擎实现复用 ``tools/video2article/asr_xf.py``（移植自一览成文，协议逻辑未改），
不在这里重复实现一遍。
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
    from tools.video2article.asr_xf import transcribe_audio_with_xf_asr

    cfg = dict(engine_cfg)
    if language:
        # 命令行指定的语言优先于配置文件里的设置
        cfg["language"] = language
    return transcribe_audio_with_xf_asr(str(path), cfg)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", type=str, required=True, help="存放音频文件的目录")
    parser.add_argument("-o", "--output_folder", type=str, required=True, help="输出目录")
    parser.add_argument("-s", "--model_size", type=str, default="cloud", help="云端引擎无尺寸概念，忽略")
    parser.add_argument("-l", "--language", type=str, default="", help="语言（cn/en/ja/auto）")
    parser.add_argument("-p", "--precision", type=str, default="float32", help="云端引擎无精度概念，忽略")
    cmd = parser.parse_args()

    engine_cfg = cloud_config.get_engine_config("xf_asr")
    if not engine_cfg:
        print("未找到讯飞听见配置，请在「0c-语音识别工具 → ⚙ 设置」里配置或从一览成文导入。")
        return 2

    state = cloud_config.readiness()
    if not state["xf_asr"]["ready"]:
        print(f"讯飞听见未就绪：{state['xf_asr']['detail']}")
        return 2

    print(f"讯飞听见 | {state['xf_asr']['detail']}")
    print(f"请求头/凭证：app_id={cloud_config.mask(engine_cfg.get('app_id'))} "
          f"secret_key={cloud_config.mask(engine_cfg.get('secret_key'))}")

    def _one(path: Path) -> str:
        return transcribe_one(path, engine_cfg, cmd.language)

    try:
        out = cloud_common.transcribe_folder(
            input_folder=cmd.input_folder,
            output_folder=cmd.output_folder,
            transcribe_one=_one,
            language=cmd.language or engine_cfg.get("language") or "cn",
            concurrency=int(state.get("concurrency") or 4),
        )
    except Exception:
        print(traceback.format_exc())
        return 1

    print(f"完成，标注文件：{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
