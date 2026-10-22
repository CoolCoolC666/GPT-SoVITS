"""云端 ASR 接入的离线契约测试（不联网、不调任何云端 API）。

跑法（在 GPT-SoVITS 项目根）::

    runtime/python.exe tools/asr/tests/smoke_cloud_offline.py

## 为什么要有这个文件

云端引擎接进「0c-语音识别工具」后，**下游能不能吃**是比「能不能转写」更要命的问题。
链路上有三个静默失败的点：

1. `.list` 格式不对 → `subfix_webui.py`（0d 标注）打印 "error line" 后**跳过**
2. 语言 token 不在白名单 → `1-get-text.py` 打印警告后**跳过**
3. 文本里残留 `[00:10]` / `【说话人1】` → 模型把标签当内容学走

这三处都不会报错退出，只会让数据悄悄少掉或变脏。所以本文件把它们锁成断言。

本测试用**桩函数**替代真实云端调用，验证的是格式契约，不是网络。
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PASS = 0
FAIL = 0
FAILURES: list = []


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        FAILURES.append(f"{name}: {detail}")
        print(f"  FAIL  {name}  {detail}")


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> int:
    from tools.asr import cloud_common as cc
    from tools.asr import cloud_config

    section("文本清洗（云端标记不能漏进训练语料）")
    raw = "[00:10] 日常生活呢其实这是我们非常\n[00:13] 【说话人1】普及的，我们可以说"
    cleaned = cc.clean_text(raw)
    check("时间戳被剥掉", "[00:10]" not in cleaned and "[00:13]" not in cleaned, cleaned)
    check("说话人标签被剥掉", "【说话人1】" not in cleaned and "说话人" not in cleaned, cleaned)
    check("正文保留", "普及的" in cleaned, cleaned)
    check("中文不加空格", " " not in cleaned, cleaned)

    en = cc.clean_text("[00:01] hello world\n[00:02] again")
    check("英文加空格", "hello world again" == en, repr(en))

    check("空行不留空串", cc.clean_text("[00:05]\n[00:06]  ") == "", repr(cc.clean_text("[00:05]\n[00:06]  ")))

    section("语言 token 映射（映射错 = 整份 .list 被下游丢弃）")
    # 1-get-text.py 的白名单
    allowed = {"ZH", "EN", "JA", "KO", "YUE"}
    for src, expect in [("cn", "ZH"), ("zh", "ZH"), ("en", "EN"), ("ja", "JA"),
                        ("ko", "KO"), ("yue", "YUE"), ("chinese", "ZH"), ("english", "EN")]:
        got = cc.map_lang(src)
        check(f"{src} → {expect}", got == expect, f"得到 {got}")
    check("auto + 中文 → ZH", cc.map_lang("auto", "你好世界") == "ZH")
    check("auto + 英文 → EN", cc.map_lang("auto", "hello world") == "EN")
    for src in ("cn", "zh", "en", "ja", "ko", "yue", "auto", "unknown"):
        check(f"{src} 落在白名单内", cc.map_lang(src, "测试文本") in allowed, cc.map_lang(src, "测试文本"))

    section("端到端产出 .list（用桩函数，不联网）")
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        inp = base / "raw_poc"
        inp.mkdir()
        ffmpeg = ROOT / "runtime" / "ffmpeg.exe"
        for i in range(3):
            wav = inp / f"lesson_{i}.wav"
            if ffmpeg.is_file():
                subprocess.run(
                    [str(ffmpeg), "-y", "-loglevel", "error", "-f", "lavfi",
                     "-i", f"sine=frequency=440:duration=1", str(wav)],
                    capture_output=True,
                )
            else:  # 没有 ffmpeg 就造个空占位文件，契约测试不依赖音频内容
                wav.write_bytes(b"RIFF" + b"0" * 64)

        def stub(_path: Path) -> str:
            return "[00:10] 【说话人1】你好世界\n[00:13] 这是第二段"

        out_dir = base / "asr_opt"
        list_path = cc.transcribe_folder(
            input_folder=str(inp),
            output_folder=str(out_dir),
            transcribe_one=stub,
            language="cn",
            concurrency=2,
        )
        check("标注文件已生成", Path(list_path).is_file(), list_path)
        check("文件名 = 输入目录名 + .list", Path(list_path).name == "raw_poc.list", Path(list_path).name)

        lines = [ln for ln in Path(list_path).read_text(encoding="utf-8").splitlines() if ln.strip()]
        check("行数 = 文件数", len(lines) == 3, f"得到 {len(lines)}")

        section("下游消费契约（0d 标注 + 1A 训练集格式化）")
        for idx, line in enumerate(lines):
            parts = line.split("|")
            check(f"第{idx+1}行能被 split('|') 切成 4 段", len(parts) == 4, f"{len(parts)} 段: {line[:60]}")
            if len(parts) == 4:
                wav_path, spk, lang, text = parts
                check(f"第{idx+1}行 路径存在", Path(wav_path).is_file(), wav_path)
                check(f"第{idx+1}行 说话人列非空", bool(spk.strip()), spk)
                check(f"第{idx+1}行 语言列在白名单", lang in allowed, lang)
                check(f"第{idx+1}行 文本无时间戳", "[" not in text and "]" not in text, text)
                check(f"第{idx+1}行 文本无说话人标签", "说话人" not in text, text)
                check(f"第{idx+1}行 文本非空", bool(text.strip()), text)

        # 复刻 subfix_webui.py 的 b_load_list 解析逻辑
        parsed = []
        for line in lines:
            data = line.split("|")
            if len(data) == 4:
                wav_path, spk, lang, text = data
                parsed.append({"wav_path": wav_path, "speaker_name": spk, "language": lang, "text": text.strip()})
        check("0d 标注工具能解析全部行", len(parsed) == 3, f"{len(parsed)}/3")

    section("单个文件失败不应拖垮整批")
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        inp = base / "raw_fail"
        inp.mkdir()
        for i in range(3):
            (inp / f"x_{i}.wav").write_bytes(b"RIFF" + b"0" * 64)

        calls = {"n": 0}

        def flaky(_path: Path) -> str:
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("模拟云端 500")
            return "成功的一段文本"

        out = cc.transcribe_folder(str(inp), str(base / "out"), flaky, language="cn", concurrency=1)
        out_lines = [ln for ln in Path(out).read_text(encoding="utf-8").splitlines() if ln.strip()]
        check("失败文件被跳过而非中断整批", len(out_lines) == 2, f"得到 {len(out_lines)}")

    section("配置层")
    check("本地配置文件路径正确", cloud_config.CLOUD_CONFIG_FILE.name == "cloud_config.json")
    check("云端配置文件已被 gitignore",
          "tools/asr/cloud_config.json" in (ROOT / ".gitignore").read_text(encoding="utf-8"))
    check("请求头文件已被 gitignore",
          "tools/asr/cloud_headers.txt" in (ROOT / ".gitignore").read_text(encoding="utf-8"))
    # 密钥掩码不能还原出原文
    check("掩码不泄漏原文", cloud_config.mask("sk-abcdef123456") != "sk-abcdef123456")

    from tools.asr.config import asr_dict
    cloud_entries = {k: v for k, v in asr_dict.items() if v.get("cloud")}
    check("asr_dict 里有 2 个云端引擎", len(cloud_entries) == 2, str(list(cloud_entries)))
    for name, entry in cloud_entries.items():
        check(f"「{name}」有脚本路径", (ROOT / "tools" / "asr" / entry["path"]).is_file(), entry["path"])
        check(f"「{name}」有语言选项", len(entry["lang"]) > 0)
        check(f"「{name}」有尺寸选项", len(entry["size"]) > 0)

    print("\n" + "=" * 56)
    print(f"通过 {PASS} 项，失败 {FAIL} 项")
    if FAILURES:
        print("\n失败明细：")
        for f in FAILURES:
            print("  -", f)
    print("=" * 56)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
