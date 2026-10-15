"""离线冒烟测试（不联网、不调任何云端 API）。

跑法（在 GPT-SoVITS 项目根）::

    runtime/python.exe tools/video2article/tests/smoke_offline.py

## 为什么要有这个文件

新加的模块里有大量「看起来对、其实没被调用过」的路径。界面能渲染不等于
链路能跑通。这个脚本逐条验证**可离线验证**的那部分契约；真正花钱的云端
转写和 LLM 成文不在这里测（它们要真金白银），改用 `mock=True` 在界面上验。

⚠ 本文件里任何一条断言都不许「顺手删掉」——它们锁的是下面几类曾经踩过的坑：
配置桥接读错路径、提示词占位符缺失、ffmpeg 找不到、成文模板名对不上。
"""

from __future__ import annotations

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
    from tools.video2article import config_bridge, pipeline
    from tools.video2article.llm_providers import effective_vendor, resolve_protocol
    from tools.video2article.media import extract_audio, find_ffmpeg, probe_duration
    from tools.video2article.paths import PROMPTS_DIR
    from tools.video2article.prompts_store import list_article_prompts, load_prompt, render_prompt

    section("提示词资产")
    templates = list_article_prompts()
    check("发现成稿模板", len(templates) > 0, f"实际 {templates}")
    check("含通用模板 general_article", "general_article" in templates, str(templates))
    check("含美食模板 snack_recipe", "snack_recipe" in templates, str(templates))
    check("README 不被当成模板", "README" not in templates, str(templates))
    check("prompts 目录存在", PROMPTS_DIR.is_dir(), str(PROMPTS_DIR))

    for name in templates:
        body = load_prompt(name) or ""
        check(
            f"模板「{name}」含占位符",
            "{transcript_text}" in body,
            "缺占位符会导致成文只有提示词没有内容",
        )

    rendered = render_prompt("前缀 {transcript_text} 后缀", "正文ABC")
    check("占位符替换正确", rendered == "前缀 正文ABC 后缀", rendered)

    # 反向验证：占位符缺失必须被补上，而不是静默产出空文章
    body = load_prompt("不存在的模板名-xyz")
    check("找不到模板时回落兜底", body is not None and "{transcript_text}" in body, str(body))

    section("协议/厂商解析")
    check("protocol 显式优先", resolve_protocol({"protocol": "anthropic"}) == "anthropic")
    check("旧 provider 回落", resolve_protocol({"provider": "anthropic"}) == "anthropic")
    check("未知 provider 兜底 openai", resolve_protocol({"provider": "乱七八糟"}) == "openai_chat")
    check("空配置兜底 openai", resolve_protocol({}) == "openai_chat")
    check(
        "厂商从 base_url 反推",
        effective_vendor({"base_url": "https://api.deepseek.com"}) == "deepseek",
        str(effective_vendor({"base_url": "https://api.deepseek.com"})),
    )

    section("配置桥接（只读外链）")
    cfg = config_bridge.load_config()
    check("能读到某个 config.json 或返回空 dict", isinstance(cfg, dict), type(cfg).__name__)
    paths = config_bridge.candidate_paths()
    for p in paths:
        check(f"候选路径存在: {p.name}", p.is_file(), str(p))

    # 绝不能把密钥写进自己的 .gitignore 覆盖范围内、或日志里
    desc = config_bridge.describe()
    check("describe 不抛异常", isinstance(desc, dict))
    check(
        "describe 不泄漏密钥",
        "api_key" not in str(desc) and "secret_key" not in str(desc),
        str(desc),
    )
    check(
        "仓库内不跟踪含密钥的 config.json",
        any(
            "tools/video2article/config.json" == line.strip()
            for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        ),
        ".gitignore 缺少 tools/video2article/config.json",
    )

    section("媒体处理")
    ffmpeg = find_ffmpeg()
    check("找到 ffmpeg", bool(ffmpeg), "GPT-SoVITS 便携包自带 runtime/ffmpeg.exe")

    if ffmpeg:
        with tempfile.TemporaryDirectory() as td:
            tone = Path(td) / "tone.wav"
            rc = __import__("subprocess").run(
                [ffmpeg, "-y", "-loglevel", "error", "-f", "lavfi",
                 "-i", "sine=frequency=440:duration=3", str(tone)],
                capture_output=True,
            )
            check(
                "能生成测试音频",
                rc.returncode == 0 and tone.is_file(),
                f"ffmpeg rc={rc.returncode} err={rc.stderr[:200]!r}",
            )
            if tone.is_file():
                dur = probe_duration(tone)
                check("探测时长正确(约3秒)", 2.5 < dur < 3.5, f"得到 {dur}")
                mp3 = extract_audio(tone, Path(td) / "out")
                check("抽取音频产出 mp3", mp3.is_file() and mp3.suffix == ".mp3", str(mp3))
                check("mp3 体积合理", mp3.is_file() and 1000 < mp3.stat().st_size < 5_000_000,
                      str(mp3.stat().st_size) if mp3.is_file() else "无文件")

    section("引擎入口存在性")
    from tools.video2article import asr_custom_post, asr_xf

    check("讯飞引擎主入口", hasattr(asr_xf, "transcribe_audio_with_xf_asr"))
    check("讯飞签名函数", hasattr(asr_xf, "_sign_request"))
    check("custom_post 主入口", hasattr(asr_custom_post, "transcribe_audio_with_custom_post"))
    check("custom_post 保留旧名兼容", hasattr(asr_custom_post, "transcribe_audio_with_minimax_asr"))

    section("讯飞签名（协议契约）")
    import base64
    import hashlib
    import hmac

    app_id, secret, ts = "abc123", "sekret", 1700000000
    md5_hex = hashlib.md5(f"{app_id}{ts}".encode()).hexdigest().encode()
    expect = base64.b64encode(
        hmac.new(secret.encode(), md5_hex, hashlib.sha1).digest()
    ).decode()
    got = asr_xf._sign_request(app_id, secret, ts)
    check("签名 = base64(HMAC-SHA1(secret, MD5(app_id+ts)))", got == expect, f"{got} != {expect}")

    section("流水线自检")
    info = pipeline.selfcheck()
    check("selfcheck 返回 dict", isinstance(info, dict))
    check("selfcheck 报告 ffmpeg", "ffmpeg" in info, str(list(info)))
    check("selfcheck 列出模板", isinstance(info.get("templates"), list), str(info.get("templates")))

    section("反向验证：闸门真的有牙齿吗")
    # 故意给一个不支持的引擎，pipeline 必须**拒绝**而不是默默跑本地 funasr
    res = pipeline.run("不存在的文件.mp4", asr_engine="funasr")
    check("拒绝不支持的引擎", not res.ok and "云端引擎" in res.error, res.error)
    check("失败结果带 error 字段", bool(res.error))

    res2 = pipeline.run("")
    check("空输入被拒绝", not res2.ok, "空输入居然没被拦")

    section("临时目录必须落在项目内（不许吃 C 盘）")
    from tools.video2article.paths import APP_ROOT, MODULE_TEMP_DIR, use_local_temp

    check("模块临时目录在项目内（同一分区）",
          str(MODULE_TEMP_DIR).lower().startswith(str(APP_ROOT).lower()),
          f"{MODULE_TEMP_DIR} 不在 {APP_ROOT} 下")

    original = tempfile.tempdir
    with use_local_temp() as tmp_dir:
        made = tempfile.mkdtemp(prefix="v2a_smoke_")
        check("上下文中 mkdtemp 落到项目 TEMP",
              Path(made).is_relative_to(Path(tmp_dir)) if hasattr(Path, "is_relative_to")
              else Path(made).parent == Path(tmp_dir),
              f"实际落在 {made}")
        Path(made, "probe.bin").write_bytes(b"x" * 16)
    check("退出后 tempfile.tempdir 已还原", tempfile.tempdir == original,
          f"{tempfile.tempdir!r} != {original!r}")
    # 清理本测试自己造的东西（只删自己建的 v2a_smoke_ 前缀目录）
    import shutil as _shutil
    for leftover in Path(MODULE_TEMP_DIR).glob("v2a_smoke_*"):
        _shutil.rmtree(leftover, ignore_errors=True)

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
