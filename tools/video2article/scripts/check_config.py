"""打印配置就绪状态（脱敏）。既是自己的一次性排查工具，也是给用户的日常体检。

用法::

    runtime/python.exe tools/video2article/scripts/check_config.py

**永远不打印密钥明文**：只输出「有没有」和掩码。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.video2article import config_bridge as cb  # noqa: E402
from tools.video2article.llm_providers import mask_secret  # noqa: E402


def main() -> None:
    print("=== 查找顺序命中的配置文件 ===")
    paths = cb.candidate_paths()
    if paths:
        for p in paths:
            print(f"  [命中] {p}")
    else:
        print("  [未命中] 没有任何 config.json")
        print()
        print("怎么办（任选一种）：")
        print(f"  1) 设环境变量：{cb.ENV_CONFIG_PATH}=<一览成文的 config.json 路径>")
        print(f"  2) 或在本目录放一份：{cb.LOCAL_CONFIG_FILE}")
        print("     （该文件已被 .gitignore 忽略，不会进版本库）")

    print()
    print("=== 转写引擎 ===")
    transcribe = cb.get_section("transcribe")
    engine = transcribe.get("asr_engine")
    print(f"  transcribe.asr_engine = {engine or '(空)'}")
    for name in ("xf_asr", "custom_post", "minimax_asr"):
        cfg = cb.get_section("transcribe", name)
        if not cfg:
            continue
        keys = sorted(k for k in cfg if k != "profiles")
        print(f"  [{name}] mock={cfg.get('mock')} 字段={keys}")
        for secret_key in ("app_id", "secret_key", "api_key"):
            val = str(cfg.get(secret_key) or "").strip()
            if val:
                print(f"        {secret_key} = {mask_secret(val)}")

    ok, name, why = cb.asr_ready()
    print(f"  → {'就绪' if ok else '未就绪'}：{name} — {why}")

    print()
    print("=== 大模型 ===")
    llm = cb.get_section("llm")
    if not llm:
        print("  [未配置] config.json 里没有 llm 段")
    else:
        print(f"  protocol = {llm.get('protocol') or llm.get('provider') or '(空)'}")
        print(f"  vendor   = {llm.get('vendor') or '(由 base_url 推断)'}")
        print(f"  model    = {llm.get('model')}")
        print(f"  base_url = {llm.get('base_url')}")
        key = str(llm.get("api_key") or "").strip()
        print(f"  api_key  = {mask_secret(key) if key else '(空)'}")
    ok, why = cb.llm_ready()
    print(f"  → {'就绪' if ok else '未就绪'}：{why}")


if __name__ == "__main__":
    main()
