"""提交前密钥扫描：把 config.json 里的**每个**密钥值拿去和暂存区内容比对。

为什么用精确比对而不是正则找 `sk-`：

正则只能告诉你「看起来像密钥」，还会把文档里的占位符（`Bearer sk-xxx`）
一并报出来，久了就没人看了。精确比对是**零误报**的判据——
真密钥出现在暂存区，一定报出来。

用法（在项目根）::

    runtime/python.exe tools/video2article/scripts/scan_secrets.py
    runtime/python.exe tools/video2article/scripts/scan_secrets.py --config <config.json>

⚠ 本脚本**只读**，且只打印密钥的掩码前缀，不打印原值。
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

# 可能含密钥的字段名（中英文都算一层）
SECRET_FIELDS = (
    "api_key", "apikey", "secret_key", "secretkey", "token", "access_key_id",
    "access_key_secret", "secret", "password", "appid", "app_id", "signa",
    "authorization", "sk", "key",
)

# 一眼假的值，不算
PLACEHOLDERS = re.compile(
    r"^(your[-_ ]?|xxx|placeholder|changeme|<.*>|\.\.\.|)$|your-api-key|your-token",
    re.IGNORECASE,
)


def collect_secrets(node, out: list, path: str = "") -> None:
    """递归收集配置里所有看起来像密钥的标量值。"""
    if isinstance(node, dict):
        for k, v in node.items():
            child = f"{path}.{k}" if path else str(k)
            if str(k).lower() in SECRET_FIELDS and isinstance(v, str) and v.strip():
                if not PLACEHOLDERS.match(v.strip()):
                    out.append((child, v.strip()))
            else:
                collect_secrets(v, out, child)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            collect_secrets(v, out, f"{path}[{i}]")


def mask(s: str) -> str:
    return f"{s[:6]}***{s[-2:]}" if len(s) > 10 else "***"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="", help="一览成文的 config.json 路径")
    ap.add_argument("--stdin-diff", action="store_true",
                    help="改为扫描 git diff --cached 的输出")
    args = ap.parse_args()

    # 1) 收集密钥
    cfg_path = Path(args.config) if args.config else None
    if cfg_path is None:
        from tools.video2article import config_bridge as cb

        paths = cb.candidate_paths()
        cfg_path = paths[0] if paths else None
    if cfg_path is None or not cfg_path.is_file():
        print("!! 找不到 config.json，无从比对（这不是失败，只是没法验）")
        return 0

    secrets: list = []
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"!! 读取 {cfg_path} 失败: {e}")
        return 2
    collect_secrets(data, secrets)
    print(f"从 {cfg_path} 收集到 {len(secrets)} 个疑似密钥值")

    # 2) 扫描暂存区
    proc = subprocess.run(
        ["git", "diff", "--cached"], cwd=ROOT, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    blob = proc.stdout or ""
    print(f"暂存区 diff 大小: {len(blob)} 字符\n")

    leaked = []
    for field, value in secrets:
        if value in blob:
            leaked.append((field, value))

    if leaked:
        print(f"!! 发现 {len(leaked)} 个密钥出现在暂存区，**不要提交**：")
        for field, value in leaked:
            print(f"   {field} = {mask(value)}")
        return 1

    print("✅ 暂存区不含任何来自 config.json 的密钥")
    for field, value in secrets:
        print(f"   已核对（未出现）: {field} = {mask(value)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
