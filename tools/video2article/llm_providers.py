"""LLM 协议 / 厂商解析（移植自一览成文 providers/llm_providers.py）。

**为什么 protocol 和 vendor 必须拆成两个字段**：

一览成文原来的 `llm.provider` 一个字段同时承担两件事——「用哪个 SDK」
和「连哪家服务」。真实配置长这样：

    "provider": "openai",
    "base_url": "https://api.minimaxi.com/v1"

它的真实含义是「**用 OpenAI 兼容协议** 连 **MiniMax**」，而不是「厂商是 OpenAI」。
一个字段背两个语义，改配置时必然会配错。

本模块保留全部旧字段兼容（见 `resolve_protocol` 的判定顺序），旧 config.json
**零改动**继续能用。

本文件只保留本模块真正用到的函数（UI 预填那套在 GPT-SoVITS 里没有对应界面）。
"""

from __future__ import annotations

from typing import Dict

PROTOCOL_LABELS = {
    "openai_chat": "OpenAI 兼容（绝大多数云 / 中转都属此类）",
    "anthropic": "Anthropic 原生（Claude）",
}

# 旧 provider 值 → 新 protocol 的迁移映射
_LEGACY_PROVIDER_MAP = {
    "anthropic": "anthropic",
}


def resolve_protocol(llm_cfg: dict) -> str:
    """从 llm 配置块解析出协议。旧配置零改动可用。

    判定顺序：
    1. 有 `protocol` 字段且合法 → 用它
    2. 否则回落旧 `provider` 字段
    3. `provider == "anthropic"` → anthropic
    4. 其他一切（含空 / 拼错 / 未来新增值）→ openai_chat

    第 4 条是关键的兜底：绝大多数 OpenAI 兼容服务都该走 openai_chat，
    拼错就整条链路挂掉是最差的表现。
    """
    cfg = llm_cfg or {}
    explicit = str(cfg.get("protocol") or "").strip()
    if explicit in PROTOCOL_LABELS:
        return explicit
    legacy = str(cfg.get("provider") or "").strip().lower()
    return _LEGACY_PROVIDER_MAP.get(legacy, "openai_chat")


VENDOR_REGISTRY: Dict[str, Dict[str, str]] = {
    "openai": {"label": "OpenAI", "base_url": "https://api.openai.com/v1"},
    "minimax": {"label": "MiniMax", "base_url": "https://api.minimaxi.com/v1"},
    "deepseek": {"label": "DeepSeek", "base_url": "https://api.deepseek.com"},
    "qwen": {"label": "阿里通义千问（DashScope）",
             "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"},
    "moonshot": {"label": "Moonshot（Kimi）", "base_url": "https://api.moonshot.cn/v1"},
    "siliconflow": {"label": "硅基流动（SiliconFlow）", "base_url": "https://api.siliconflow.cn/v1"},
    "xai": {"label": "xAI（Grok）", "base_url": "https://api.x.ai/v1"},
    "anthropic": {"label": "Anthropic（Claude）", "base_url": "https://api.anthropic.com"},
    "custom": {"label": "自定义 / 其他（手填 Base URL）", "base_url": ""},
}

_BASE_URL_HINTS = [
    ("api.openai.com", "openai"),
    ("minimax", "minimax"),
    ("deepseek", "deepseek"),
    ("dashscope", "qwen"),
    ("moonshot", "moonshot"),
    ("siliconflow", "siliconflow"),
    ("api.x.ai", "xai"),
    ("api.anthropic.com", "anthropic"),
]


def guess_vendor_from_url(base_url: str) -> str:
    """从 base_url 反推厂商。猜不中返回 "custom"。

    ⚠ 这是**兜底推断**，不是权威判定。
    """
    u = str(base_url or "").lower()
    if not u:
        return "custom"
    for needle, vendor in _BASE_URL_HINTS:
        if needle in u:
            return vendor
    return "custom"


def effective_vendor(llm_cfg: dict) -> str:
    """解析生效厂商：显式 vendor 优先，否则从 base_url 猜。"""
    cfg = llm_cfg or {}
    explicit = str(cfg.get("vendor") or "").strip()
    if explicit in VENDOR_REGISTRY:
        return explicit
    return guess_vendor_from_url(cfg.get("base_url", ""))


def mask_secret(secret: str, keep: int = 4) -> str:
    """密钥掩码，用于界面回显 / 日志。**永远不要把原值写进日志。**"""
    s = str(secret or "")
    if not s:
        return ""
    if len(s) <= keep * 2:
        return "*" * len(s)
    return f"{s[:keep]}***{s[-2:]}"
