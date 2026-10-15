"""LLM 成文调用（移植自一览成文 providers/llm.py）。

保留了上游两个关键处理，**不要**在移植时删掉：

## 1. max_tokens 被拒时降级重试

各家上限不一样（实测 DeepSeek 是 [1, 393216]），而配置里只有一个统一的
max_tokens 字段。换厂商就可能被服务端拒——而且这个错误发生在**请求发出前**，
用户看到的现象是「点了没产出」，完全不知道是超限。

处理：只在报错信息**明确指向 max_tokens** 时降级重试，并优先用服务端自己
报出来的上限（比任何硬编码都准）。不做无条件重试，避免把真实的鉴权/额度
错误也重试一遍。

## 2. 剥离开头的 <think>...</think>

推理模型（M3 / Qwen3 等）会把思维链塞进 content 里，直接展示给用户是噪音。
DeepSeek 这类把思维链放在独立 reasoning_content 字段的，OpenAI SDK 会自动
分开，本函数对它不命中也无害——保留是为了兼容前者。
"""

from __future__ import annotations

import re
import time
from typing import Optional

from .llm_providers import effective_vendor, resolve_protocol
from .logutil import get_logger
from .prompts_store import load_prompt, render_prompt
from .text_utils import format_time

logger = get_logger("llm")

# 服务端没在报错里写明上限时的降级目标（「几乎所有厂商都接受」的值）
_FALLBACK_MAX_TOKENS = 65536

# 形如 "the valid range of max_tokens is [1, 393216]"
_MAX_TOKENS_RANGE_RE = re.compile(
    r"max_tokens[^\[\]]*\[\s*(\d+)\s*,\s*(\d+)\s*\]", re.IGNORECASE
)

# 只处理开头首个块（含前后空白），不破坏正文
_THINK_BLOCK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


class LLMError(RuntimeError):
    """LLM 调用失败（消息面向用户，必须可操作）。"""


def _max_tokens_ceiling(error_text: str) -> int:
    """从报错里抠出服务端真实的 max_tokens 上限，取不到返回 0。"""
    m = _MAX_TOKENS_RANGE_RE.search(str(error_text or ""))
    if not m:
        return 0
    try:
        return int(m.group(2))
    except (TypeError, ValueError):
        return 0


def _strip_thinking_block(text: str) -> str:
    if not text:
        return text
    return _THINK_BLOCK_RE.sub("", text, count=1)


def _looks_like_html(text: str) -> bool:
    t = str(text or "").strip().lower()
    return t.startswith("<!doctype") or t.startswith("<html")


def build_prompt(transcript: str, prompt_name: str) -> str:
    """组装最终提示词（成文模板 + 转写正文）。"""
    template = load_prompt(prompt_name)
    if not template:
        raise LLMError(f"提示词模板不可用: {prompt_name}")
    return render_prompt(template, transcript)


def generate_article(
    transcript: str,
    llm_cfg: dict,
    prompt_name: str = "general_article",
) -> str:
    """用配置的 LLM 把转写正文整理成文章。失败抛 LLMError。"""
    if not isinstance(llm_cfg, dict) or not llm_cfg:
        raise LLMError("未配置大模型（config.json 的 llm 段缺失）")

    transcript = (transcript or "").strip()
    if not transcript:
        raise LLMError("转写文本为空，没有可整理的内容")

    protocol = resolve_protocol(llm_cfg)
    vendor = effective_vendor(llm_cfg)
    prompt = build_prompt(transcript, prompt_name)

    logger.info("使用 %s（%s 协议）+ 模板「%s」进行成文...", vendor, protocol, prompt_name)

    start = time.time()
    if protocol == "anthropic":
        text = _generate_anthropic(prompt, llm_cfg)
    else:
        text = _generate_openai(prompt, llm_cfg)
    logger.info("成文完成（耗时 %s，%d 字符）", format_time(time.time() - start), len(text))
    return text


def _generate_openai(prompt: str, cfg: dict) -> str:
    """OpenAI 兼容协议（含所有中转/自建网关）。"""
    try:
        from openai import OpenAI
    except ImportError as e:
        raise LLMError(
            "未安装 openai 库，请运行: runtime/python.exe -m pip install openai"
        ) from e

    client = OpenAI(
        api_key=cfg.get("api_key"),
        base_url=cfg.get("base_url", "https://api.openai.com/v1"),
        timeout=float(cfg.get("timeout_seconds", cfg.get("timeout", 180))),
        max_retries=int(cfg.get("max_retries", 3)),
    )

    want_max = int(cfg.get("max_tokens", 12000) or 12000)

    def _create(mt: int):
        return client.chat.completions.create(
            model=cfg.get("model", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=cfg.get("temperature", 0.3),
            max_tokens=mt,
        )

    try:
        try:
            response = _create(want_max)
        except Exception as first_exc:
            msg = str(first_exc)
            if "max_tokens" not in msg:
                raise
            ceiling = _max_tokens_ceiling(msg)
            safe = ceiling if ceiling > 0 else _FALLBACK_MAX_TOKENS
            logger.warning(
                "max_tokens=%d 被服务端拒绝（各家上限不同）。%s降到 %d 重试一次。",
                want_max,
                f"服务端报告上限为 {ceiling}，" if ceiling else "未能从报错里读出上限，",
                safe,
            )
            want_max = safe
            response = _create(want_max)
    except Exception as e:
        raise LLMError(f"LLM 调用失败：{e}") from e

    if hasattr(response, "choices"):
        content = response.choices[0].message.content
    elif isinstance(response, dict):
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
    else:
        raise LLMError(f"未知的响应格式: {type(response)}")

    if not content:
        raise LLMError("LLM 返回空内容")
    if _looks_like_html(content):
        raise LLMError("LLM 返回了 HTML 页面而不是文本（base_url 可能填错了）")

    cleaned = _strip_thinking_block(content)
    if not cleaned:
        raise LLMError("LLM 返回内容在剥离 thinking 块后为空")
    return cleaned


def _generate_anthropic(prompt: str, cfg: dict) -> str:
    """Anthropic 原生协议。"""
    try:
        from anthropic import Anthropic
    except ImportError as e:
        raise LLMError(
            "未安装 anthropic 库，请运行: runtime/python.exe -m pip install anthropic"
        ) from e

    client = Anthropic(api_key=cfg.get("api_key"))
    try:
        response = client.messages.create(
            model=cfg.get("model", "claude-3-5-sonnet-20241022"),
            max_tokens=int(cfg.get("max_tokens", 12000) or 12000),
            temperature=cfg.get("temperature", 0.3),
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        raise LLMError(f"Anthropic 调用失败：{e}") from e

    text = response.content[0].text
    cleaned = _strip_thinking_block(text)
    if _looks_like_html(cleaned):
        raise LLMError("LLM 返回了 HTML 页面而不是文本（base_url 可能填错了）")
    return cleaned
