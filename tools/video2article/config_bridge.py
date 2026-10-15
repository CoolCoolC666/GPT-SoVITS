"""配置桥接：把「一览成文」的 config.json 当作**只读外部配置**接进来。

## 为什么是只读外链，而不是复制一份到本仓库

一览成文的 `config.json` 里有讯飞 APPID/SecretKey、LLM api_key、图床密钥。
这些一旦复制进本仓库就会进 git 历史（删文件也删不掉，见 force push 也留旧 SHA）。
所以本模块只**按路径读**，从不写、从不拷贝、从不落盘到仓库内。

## 查找顺序（先到先用）

1. 环境变量 `VIDEO2ARTICLE_YILAN_CONFIG` 指向的文件
2. 本仓库侧覆盖：`<GPT-SoVITS>/tools/video2article/config.json`（已 gitignore）
3. 一览成文源码目录：`../YilanChengWen-src/config.json`、`../YilanChengWen-0.4.5/config.json`
4. 都没有 → 返回空配置，成文/LLM 步骤会给出「未配置」的可操作提示，而不是崩栈

**注意**：GPT-SoVITS 自带的 `config.py` 存在，但那管的是 TTS 模型路径等，
跟 ASR/LLM 毫无关系，**不要**去动它，也不要把它当这里的配置源。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .logutil import get_logger

logger = get_logger("config")

ENV_CONFIG_PATH = "VIDEO2ARTICLE_YILAN_CONFIG"

# 本仓库侧的本地覆盖（.gitignore 已忽略，不进版本库）
LOCAL_CONFIG_FILE = Path(__file__).resolve().parent / "config.json"

# 一览成文的常见安装位置。
# 相对路径按「GPT-SoVITS 的同级目录」解析；绝对路径是本机实际安装位置，
# 换机器时用环境变量 VIDEO2ARTICLE_YILAN_CONFIG 指过去即可（优先级最高）。
_YILAN_CANDIDATES = (
    "../YilanChengWen-src/config.json",
    "../YilanChengWen-0.4.5/config.json",
    "E:/000~/YilanChengWen-src/config.json",
    "E:/000~/YilanChengWen-0.4.5/config.json",
    "D:/Apps/YilanChengWen/config.json",
    "D:/YilanChengWen/config.json",
)


class ConfigError(RuntimeError):
    """配置缺失或不可用（面向用户，报错信息必须可操作）。"""


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("读取配置失败 %s: %s", path, e)
        return None
    return data if isinstance(data, dict) else None


def candidate_paths() -> List[Path]:
    """按优先级返回候选配置文件路径（存在的排在前面）。"""
    found: List[Path] = []

    env_path = os.environ.get(ENV_CONFIG_PATH, "").strip()
    if env_path:
        found.append(Path(env_path))

    found.append(LOCAL_CONFIG_FILE)

    app_root = Path(__file__).resolve().parent.parent.parent
    for rel in _YILAN_CANDIDATES:
        p = Path(rel)
        found.append(p if p.is_absolute() else (app_root / rel).resolve())

    seen = set()
    unique: List[Path] = []
    for p in found:
        key = str(p).lower()
        if key in seen:
            continue
        seen.add(key)
        if p.is_file():
            unique.append(p)
    return unique


def load_config(force_reload: bool = False) -> Dict[str, Any]:
    """读取配置。找不到返回空 dict（不抛异常——让上层按「缺什么补什么」提示）。"""
    global _CACHE
    if _CACHE is not None and not force_reload:
        return _CACHE

    for path in candidate_paths():
        data = _read_json(path)
        if data is not None:
            logger.info("已加载配置: %s", path)
            _CACHE = data
            return data

    logger.warning(
        "未找到一览成文 config.json；可用环境变量 %s 指定，或在 "
        "%s 放一份（该文件不会被提交）",
        ENV_CONFIG_PATH,
        LOCAL_CONFIG_FILE,
    )
    _CACHE = {}
    return _CACHE


_CACHE: Optional[Dict[str, Any]] = None


def get_section(*path: str) -> Dict[str, Any]:
    """取嵌套配置段，例如 ``get_section("transcribe", "xf_asr")``。"""
    node: Any = load_config()
    for key in path:
        if not isinstance(node, dict):
            return {}
        node = node.get(key)
    return node if isinstance(node, dict) else {}


# ============ 各能力的「是否就绪」检查（给界面用，别让用户点了才报错） ============


def llm_ready() -> Tuple[bool, str]:
    """大模型是否可用。返回 (是否就绪, 给用户看的原因)。"""
    cfg = get_section("llm")
    if not cfg:
        return False, "未找到 config.json 的 llm 段"
    if not str(cfg.get("api_key") or "").strip():
        return False, "llm.api_key 为空"
    if not str(cfg.get("model") or "").strip():
        return False, "llm.model 为空"
    return True, f"{resolve_protocol(cfg)} / {effective_vendor(cfg)} / {cfg.get('model')}"


def asr_ready() -> Tuple[bool, str, str]:
    """云端 ASR 是否可用。返回 (是否就绪, 引擎名, 原因)。"""
    transcribe = get_section("transcribe")
    engine = str(transcribe.get("asr_engine") or "").strip()

    if engine == "xf_asr":
        cfg = get_section("transcribe", "xf_asr")
        if bool(cfg.get("mock", False)):
            return True, "xf_asr", "Mock 模式（未调真实 API）"
        missing = [k for k in ("app_id", "secret_key") if not str(cfg.get(k) or "").strip()]
        if missing:
            return False, "xf_asr", f"缺少凭证: {', '.join(missing)}"
        return True, "xf_asr", f"讯飞听见（{cfg.get('language', 'cn')}）"

    if engine in ("custom_post", "minimax_asr"):
        cfg = get_section("transcribe", "custom_post")
        if not cfg:
            cfg = get_section("transcribe", "minimax_asr")
        if bool(cfg.get("mock", False)):
            return True, "custom_post", "Mock 模式（未调真实 API）"
        if not str(cfg.get("endpoint") or "").strip():
            return False, "custom_post", "缺少 endpoint"
        if not str(cfg.get("api_key") or "").strip() and not _has_headers_auth(cfg):
            return False, "custom_post", "缺少 api_key，且请求头文件里也没有鉴权头"
        return True, "custom_post", f"自定义 POST（{cfg.get('endpoint')}）"

    return False, engine or "(未设置)", (
        f"transcribe.asr_engine = {engine or '(空)'}，本模块只做云端引擎"
        "（xf_asr / custom_post）"
    )


def _has_headers_auth(cfg: Dict[str, Any]) -> bool:
    """请求头文件里是否带了 Authorization 一类的鉴权头。"""
    from .paths import CUSTOM_POST_HEADERS_FILE

    path_str = str(cfg.get("headers_file") or "").strip()
    path = Path(path_str) if path_str else CUSTOM_POST_HEADERS_FILE
    if not path.is_file():
        return False
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if ":" not in line:
                continue
            key = line.split(":", 1)[0].strip().lower()
            if key in ("authorization", "x-api-key", "api-key"):
                return True
    except OSError:
        return False
    return False


def describe() -> Dict[str, Any]:
    """给界面「配置状态」面板用的一份脱敏快照。"""
    cfg_path = candidate_paths()[0] if candidate_paths() else None
    llm_ok, llm_why = llm_ready()
    asr_ok, asr_engine, asr_why = asr_ready()
    return {
        "config_path": str(cfg_path) if cfg_path else "(未找到)",
        "llm_ready": llm_ok,
        "llm_detail": llm_why,
        "asr_ready": asr_ok,
        "asr_engine": asr_engine,
        "asr_detail": asr_why,
    }


# 协议/厂商解析放在 llm_providers，这里 import 避免重复实现
from .llm_providers import effective_vendor, resolve_protocol  # noqa: E402
