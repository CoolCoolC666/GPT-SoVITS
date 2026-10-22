"""云端 ASR 引擎的配置存储（GPT-SoVITS 侧）。

## 为什么要单独一份配置

上游 GPT-SoVITS 的 `config.py` 管的是 TTS（模型权重、bert/ssl 路径），
跟 ASR 引擎毫无关系。所以这里另起一份，专门放**云端 ASR 引擎**的凭证与参数。

存放位置：`tools/asr/cloud_config.json`（已被 .gitignore 忽略）。

## 配置优先级

    tools/asr/cloud_config.json   ← 本文件，界面「设置」保存的（最高）
    ↓ 为空则回落
    一览成文的 config.json          ← 只读外链，界面「从一览成文导入」的来源
    ↓ 再为空
    环境变量

「从一览成文导入」是**拷贝**（把值写进本文件），不是引用——之后一览成文那边
改了配置，这边不会跟着变。导入按钮会提示这一点，免得两边不一致时找不到原因。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

ASR_DIR = Path(__file__).resolve().parent
CLOUD_CONFIG_FILE = ASR_DIR / "cloud_config.json"
DEFAULT_HEADERS_FILE = ASR_DIR / "cloud_headers.txt"

# 一览成文 config.json 的候选位置（与 tools/video2article/config_bridge.py 同源）
_YILAN_CANDIDATES = (
    "../YilanChengWen-src/config.json",
    "../YilanChengWen-0.4.5/config.json",
    "E:/000~/YilanChengWen-src/config.json",
    "E:/000~/YilanChengWen-0.4.5/config.json",
    "D:/Apps/YilanChengWen/config.json",
    "D:/YilanChengWen/config.json",
)

XF_DEFAULTS: Dict[str, Any] = {
    "app_id": "",
    "secret_key": "",
    "language": "cn",
    "mock": False,
    "role_separation": False,
    "role_num": 2,
    "pd_domain": "",
    "colloquial_proc": False,
    "timestamps": True,
    "max_wait_seconds": 600,
}

CUSTOM_POST_DEFAULTS: Dict[str, Any] = {
    "endpoint": "",
    "api_key": "",
    "api_style": "openai_compat",
    "model": "",
    "headers_file": "",
    "language": "",
    "mock": False,
    "role_separation": False,
    "speaker_count": 2,
    "timestamps": True,
    "colloquial_proc": False,
    "max_wait_seconds": 600,
}

DEFAULTS: Dict[str, Any] = {
    "engine": "xf_asr",
    "concurrency": 4,
    "xf_asr": dict(XF_DEFAULTS),
    "custom_post": dict(CUSTOM_POST_DEFAULTS),
}

SECRET_FIELDS = ("secret_key", "api_key", "app_id")


def _deep_merge(base: dict, override: dict) -> dict:
    """递归合并；override 里的 dict 逐键覆盖 base。"""
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _read_json(path: Path) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def find_yilan_config() -> Optional[Path]:
    """找到一览成文的 config.json（只读）。"""
    env = os.environ.get("VIDEO2ARTICLE_YILAN_CONFIG", "").strip()
    if env and Path(env).is_file():
        return Path(env)
    root = ASR_DIR.parent.parent
    for rel in _YILAN_CANDIDATES:
        p = Path(rel)
        candidate = p if p.is_absolute() else (root / rel)
        if candidate.is_file():
            return candidate
    return None


def load_yilan_config() -> Optional[dict]:
    path = find_yilan_config()
    if path is None:
        return None
    return _read_json(path)


def load_local() -> dict:
    """读 GPT-SoVITS 侧的配置（不含回落）。"""
    data = _read_json(CLOUD_CONFIG_FILE)
    if data is None:
        return {}
    return data


def load() -> dict:
    """最终生效配置：本地 > 一览成文（只读） > 默认值。"""
    merged = _deep_merge(DEFAULTS, {})
    yilan = load_yilan_config()
    if yilan:
        transcribe = yilan.get("transcribe") or {}
        merged = _deep_merge(merged, {
            "xf_asr": transcribe.get("xf_asr") or {},
            "custom_post": transcribe.get("custom_post") or transcribe.get("minimax_asr") or {},
        })
        engine = transcribe.get("asr_engine")
        if engine:
            merged["engine"] = "custom_post" if engine == "minimax_asr" else engine
    merged = _deep_merge(merged, load_local())
    return merged


def save(config: dict) -> Path:
    """保存到 tools/asr/cloud_config.json（该文件在 .gitignore 里）。"""
    payload = _deep_merge(DEFAULTS, config or {})
    with open(CLOUD_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return CLOUD_CONFIG_FILE


def import_from_yilan() -> tuple:
    """从一览成文 config.json 导入云端 ASR 配置。

    返回 (是否成功, 说明文字, 导入后的配置)。**只拷贝 ASR 相关字段**，
    不会把 LLM/图床的密钥一并搬过来。
    """
    path = find_yilan_config()
    if path is None:
        return False, "未找到一览成文的 config.json", None

    yilan = _read_json(path)
    if not yilan:
        return False, f"一览成文的 config.json 解析失败: {path}", None

    transcribe = yilan.get("transcribe") or {}
    xf = transcribe.get("xf_asr") or {}
    cp = transcribe.get("custom_post") or transcribe.get("minimax_asr") or {}

    if not xf and not cp:
        return False, "一览成文的 config.json 里没有 transcribe.xf_asr / custom_post 段", None

    imported = {
        "xf_asr": {k: v for k, v in xf.items() if k in XF_DEFAULTS},
        "custom_post": {k: v for k, v in cp.items() if k in CUSTOM_POST_DEFAULTS},
    }
    engine = str(transcribe.get("asr_engine") or "").strip()
    if engine:
        imported["engine"] = "custom_post" if engine == "minimax_asr" else engine

    # 只导入有值的字段，空值不覆盖用户本地已填的
    current = load_local()
    for section in ("xf_asr", "custom_post"):
        for key, value in (imported.get(section) or {}).items():
            if value not in ("", None, [], {}):
                current.setdefault(section, {})[key] = value
    if imported.get("engine"):
        current["engine"] = imported["engine"]

    save(current)
    return True, f"已从 {path} 导入云端 ASR 配置", current


def get_engine_config(engine: str) -> dict:
    """取某个引擎的最终配置（已合并默认值与回落）。"""
    cfg = load()
    if engine == "xf_asr":
        return dict(cfg.get("xf_asr") or {})
    if engine in ("custom_post", "minimax_asr"):
        return dict(cfg.get("custom_post") or {})
    return {}


def readiness() -> dict:
    """就绪状态自检（只报告「有没有」，不打印密钥原文）。"""
    cfg = load()
    xf = cfg.get("xf_asr") or {}
    cp = cfg.get("custom_post") or {}

    if bool(xf.get("mock")):
        xf_state, xf_why = True, "Mock 模式（不调真实 API）"
    elif xf.get("app_id") and xf.get("secret_key"):
        xf_state, xf_why = True, f"讯飞听见（{xf.get('language', 'cn')}）"
    else:
        xf_state, xf_why = False, "缺少 app_id 或 secret_key"

    if bool(cp.get("mock")):
        cp_state, cp_why = True, "Mock 模式（不调真实 API）"
    elif cp.get("endpoint") and (cp.get("api_key") or _headers_have_auth(cp)):
        cp_state, cp_why = True, f"自定义 POST（{cp.get('endpoint')}）"
    else:
        cp_state, cp_why = False, "缺少 endpoint，或缺少 api_key / 请求头文件里的鉴权头"

    return {
        "config_path": str(CLOUD_CONFIG_FILE),
        "config_exists": CLOUD_CONFIG_FILE.is_file(),
        "yilan_config": str(find_yilan_config()) if find_yilan_config() else "(未找到)",
        "engine": cfg.get("engine"),
        "concurrency": cfg.get("concurrency", 4),
        "xf_asr": {"ready": xf_state, "detail": xf_why},
        "custom_post": {"ready": cp_state, "detail": cp_why},
    }


def _headers_have_auth(cp: dict) -> bool:
    """请求头文件里是否带了鉴权头。"""
    path_str = str(cp.get("headers_file") or "").strip()
    path = Path(path_str) if path_str else DEFAULT_HEADERS_FILE
    if not path.is_file():
        return False
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if ":" not in line:
                continue
            if line.split(":", 1)[0].strip().lower() in ("authorization", "x-api-key", "api-key"):
                return True
    except OSError:
        return False
    return False


def mask(value: str, keep: int = 4) -> str:
    """密钥掩码。"""
    s = str(value or "")
    if not s:
        return ""
    if len(s) <= keep * 2:
        return "*" * len(s)
    return f"{s[:keep]}***{s[-2:]}"
