"""文本小工具（移植自一览成文 text_utils.py，只取被本模块用到的部分）。

移植说明：原文件还有繁简转换（opencc）和字幕清洗，本模块用不到，
所以不搬——搬进来就是两份会各自漂移的同义实现。
"""

from __future__ import annotations

from typing import Optional


def format_time(seconds: float) -> str:
    """秒 → 中文紧凑时长显示。"""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    if seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}分{secs:.0f}秒"

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours}小时{minutes}分"


def sanitize_filename(name: str, max_length: int = 60) -> str:
    """生成文件名安全的一段。"""
    safe_name = "".join(c for c in name if c.isalnum() or c in (" ", "-", "_")).strip()
    safe_name = "_".join(safe_name.split())
    return safe_name[:max_length] or "untitled"


def sanitize_path_component(name: str) -> str:
    """目录名安全化，但保留中文（文章标题是中文，直接过滤掉就没意义了）。"""
    invalid_chars = '<>:"/\\|?*'
    safe_name = "".join("_" if c in invalid_chars else c for c in name).strip()
    safe_name = safe_name.rstrip(". ")
    return safe_name or "untitled"


def import_required(module_name: str, install_name: Optional[str] = None):
    """延迟 import，缺依赖时给一句能照做的提示。"""
    try:
        return __import__(module_name)
    except ImportError as e:
        package_name = install_name or module_name
        raise RuntimeError(
            f"缺少依赖 {package_name}，请先运行: "
            f"runtime/python.exe -m pip install {package_name}"
        ) from e
