"""成稿提示词模板的发现与加载（移植自一览成文 prompts.py）。

目录约定（与一览成文一致，模板文件可直接互相拷贝）：

    tools/video2article/prompts/
      articles/   # 成稿模板，界面下拉里就是这一组
      system/     # 基础/中间提示词，界面上默认不展示

**关键契约**：成稿模板正文里必须含 `{transcript_text}` 占位符 ——
渲染时只做字面替换，不走 str.format()。原因见 `render_prompt`：

模板正文里常有 Hexo / AnZhiYu 这类 `<%= %>` `<% %>` 标签，以及 Markdown 里的
`{1,3}` 之类的花括号量词。若走 str.format()，这些会被当成 format field 抛
KeyError / IndexError。一览成文在这上面已经踩过一次，本移植保持同样的做法。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from .logutil import get_logger
from .paths import PROMPTS_DIR

logger = get_logger("prompts")

SYSTEM_SUBDIR = "system"
ARTICLES_SUBDIR = "articles"

# 模板缺占位符时的兜底（保证一定有产出可问）
FALLBACK_TEMPLATE = "请优化以下文本，去除冗余，重构逻辑结构：\n\n{transcript_text}"

# 旧版扁平布局里的「基础提示词」名，成稿下拉里要排除掉
_SYSTEM_STEMS = {"format", "summary", "evaluation"}


def ensure_prompt_dirs() -> None:
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    (PROMPTS_DIR / SYSTEM_SUBDIR).mkdir(exist_ok=True)
    (PROMPTS_DIR / ARTICLES_SUBDIR).mkdir(exist_ok=True)


def _is_prompt_file(path: Path) -> bool:
    if path.suffix.lower() != ".md":
        return False
    if path.name.lower() == "readme.md":
        return False
    try:
        return path.stat().st_size > 0 and bool(path.read_text(encoding="utf-8").strip())
    except OSError:
        return False


def _collect_stems(directory: Path) -> List[str]:
    if not directory.is_dir():
        return []
    return [p.stem for p in sorted(directory.glob("*.md")) if _is_prompt_file(p)]


def list_article_prompts() -> List[str]:
    """成稿模板名（界面下拉用）。"""
    ensure_prompt_dirs()
    names = list(_collect_stems(PROMPTS_DIR / ARTICLES_SUBDIR))

    # 兼容旧版扁平布局：prompts/xxx.md（排除基础提示词）
    for stem in _collect_stems(PROMPTS_DIR):
        if stem not in _SYSTEM_STEMS and stem not in names:
            names.append(stem)

    if not names:
        names = [n for n in _collect_stems(PROMPTS_DIR) if n not in _SYSTEM_STEMS]
    return sorted(set(names))


def list_system_prompts() -> List[str]:
    """基础/中间提示词名（界面默认不展示）。"""
    ensure_prompt_dirs()
    names = set(_collect_stems(PROMPTS_DIR / SYSTEM_SUBDIR))
    for stem in _SYSTEM_STEMS:
        if (PROMPTS_DIR / f"{stem}.md").is_file() and _is_prompt_file(PROMPTS_DIR / f"{stem}.md"):
            names.add(stem)
    return sorted(names)


def resolve_prompt_path(prompt_name: str) -> Optional[Path]:
    """把模板名解析成文件路径。"""
    name = (prompt_name or "").strip().replace("\\", "/")
    if not name:
        return None

    # 显式相对路径（如 "articles/general_article.md"）
    if "/" in name or name.endswith(".md"):
        rel = name if name.endswith(".md") else f"{name}.md"
        candidate = PROMPTS_DIR / rel
        if candidate.is_file():
            return candidate

    stem = name[:-3] if name.endswith(".md") else name
    for path in (
        PROMPTS_DIR / ARTICLES_SUBDIR / f"{stem}.md",
        PROMPTS_DIR / SYSTEM_SUBDIR / f"{stem}.md",
        PROMPTS_DIR / f"{stem}.md",
    ):
        if path.is_file():
            return path
    return None


def render_prompt(template: str, transcript: str) -> str:
    """把转写文本填进模板。

    只做字面替换，**不**用 str.format()——见模块 docstring。
    """
    return template.replace("{transcript_text}", transcript)


def load_prompt(prompt_name: str = "general_article") -> Optional[str]:
    """按名字加载成稿模板；找不到时回落到兜底模板而不是返回 None。"""
    ensure_prompt_dirs()
    prompt_file = resolve_prompt_path(prompt_name)

    if prompt_file is None:
        logger.warning("提示词文件不存在: %s", prompt_name)
        logger.warning("可用的提示词: %s", ", ".join(list_article_prompts()))
        logger.warning("使用默认提示词")
        return FALLBACK_TEMPLATE

    try:
        content = prompt_file.read_text(encoding="utf-8").strip()
    except OSError as e:
        logger.error("读取提示词失败 %s: %s", prompt_file, e)
        return FALLBACK_TEMPLATE

    if not content:
        logger.error("提示词文件为空: %s", prompt_file)
        return FALLBACK_TEMPLATE

    if "{transcript_text}" not in content:
        # 缺占位符 = 模板作者忘了写。不静默产出「只有提示词没有内容」的文章，
        # 补上并明确告警。
        logger.warning("提示词文件缺少 {transcript_text} 占位符: %s", prompt_file)
        logger.warning("已在末尾自动补上占位符")
        content += "\n\n{transcript_text}"

    return content


def default_article_prompt_names() -> List[str]:
    """默认成稿模板：优先 general_article，退到第一个。"""
    articles = list_article_prompts()
    if "general_article" in articles:
        return ["general_article"]
    return articles[:1]
