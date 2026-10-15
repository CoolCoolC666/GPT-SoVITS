# 成稿提示词（articles）

本目录存放**会产出可发布/可交付文章**的模板。

- GUI 默认只展示这里的模板（下拉选择）。
- 新增类型：复制一份 md，文件名即 `--prompts` 名称，内容须包含 `{transcript_text}`。

当前示例：

- `snack_recipe.md` — 美食教程（Hexo / 博客向）
- `general_article.md` — 通用模板（任何主题都能用，带游戏识别规则）
- `limbus_company.md` — **Limbus Company / Project Moon 专版**（2026-09-12 fork 新增）

Qwen3-ASR 专名词表参考站清单见 `sources/` 子目录：

- `sources/starrail.yaml` — 崩坏：星穹铁道（v1 已验证 2026-09-07）
- `sources/limbus_company.yaml` — Limbus Company（2026-09-12 v1 骨架，待用户补充 + 验证）

词表生成流程：`build_context.py` 读 yaml → fetch URL → LLM 抽取 → 写 `contexts/*.txt`（Qwen3-ASR `context_file` 用）。
