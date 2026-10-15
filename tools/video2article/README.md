# 视频转文章（video2article）

给 GPT-SoVITS 补上「读完之后**写**」的一环。

GPT-SoVITS 原生能读（ASR 识别音频），但读出来的只是一坨带时间戳的流水账；
这个模块把它变成一篇**有结构、有 frontmatter、可直接发布**的文章。

```
链接 / 本地音视频
    → yt-dlp 下载（可选，只要音频）
    → ffmpeg 抽音（16kHz 单声道）
    → 云端转写（讯飞听见 / 自定义 POST）
    → 大模型成文（prompts/articles 模板）
    → output/video2article/<文件名>/{transcript.md, article.md}
```

能力移植自 **[一览成文 YilanChengWen](https://github.com/CoolCoolC666/video-to-article)**
（MIT，同一作者），并适配 GPT-SoVITS 的运行环境。

---

## 快速开始

### 1. 依赖

```bash
runtime/python.exe -m pip install openai yt-dlp anthropic
```

`requests` / `gradio` 在 GPT-SoVITS 里已有。不装也能启动——模块会在用到的
地方给出明确提示，其余功能不受影响。

### 2. 配置

本模块**不自带配置**，而是**只读**一览成文的 `config.json`：

| 查找顺序 | 路径 |
|---|---|
| 1 | 环境变量 `VIDEO2ARTICLE_YILAN_CONFIG` 指向的文件 |
| 2 | `tools/video2article/config.json`（已 gitignore） |
| 3 | 一览成文的常见安装位置（见 `config_bridge.py`） |

检查是否接上了：

```bash
runtime/python.exe tools/video2article/scripts/check_config.py
```

> ⚠ **密钥不要进仓库。** 一览成文的 `config.json` 里有讯飞
> APPID/SecretKey、LLM api_key、图床密钥。本模块只按路径读它，
> 从不复制、从不写入本仓库；`tools/video2article/config.json` 和
> `*_headers.txt` 已在 `.gitignore` 里。

需要的配置段：

```jsonc
{
  "transcribe": {
    "asr_engine": "xf_asr",        // 或 "custom_post"
    "xf_asr":     { "app_id": "...", "secret_key": "...", "language": "cn" },
    "custom_post": { "endpoint": "...", "api_key": "...", "api_style": "openai_chat" }
  },
  "llm": { "protocol": "openai_chat", "model": "...", "base_url": "...", "api_key": "..." }
}
```

### 3. 使用

- **WebUI**：重启 GPT-SoVITS，顶部多出「3-视频转文章」tab
- **独立窗口**（不重启 WebUI，推荐先这样试）：

  ```bash
  runtime/python.exe tools/video2article/scripts/run_tab_preview.py
  ```

- **代码调用**：

  ```python
  from tools.video2article import pipeline
  r = pipeline.run(r"D:\video\lecture.mp4", prompt_name="general_article")
  print(r.article_path)
  ```

---

## 成稿模板

文章长什么样，**完全由提示词模板决定**，改模板不用改代码。

```
tools/video2article/prompts/
  articles/     # 成稿模板，界面下拉里就是这一组
    general_article.md      # 通用长文（默认）
    snack_recipe.md         # 美食 / 食谱向
    limbus_company.md       # 垂类示例（含 contexts/ sources/ 配套素材）
  system/       # 基础提示词，界面默认不展示
```

新增模板：往 `articles/` 丢一个 `.md` 即可，**正文里必须含 `{transcript_text}`**
（缺了会告警并自动补在末尾）。

> 模板里的 `{transcript_text}` 是**字面替换**，不走 `str.format()`。
> 因为模板正文常含 Hexo / AnZhiYu 标签（`<%= %>`）和 Markdown 的 `{1,3}` 量词，
> 用 `str.format()` 会把它们当成字段直接抛异常。

---

## 目录结构

```
tools/video2article/
  config_bridge.py    只读外链一览成文的 config.json；就绪状态自检
  paths.py            运行期路径 + use_local_temp()（临时文件不许写 C 盘）
  logutil.py          本模块自己的 logger（**不动 root logger**）
  media.py            ffmpeg 定位 / 抽音 / 探测时长
  download.py         yt-dlp 薄封装
  prompts_store.py    模板发现与加载
  llm_providers.py    协议 / 厂商解析
  llm_client.py       LLM 成文调用
  asr_xf.py           讯飞听见（移植，见文件头「移植说明」）
  asr_custom_post.py  自定义 POST（移植，同上）
  pipeline.py         编排 + 自检
  webui_tab.py        Gradio 界面
  tests/              离线冒烟测试
  scripts/            check_config.py / run_tab_preview.py
```

---

## 两个移植时踩过的坑（别改回去）

### 1. 不劫持 root logger

一览成文的 `logging_config.py` 会把 **root logger** 的 handler 全摘掉换成自己的
——因为它假设自己是整个进程里唯一的日志来源。本模块是寄生在 GPT-SoVITS WebUI
里的一个 tab，`webui.py` 自己也往 root 挂了输出捕获。这里再动 root 的表现是
「点一下成文按钮，整个 WebUI 的日志和进度条全没了」，且极难定位。

所以 `logutil.py` 只给 `video2article` 自己的 logger 挂 handler，root 一个不碰。

### 2. 临时文件不许写 C 盘

两个云端引擎都对长音频切段（custom_post 每 450 秒一段）。`tempfile.mkdtemp()`
默认写系统 TEMP，也就是 **C 盘**——一次 1 小时视频的切片能到几百 MB。

`paths.use_local_temp()` 在「跑一次转写」这个明确边界内把落点改到项目内
`TEMP/video2article/`（E 盘），跑完立刻还原，且用上下文管理器保证异常路径也还原。
测试里有断言锁住这个行为。

---

## 移植来源与改动范围

`asr_xf.py` / `asr_custom_post.py` / 提示词模板来自
[CoolCoolC666/video-to-article](https://github.com/CoolCoolC666/video-to-article)（MIT）。

**协议逻辑一行未改**——原实现在字段名（`orderId` 而非 `taskId`）、状态字段
（`content.orderInfo.status` 而非 `taskStatus`）、`orderResult` 是 JSON 字符串、
上传必须 query string + raw body 而非 multipart 这些地方踩过坑，注释都留在
原处。移植只改了四处：导入路径、ffmpeg 定位、日志、以及删掉不属于本仓库的
「本 fork 改动」文档章节。

`custom_post_asr.py` 原先依赖一览成文的 `cover`（图床上传，67KB，含生图全套）
来给 DashScope 换公网 URL。本模块**没有** vendoring 它——GPT-SoVITS 没有生图
需求，搬进来是死代码；改成「装了 video-to-article 就借道，没装就给出三条
可操作出路」。
