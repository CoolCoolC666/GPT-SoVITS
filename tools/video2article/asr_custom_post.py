"""自定义 POST 云端 ASR（OpenAI 兼容 / DashScope 异步）。

=========================== 移植说明 ===========================

本文件由 CoolCoolC666/video-to-article（MIT）的
src/video_to_article/media/custom_post_asr.py 移植而来，只改了四处：

1. 相对导入改为本模块内的绝对导入（logging_config -> logutil 等）
2. ffmpeg/ffprobe 定位改为复用 media.py（优先 GPT-SoVITS 自带的
   runtime/ffmpeg.exe），原版硬编码的一览成文安装路径已删除
3. 日志不再劫持 root logger
4. 原 docstring 里的「本 fork 改动」章节已删除（那是另一个仓库的说明）

**协议逻辑一行未改。** 原实现踩过的坑都写在下面各处注释里
（字段名是 orderId 不是 taskId、状态是 content.orderInfo.status 不是
taskStatus、orderResult 是 JSON 字符串、上传必须 query string + raw body
而不是 multipart……）。改动前请先读那些注释。
===============================================================
"""

from __future__ import annotations

import atexit
import glob
import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.parse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .logutil import configure_logging
from .paths import CUSTOM_POST_HEADERS_FILE
from .text_utils import format_time, import_required

logger = configure_logging()


# 默认端点（用户可在 GUI 改成任意兼容服务）
DEFAULT_ENDPOINT = "https://api.minimax.cn/v1/speech_to_text"
LEGACY_ENDPOINT = DEFAULT_ENDPOINT
_MODEL = "asr-1.0"

# 请求头文件：程序内，默认文件名。用戶可直接编辑（每行一条 Key: Value）
HEADERS_FILENAME = "custom_post_headers.txt"

# ============ API 风格（2026-10-04 新增）============
# 同一个「自定义 POST」框要接两类**协议完全不同**的服务，所以必须显式选风格。
# 不是「换个 URL」那么简单——请求体、鉴权、音频传递方式、拿结果的方式全都不同。
#   2026-10-04 实测：往 DashScope 走 OpenAI 兼容的 multipart，会被中间层断连（10054）。
API_STYLES = {
    "openai_compat": "OpenAI 兼容（MiniMax / 中转 / vLLM 等）",
    "dashscope_async": "DashScope 异步（阿里云百炼 Qwen3-ASR）",
}

# ---------- 风格 openai_compat：一次性 multipart 上传，直接返回结果 ----------
OPENAI_COMPAT_PATH = "/speech_to_text"
# 单请求 ≤ 500 秒 / 50 MB（OpenAI 兼容侧 OpenAPI 明确写出）
MAX_DURATION_SEC = 500     # 时长上限，超了 400
MAX_FILE_BYTES = 50 * 1024 * 1024  # 大小上限，超了 413
# 切段留 50 秒余量：ffprobe 探测有误差，卡着 500s 切必 400
CHUNK_SEGMENT_SEC = 450

# ---------- 风格 dashscope_async：提交任务 → 轮询 → 下载结果 ----------
# 官方文档（2026-10-04 查证并实测确认）：
#   POST {base}/services/audio/asr/transcription
#     Header: Authorization: Bearer <key>  +  X-DashScope-Async: enable
#     Body:   application/json（**不是 multipart**）
#            {"model": "...", "input": {"file_urls": ["<公网URL>"]},
#             "parameters": {"channel_id": [0], "language_hints": ["zh","en"]}}
#   GET  {base}/tasks/{task_id}   → output.results[].transcription_url
#   GET  {transcription_url}      → 完整结果 JSON（24 小时有效，务必及时下载）
# ⚠ 关键约束：**音频必须是公网 URL**，不接受文件上传。本模块靠图床中转拿 URL。
DASHSCOPE_TASK_PATH = "/services/audio/asr/transcription"
# 异步转写支持单文件 12 小时 / 2 GB —— 比 OpenAI 兼容宽松得多，基本不需要切段
DASHSCOPE_MAX_DURATION_SEC = 12 * 3600
DASHSCOPE_MAX_BYTES = 2 * 1024 * 1024 * 1024
DASHSCOPE_POLL_INTERVAL = 5.0
DASHSCOPE_DONE = {"SUCCEEDED", "SUCCESS", "COMPLETED", "DONE"}
DASHSCOPE_FAILED = {"FAILED", "CANCELED", "CANCELLED", "UNKNOWN"}

# 支持的音频格式（OpenAI 兼容侧 OpenAPI 列的）
SUPPORTED_EXTS = {".wav", ".aiff", ".aif", ".flac", ".alac", ".m4a",
                  ".mp3", ".aac", ".opus", ".ogg"}

# 官方支持的语言（BCP-47）；留空 = 自动检测 + 中英混说
MINIMAX_LANGUAGES = {
    "": "自动检测（含中英混说）",
    "zh": "中文（zh）",
    "yue": "粤语（yue）",
    "en": "英语（en）",
    "ja": "日语（ja）",
    "ko": "韩语（ko）",
    "th": "泰语（th）",
    "vi": "越南语（vi）",
    "id": "印尼语（id）",
    "ms": "马来语（ms）",
    "fil": "菲律宾语（fil）",
    "ar": "阿拉伯语（ar）",
    "tr": "土耳其语（tr）",
    "fr": "法语（fr）",
    "de": "德语（de）",
    "es": "西班牙语（es）",
    "it": "意大利语（it）",
    "pt": "葡萄牙语（pt）",
    "pl": "波兰语（pl）",
    "ru": "俄语（ru）",
    "uk": "乌克兰语（uk）",
}

# 错误码 → 人话（用于把 OpenAI 风格的 error.message 翻译成可操作提示）
_ERROR_HINTS = {
    400: "请求参数错误（最常见：音频超过 500 秒或格式不支持）",
    401: "鉴权失败：检查 API Key / 请求头文件里的 Authorization 是否正确",
    402: "账户余额不足：去 https://platform.minimax.cn/user-center/basic-information 充值",
    403: "无权限：该端点拒绝了当前凭证（检查请求头文件里的鉴权信息）",
    404: "端点不存在：检查「接口地址」填的 URL 是否正确",
    413: "音频超过 50 MB 上限：转成单声道 16kHz 或 mp3 压缩后再试",
    422: "音频内容涉及敏感内容，被平台拒绝",
    429: "触发限流：稍后重试",
    500: "服务端错误：稍后重试",
}

# ffmpeg/ffprobe 路径（与 xf_asr 同款，优先项目内副本）

_cached_session: Optional[object] = None


# ============ 请求头文件 ============
def default_headers_file() -> Path:
    """程序内默认请求头文件路径（与 exe / 源码同目录）。"""
    return Path(CUSTOM_POST_HEADERS_FILE)


def write_headers_template(path: Optional[Path] = None, overwrite: bool = False) -> Path:
    """生成请求头文件模板（首次用 / 用户点「生成模板」时调用）。

    只写占位说明，**不写入真实密钥**。
    """
    target = Path(path) if path else default_headers_file()
    if target.exists() and not overwrite:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "# 自定义 POST ASR — 附加请求头\n"
        "#\n"
        "# 格式：每行一条  Key: Value\n"
        "#  - # 开头是注释，空行忽略\n"
        "#  - 这里的头会与程序自动生成的合并（同名以本文件为准）\n"
        "#  - Authorization 通常在「API Key」输入框里填，不用写这里；\n"
        "#    只有当服务端要求非 Bearer 鉴权时才写在这里\n"
        "#  - 本文件可能含密钥，已加入 .gitignore，不要提交到仓库\n"
        "#\n"
        "# 例：私有部署要额外带一个 token 头\n"
        "# X-Deploy-Token: your-token-here\n"
        "#\n"
        "# 例：走网关要加路由头\n"
        "# X-Gateway-Route: asr-prod\n",
        encoding="utf-8",
    )
    logger.info(f"已生成请求头模板: {target}")
    return target


# === HTTP 头名合法性（RFC 7230 token）===
# ⚠ 用户已经三次把「不是请求头的东西」填进请求头文件：
#   ① requirements.txt 的依赖清单  ② DashScope 的 Python 示例  ③ 文档里的 curl 示例
# 第③次里程序解析出的"头名"是 '"model"' / '--header "Authorization' /
#   "curl --location 'https" —— 带引号、空格、整条命令，然后**真的发出去**，
#   服务器直接 ConnectionResetError 10054，跟模型、密钥都毫无关系。
# 所以这里按 HTTP 规范校验：头名只能是 token（字母数字 + !#$%&'*+-.^_`|~），
# 出现空格/引号/中文/斜杠一律不是头名。
_HTTP_TOKEN_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "!#$%&'*+-.^_`|~"
)


def _is_valid_http_header_name(name: str) -> bool:
    """头名是否合法 HTTP token。中文头名在实践中也不被支持，一并拒。"""
    n = str(name or "")
    return bool(n) and all(ch in _HTTP_TOKEN_CHARS for ch in n)


def load_headers_file(path: Optional[Path] = None) -> Dict[str, str]:
    """读请求头文件 → dict。文件不存在就生成模板并返回空 dict。

    解析规则：
      - 一行一条 `Key: Value`
      - `#` 开头或行内 `  #` 之后是注释
      - 空行忽略
      - 没有冒号的行忽略（并 WARNING，避免静默吃掉用户的输入）
      - 键大小写不敏感去重（后写的覆盖先写的）
    """
    target = Path(path) if path else default_headers_file()
    if not target.exists():
        write_headers_template(target)
        return {}

    headers: Dict[str, str] = {}
    # 收集「有 Key: Value 但键名不是合法 HTTP 头名」的项（curl/JSON/Python 示例）
    bad_names: List[Tuple[int, str]] = []
    # 区分「纯注释模板」与「非请求头文件」：
    # 程序自动生成的默认模板**本来就只有注释**，解析不出 Key: Value 是正常的，
    # 再警告一次就变成了「让你留空 → 读默认 → 默认又报警」的绕圈提示。
    # 只有当文件里**确实有非注释内容**却一个 Key: Value 都解析不出时，才该报警。
    saw_content = False
    try:
        raw = target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning(f"读取请求头文件失败（已忽略该文件）: {target} — {exc}")
        return {}

    for lineno, line in enumerate(raw.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # 去掉行内注释（要求 # 前有空白，避免砍掉 token 里的 #）
        if " #" in line:
            line = line.split(" #", 1)[0].strip()
        if not line:
            continue
        saw_content = True
        if ":" not in line:
            logger.warning(
                f"请求头文件第 {lineno} 行没有 ':'，已忽略: {line[:60]!r}"
            )
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        # ⚠ 头名必须是 HTTP token。带引号/空格/中文/斜杠的（多半来自
        #   curl 示例、JSON、Python 代码）不是头名，**发出去服务端会直接掐连接**
        #   报一个毫无线索的 ConnectionResetError 10054。
        if not _is_valid_http_header_name(key):
            bad_names.append((lineno, key))
            continue
        # 同名（忽略大小写）后写的覆盖
        for existing in list(headers):
            if existing.lower() == key.lower() and existing != key:
                headers[key] = headers.pop(existing)
                break
        headers[key] = value

    if bad_names:
        sample = ", ".join(repr(k) for _, k in bad_names[:3])
        more = f" 等 {len(bad_names)} 个" if len(bad_names) > 3 else ""
        logger.warning(
            f"请求头文件第 "
            + "、".join(str(n) for n, _ in bad_names[:5])
            + ("…" if len(bad_names) > 5 else "")
            + f" 行的「键名」不是合法 HTTP 头名，已全部忽略: {sample}{more}\n"
            "  （HTTP 头名不能含空格 / 引号 / 中文 / 斜杠）\n"
            "  看到这种形态，通常是**把文档里的 curl 示例或 JSON 片段填进来了**。\n"
            "  请求头文件每行只能是 `X-Header-Name: value` 这种形式。"
        )

    # ⚠ 一个合法头都没解析出来，但文件里**确实有**非注释内容 →
    #   说明整个文件都不是请求头。此时必须**明确失败**：
    #   静默继续的话，这些垃圾头会被真的发出去，服务端掐连接，
    #   用户看到的是 ConnectionResetError 10054 —— 与真实原因毫无关联。
    if not headers and bad_names:
        raise RuntimeError(
            f"请求头文件 {target} 里没有任何合法请求头"
            f"（解析出 {len(bad_names)} 个键名，但全都不是合法 HTTP 头名）。\n\n"
            "这个文件很可能是**文档里的 curl / JSON / Python 示例**，"
            "不是请求头文件。\n"
            "「请求头文件」输入框只接受每行形如：\n"
            "    X-Deploy-Token: your-token-here\n"
            "    Authorization: Bearer sk-xxx\n"
            "\n处理办法：把「请求头文件」输入框**留空**（用程序内默认的 "
            f"{HEADERS_FILENAME}），或改成真正的请求头内容。\n"
            "⚠ 继续运行会把这些垃圾头发给服务端，表现为毫无线索的"
            "「远程主机强迫关闭了一个现有的连接」(10054)。"
        )

    if headers:
        # 只打印键名，绝不打印值（值里通常是密钥）
        logger.info(f"已从 {target.name} 加载 {len(headers)} 个自定义请求头: {sorted(headers)}")
        return headers

    # ⚠ 一个 Key: Value 都没解析出来 —— 多半是把**依赖清单 / 需求文档 / 任意文本**
    #   当成请求头文件填了。逐行 WARNING 之外，再给一条整体提示，
    #   否则用户会以为「程序读到了，就是没生效」。
    #   纯注释模板（saw_content=False）不算 —— 那正是程序自己生成的样子。
    if saw_content:
        logger.warning(
            f"请求头文件 {target} 里没有解析出任何 Key: Value —— "
            f"它不是请求头文件（要求每行形如 `Authorization: xxx`）。"
            f"留空「请求头文件」输入框即可改用程序内默认的 {HEADERS_FILENAME}。"
        )
    return headers


def build_headers(
    api_key: str,
    language: str,
    extra: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """合并请求头：程序生成的 < 请求头文件 < 用户显式设置。

    优先级（后者覆盖前者）：
      1. `language`（程序按 config 生成；用户文件里写了 language 也以文件为准）
      2. `Authorization`（有 api_key 才加；请求头文件里写了就以文件为准）
      3. 请求头文件内容
    """
    headers: Dict[str, str] = {}
    if language:
        headers["language"] = language
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    for k, v in (extra or {}).items():
        # 空值视为「用户想删掉这个头」
        if v == "":
            headers.pop(k, None)
            continue
        headers[k] = v
    return headers



# ============ 工具：ffmpeg / ffprobe / 切段 / 清理 ============

def _find_ffmpeg() -> Optional[str]:
    """复用 media.py 的定位逻辑（GPT-SoVITS 自带 runtime/ffmpeg.exe 优先）。"""
    from .media import find_ffmpeg

    return find_ffmpeg()


def _find_ffprobe() -> Optional[str]:
    """复用 media.py 的定位逻辑。"""
    from .media import find_ffprobe

    return find_ffprobe()


def _get_session():
    global _cached_session
    if _cached_session is None:
        import requests

        _cached_session = requests.Session()
    return _cached_session


def _release_cached_session() -> None:
    global _cached_session
    if _cached_session is not None:
        try:
            _cached_session.close()
        except Exception:
            pass
        _cached_session = None


def _cleanup_stale_chunk_dirs() -> None:
    """进程启动时清残留切段目录（GUI 强杀 / 崩溃场景）。"""
    pattern = os.path.join(tempfile.gettempdir(), "custom_post_asr_chunks_*")
    for d in glob.glob(pattern):
        shutil.rmtree(d, ignore_errors=True)


atexit.register(_cleanup_stale_chunk_dirs)


def _probe_audio_duration(audio_path: Path) -> float:
    """ffprobe 探测时长（秒）。失败返回 -1。"""
    ffprobe = _find_ffprobe()
    if not ffprobe:
        logger.warning("ffprobe 找不到（无法判断是否需要切段）")
        return -1.0
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "json", str(audio_path)],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            logger.warning(f"ffprobe 失败 (code={result.returncode})")
            return -1.0
        return float(json.loads(result.stdout)["format"]["duration"])
    except (subprocess.TimeoutExpired, json.JSONDecodeError, ValueError, OSError) as e:
        logger.warning(f"ffprobe 异常: {e}")
        return -1.0


def _do_split(audio_path: Path, segment_sec: int) -> List[Path]:
    """ffmpeg 切段到 tempdir，返回所有段路径。"""
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        logger.warning("ffmpeg 找不到，无法切段（超过 500 秒会被 MiniMax 拒绝）")
        return []
    temp_dir = Path(tempfile.mkdtemp(prefix="custom_post_asr_chunks_"))
    chunk_paths: List[Path] = []
    i = 0
    while True:
        out_path = temp_dir / f"chunk_{i:06d}.mp3"
        # 转成单声道 16kHz mp3：既压体积（50MB 上限）又压时长余量
        # 官方明说「识别不依赖高采样率与立体声，用 mp3 精度不受影响」
        cmd = [
            ffmpeg, "-y", "-loglevel", "error",
            "-i", str(audio_path),
            "-ss", str(i), "-t", str(segment_sec),
            "-ar", "16000", "-ac", "1",
            "-b:a", "32k",
            str(out_path),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            if result.returncode != 0:
                logger.warning(f"ffmpeg 切段失败 (chunk {i}): {result.stderr[:200]}")
                break
        except subprocess.TimeoutExpired:
            logger.warning(f"ffmpeg 切段超时 (chunk {i})")
            break
        if not out_path.exists() or out_path.stat().st_size < 1000:
            break
        chunk_paths.append(out_path)
        i += segment_sec
    return chunk_paths


def _cleanup_dir(path: Path) -> None:
    try:
        shutil.rmtree(path, ignore_errors=True)
    except OSError:
        pass


def _split_audio_for_long(audio_path: str, max_duration: float) -> Tuple[List[Path], bool]:
    """长音频切段策略 —— 与 xf_asr 相反，这里**必须**切。

    MiniMax 单请求上限 500 秒 / 50 MB，讯飞是 5 小时 / 500 MB。
    所以：
      - 时长 ≤ 500s 且体积 ≤ 50MB → 单文件直传
      - 超任一上限 → ffmpeg 切 450 秒一段，转单声道 16kHz mp3 压缩

    Returns:
        (段路径列表, 是否切了段)
    """
    audio_path_obj = Path(audio_path)
    duration = max_duration if max_duration > 0 else _probe_audio_duration(audio_path_obj)
    size = audio_path_obj.stat().st_size

    over_duration = duration > 0 and duration > MAX_DURATION_SEC
    over_size = size > MAX_FILE_BYTES
    if not over_duration and not over_size:
        if duration > 0:
            logger.info(
                f"音频 {duration:.1f}s / {size / 1024 / 1024:.2f}MB —— "
                f"在 MiniMax 单请求上限内（{MAX_DURATION_SEC}s / 50MB），整段直传"
            )
        return [audio_path_obj], False

    reason = []
    if over_duration:
        reason.append(f"时长 {duration:.1f}s > {MAX_DURATION_SEC}s")
    if over_size:
        reason.append(f"体积 {size / 1024 / 1024:.1f}MB > 50MB")
    logger.info(
        f"超过 MiniMax 单请求上限（{'、'.join(reason)}），"
        f"自动切 {CHUNK_SEGMENT_SEC} 秒一段并压成单声道 16kHz mp3"
    )
    chunks = _do_split(audio_path_obj, CHUNK_SEGMENT_SEC)
    if not chunks:
        raise RuntimeError(
            f"音频超过 MiniMax 上限（{'、'.join(reason)}）但自动切段失败。"
            f"请检查 ffmpeg 是否可用（程序目录 ffmpeg\\ffmpeg.exe 或 PATH）。"
        )
    logger.info(f"已切 {len(chunks)} 段到 {chunks[0].parent}")
    return chunks, True


# ============ 鉴权 / 参数校验 ============

def _validate_credentials(has_creds: bool) -> None:
    """凭证兜底校验。调用前已判定过，这里只防逻辑漏改。"""
    if not has_creds:
        raise RuntimeError(
            "自定义 POST ASR 没有可用凭证（API Key 为空，且请求头文件里也没有 Authorization）。"
        )


# ============ API 调用 ============

def _extract_error(resp, url: str = "") -> str:
    """把 OpenAI 风格错误体翻译成可读的中文提示。"""
    hint = _ERROR_HINTS.get(resp.status_code, "")
    if not hint and not url:
        hint = "（非标准错误码；若用的是自定义端点，请对照该服务的文档）"
    detail = ""
    try:
        body = resp.json()
        err = body.get("error") or {}
        detail = str(err.get("message") or body.get("message") or "")
        req_id = body.get("request_id") or ""
    except Exception:
        detail = (resp.text or "")[:300]
        req_id = ""
    parts = [f"自定义 POST ASR HTTP {resp.status_code}"]
    if hint:
        parts.append(hint)
    if detail:
        parts.append(f"服务端信息: {detail}")
    if req_id:
        parts.append(f"request_id={req_id}")
    if url:
        parts.append(f"端点={url}")
    return " | ".join(parts)


def build_base(base_url: str) -> str:
    """把用户填的「接口地址」归一成 base（去结尾斜杠 + 去尾部已知路径）。

    用户最容易填错的就是这一层——GUI 里也提示「只填到 /api/v1」。
    这里做防御性清洗：万一填了完整路径，也认得出来，
    不会拼成 /speech_to_text/speech_to_text 这种鬼东西。
    """
    base = str(base_url or "").strip().rstrip("/")
    for suffix in (OPENAI_COMPAT_PATH, DASHSCOPE_TASK_PATH, "/models"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    return base


def build_stt_endpoint(base_url: str, style: str = "openai_compat") -> str:
    """拼出实际调用的转写端点。GUI 用它做实时预览。"""
    base = build_base(base_url)
    if not base:
        return ""
    if style == "dashscope_async":
        return f"{base}{DASHSCOPE_TASK_PATH}"
    return f"{base}{OPENAI_COMPAT_PATH}"


def build_task_query_url(base_url: str, task_id: str) -> str:
    """拼任务查询端点（只有 dashscope_async 用）。"""
    base = build_base(base_url)
    return f"{base}/tasks/{task_id}" if base else ""


def _transcribe_one(
    audio_path: Path,
    api_key: str,
    language: str,
    role_separation: bool,
    timestamps: bool,
    max_wait: int,
    endpoint: str = DEFAULT_ENDPOINT,
    extra_headers: Optional[Dict[str, str]] = None,
) -> dict:
    """单文件真实 API 调用，返回解析后的 JSON dict。

    走真 multipart（与讯飞相反）。
    """
    session = _get_session()
    # verbose_json 同时给 segments[] + n_speakers；只要开分离或时间戳就必须用它
    need_verbose = bool(role_separation or timestamps)
    response_format = "verbose_json" if need_verbose else "json"

    # 统一走 build_stt_endpoint：填完整路径也能归一，不会拼出 /speech_to_text/speech_to_text
    url = build_stt_endpoint(endpoint, "openai_compat")
    headers = build_headers(api_key, language, extra_headers)

    data = {
        "model": _MODEL,
        "response_format": response_format,
        "stream": "false",
    }
    if need_verbose:
        # sentence = 句段时间戳（默认粒度）；word = 中文按字/英文按词
        data["timestamp_level"] = "sentence"

    size_mb = audio_path.stat().st_size / 1024 / 1024
    logger.info(
        f"上传到自定义 POST ASR: {audio_path.name} ({size_mb:.2f}MB, "
        f"format={response_format}, language={language or 'auto'})\n"
        f"  端点: {url}\n"
        f"  请求头键: {sorted(headers)}"
    )

    with open(audio_path, "rb") as f:
        files = {"file": (audio_path.name, f, "application/octet-stream")}
        resp = session.post(url, headers=headers, data=data, files=files,
                            timeout=max(30, max_wait))
    if resp.status_code != 200:
        raise RuntimeError(_extract_error(resp, url))
    try:
        return resp.json()
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"自定义 POST ASR 返回非 JSON (HTTP {resp.status_code}, {url}): "
            f"{(resp.text or '')[:300]}"
        ) from exc


# ============ DashScope 异步流程（2026-10-04 新增）============
# 实测确认：DashScope 的文件级 ASR **不接受文件上传**，必须是公网 URL。
# 所以这里借项目自带的图床（config.image_host）做中转：先把音频传上去拿 URL。

_IMAGE_HOST_MISSING = (
    "DashScope 模式需要音频的公网 URL，但没配置图床。\n"
    "三选一：\n"
    "  ① 到「设置 → 图床」把 provider 切成 r2（Cloudflare，出网免费）\n"
    "     或 oss（阿里云，与百炼同云拉取最快），填 bucket/endpoint/密钥\n"
    "  ② 保持 easyimage，填好真实 api_url + token\n"
    "  ③ 自己把 mp3 传到任意公网可访问的地方，把直链填进「音频直链」输入框"
)

# 与 cover._S3_ENV_KEYS 保持一致（那边是上传侧，这里是校验侧的提示文案）
_S3_ENV_ID = {"r2": "R2_ACCESS_KEY_ID", "oss": "OSS_ACCESS_KEY_ID"}
_S3_ENV_SECRET = {"r2": "R2_SECRET_ACCESS_KEY", "oss": "OSS_ACCESS_KEY_SECRET"}


def _placeholder_host_hint(url: str) -> str:
    """给占位 URL 生成一句 host 侧的说明（不回显整串 URL）。"""
    try:
        host = (urllib.parse.urlsplit(str(url)).hostname or "").strip()
    except Exception:
        return "（无法解析）"
    if not host:
        return "（空）"
    if any(ord(c) > 127 for c in host):
        return f"{host}（含中文 —— 这是占位符，没换成真实域名）"
    return f"{host}（看着像模板/示例域名）"


def _looks_like_placeholder_url(url: str) -> bool:
    """判断 URL 是不是「模板里没替换的占位符」而不是真实地址。

    ⚠ 这是实测踩到的坑：用户 config 里存着模板原样的
    `https://你的图床域名/api/index.php`，非空、能通过所有「有没有配」检查，
    于是程序去请求它 —— 而中文域名会被 requests 自动 punycode 编码成
    `xn--6qqv7isyczza37vgq1b`，报出来一堆人看不懂的东西：
        Failed to resolve 'xn--6qqv7isyczza37vgq1b' ([Errno 11001])
    真正的原因只是「这个占位符没换」。

    判据只认**域名部分**：URL 的 path 完全可以合法含中文。
    """
    raw = str(url or "").strip()
    if not raw:
        return True
    try:
        host = (urllib.parse.urlsplit(raw).hostname or "").strip()
    except Exception:
        return True
    if not host:
        return True
    # ① 域名含非 ASCII —— 真实服务不会用中文域名，几乎必然是没替换的占位
    if any(ord(c) > 127 for c in host):
        return True
    h = host.lower()
    # ② 文档/模板里最常见的占位域名
    if h in {
        "example.com", "example.org", "example.net", "your-domain.com",
        "yourdomain.com", "your-site.com", "yourdomain", "domain.com",
        "test.com", "changeme.com", "placeholder.com", "todo.com",
    }:
        return True
    # ③ your-xxx / my-domain 这类明显占位前缀
    if h.startswith(("your-", "your_", "your.", "my-domain", "mydomain", "xxx.")):
        return True
    # ④ xn-- 是 punycode 前缀；若整体像一个被随机化的字符串也算可疑
    if "xn--" in h and h.replace("-", "").replace("xn", "").replace(".", "").strip(
        "0123456789abcdef"
    ) == "":
        return True
    return False


def upload_audio_for_public_url(audio_path: Path, image_host_config: dict) -> str:
    """把音频借图床/对象存储传成公网 URL（DashScope 必需）。

    DashScope 的文件级 ASR **不接受文件上传**，音频必须是一个 DashScope
    服务端能直接 GET 的公网 URL，所以这里做中转。两条路：
      ① provider=easyimage  传统 multipart 图床（要配 api_url + token）
      ② provider=r2 / oss  S3 兼容对象存储（要配 bucket + endpoint + AK/SK）
    ② 是 2026-10-04 新增的：R2 出网免费，OSS 与百炼同云时拉取最快。
    """
    cfg = image_host_config or {}
    provider = str(cfg.get("provider") or "easyimage").strip().lower()
    api_url = str(cfg.get("api_url") or "").strip()

    # ⚠ 未知 provider 必须在这里就拦：否则像 "cloudflare" / "s3" 这类别名
    # 会静默落进 easyimage 分支，最后报「没配置图床」——
    # 提示指向完全错误的方向（用户会去填 api_url，而真正要填的是 bucket/AK）。
    if provider not in ("easyimage", "r2", "oss"):
        raise ValueError(
            f"不支持的图床提供商: {provider!r}\n"
            "可选值：\n"
            "  easyimage  传统 multipart 图床（填 API URL + Token）\n"
            "  r2         Cloudflare R2（S3 兼容，出网流量永久免费）\n"
            "  oss        阿里云 OSS（S3 兼容，与百炼同云时拉取最快）\n"
            "注意：r2 / oss 要填的是 bucket + endpoint + 密钥，**不是** API URL。"
        )

    if provider in ("r2", "oss"):
        # S3 系不靠 api_url，取而代之的是 bucket/endpoint/AK 三件套
        missing = [
            name for name, val in (
                ("bucket", cfg.get("bucket")),
                ("endpoint", cfg.get("endpoint")),
                ("access_key_id", cfg.get("access_key_id") or os.getenv(_S3_ENV_ID[provider])),
                ("access_key_secret", cfg.get("access_key_secret")
                 or os.getenv(_S3_ENV_SECRET[provider])),
            ) if not str(val or "").strip()
        ]
        if missing:
            raise RuntimeError(
                f"{'Cloudflare R2' if provider == 'r2' else '阿里云 OSS'} 模式缺少配置："
                f"{'、'.join(missing)}\n\n"
                "在「设置 → 图床」把 provider 切成 "
                f"{provider}，然后填这几项（或在 config.json 的 image_host 块）：\n"
                + (
                    "  · R2  endpoint  = https://<AccountID>.r2.cloudflarestorage.com\n"
                    "  · OSS endpoint  = https://oss-cn-<地域>.aliyuncs.com\n"
                    "     （地域建议与你的百炼同区，如 cn-beijing —— 跨云拉取会慢）\n"
                    "  · 两者都要填 bucket 名\n"
                    "  · 密钥可填 access_key_id/access_key_secret，"
                    f"或设环境变量 {_S3_ENV_ID[provider]} / {_S3_ENV_SECRET[provider]}\n\n"
                    "⚠ 别忘了让桶**公网可读**（R2 配 public bucket / r2.dev 域名，"
                    "OSS 配 public-read），否则 DashScope 拉不到；"
                    "或把 presign_seconds 设成 3600 走签名直链。"
                )
            )
        if _looks_like_placeholder_url(str(cfg.get("endpoint"))):
            raise RuntimeError(
                f"{provider} 的 endpoint 还是模板占位符，没换成真实地址。\n"
                f"当前 host 部分：{_placeholder_host_hint(str(cfg.get('endpoint')))}"
            )
    else:
        if not cfg or not api_url:
            raise RuntimeError(_IMAGE_HOST_MISSING)
        if _looks_like_placeholder_url(api_url):
            # ⚠ 不要把原始 URL 原样打出来——那正是报错的根源，重复一遍只会
            # 让用户对着同一个看不懂的 punycode 字符串发呆。只说清「哪里没填」。
            host_hint = _placeholder_host_hint(api_url)
            raise RuntimeError(
                "DashScope 模式需要音频的公网 URL，但图床地址还是**模板里的占位符**，"
                "从没换成真实域名。\n"
                f"当前 host 部分：{host_hint}\n\n"
                "这不是网络问题 —— 那个中文占位符被自动转成了 punycode，"
                "报出来是 'Failed to resolve xn--...'，看着像 DNS 故障，其实是没填。\n\n"
                "三选一：\n"
                "  ① 到「设置 → 图床」填真实域名，或把 provider 切成 r2 / oss\n"
                "     （对象存储，配 bucket + endpoint + AK/SK，R2 出网还免费）\n"
                "  ② 自己把 mp3 传到任意公网可访问处，把直链填进「音频直链」输入框\n"
                "  ③ 改用「OpenAI 兼容」风格 + 不传音频的纯文本服务"
            )

    # 图床上传依赖一览成文的 cover 模块（67KB，含生图/编辑/图床全套），
    # 本移植**不**把它搬进来——GPT-SoVITS 这边没有生图功能，搬了是死代码。
    # 所以这里只在本机确实装了 video_to_article 包时借道，否则给可操作报错。
    try:
        from video_to_article.cover import upload_image_to_host
    except ImportError as e:
        raise RuntimeError(
            "DashScope 模式需要把音频变成公网 URL，本移植没有内置图床上传能力"
            "（未 vendoring 一览成文的 cover 模块）。\n\n"
            "三选一：\n"
            "  ① 直接把 mp3 传到任意公网可访问处，把直链填进「音频直链」输入框\n"
            "  ② pip install video-to-article 后重试（会借道它的图床模块）\n"
            "  ③ 把 asr_engine 改成 custom_post 的 OpenAI 兼容风格"
            "（直接上传文件本体，不需要公网 URL）"
        ) from e

    result = upload_image_to_host(audio_path, cfg)
    url = str((result or {}).get("url") or "")
    if not url:
        raise RuntimeError(f"上传成功但没拿到 URL: {result}")
    logger.info(f"音频已转为公网 URL（provider={provider}）")
    return url


def _dashscope_poll_headers(
    api_key: str, extra: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    """轮询专用的请求头 —— **不能带 X-DashScope-Async**。

    官方文档轮询示例只有两个头：
        curl -X GET '.../api/v1/tasks/{task_id}' \\
          -H "Authorization: Bearer $KEY" \\
          -H "Content-Type: application/json"

    ⚠ 实测（2026-10-04，同一 task_id A/B 对照）：
        带 X-DashScope-Async: enable  → 403
                                     current user api does not support
                                     asynchronous calls
        只带 Authorization+Content-Type → 200，拿到 task_status
    提交接口要 `enable`，轮询接口带了就 403 —— 两者不能共用一个 header 函数。
    这句报错和「模型名填成同步模型」时的报错**完全一样**，极易误判：
    任务明明已经提交成功（拿到了 task_id），却在轮询这一步被拒。
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    for k, v in (extra or {}).items():
        if v and k.lower() != "x-dashscope-async":
            headers[k] = v
    return headers


def _dashscope_headers(api_key: str, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    headers = {"Authorization": f"Bearer {api_key}", "X-DashScope-Async": "enable"}
    for k, v in (extra or {}).items():
        if v:
            headers[k] = v
    return headers


# === 同步 / 异步模型对照（2026-10-04 实测踩到）===
# DashScope 的 ASR 模型**按音频长度分成两族，名字只差一个后缀**：
#   短音频（≤5 分钟）  qwen3-asr-flash        → **同步**，一次请求直接返回
#   长音频（≤12 小时） qwen3-asr-flash-filetrans → **异步**，提交+轮询
# 拿同步模型去打异步端点，服务端只回一句英文
#   current user api does not support asynchronous calls
# 完全看不出「只是少抄了 -filetrans」。这个坑很容易踩，必须在本地拦。
# ⚠ 只列**明确已知**的同步模型，不做「没在异步表里就报错」的推断 ——
#   厂商加新模型时不该让程序反过来挡住可用模型。
_DASHSCOPE_SYNC_ONLY_MODELS = {
    "qwen3-asr-flash": "qwen3-asr-flash-filetrans",
    "qwen3-asr-flash-realtime": "qwen3-asr-flash-filetrans",
    "qwen3-asr-flash-latest": "qwen3-asr-flash-filetrans",
    "qwen-audio-3.0-asr-flash": "qwen-audio-3.0-asr-flash-filetrans",
    "fun-asr-flash": "fun-asr",
    "paraformer-realtime-v2": "paraformer-v2",
}


def _reject_sync_only_model(model: str) -> None:
    """填了只支持同步的模型时，在发请求前用中文说清该用哪个。"""
    m = str(model or "").strip()
    if not m:
        return
    async_name = _DASHSCOPE_SYNC_ONLY_MODELS.get(m.lower())
    if not async_name:
        return
    raise RuntimeError(
        f"模型 {m} 是**同步**模型（只支持 ≤5 分钟音频、一次请求直接返回），"
        "不能用于 DashScope 异步风格。\n\n"
        f"请把「模型名」改成：{async_name}\n\n"
        "两个名字只差一个 -filetrans 后缀，含义完全不同：\n"
        f"  {m:28s} 短音频 ≤5 分钟   同步\n"
        f"  {async_name:28s} 长音频 ≤12 小时  异步（提交 → 轮询 → 下载）\n"
        "服务端只会回 'current user api does not support asynchronous calls'，"
        "看不出是这个原因。"
    )


def _raise_dashscope_submit_error(resp, url: str, model: str) -> None:
    """提交失败时抛错，并对「独享部署只支持同步」这句官方含义做展开。

    官方「非即時語音辨識」文档原文：
        如果您調用**獨享部署**的模型服務時收到報錯
        `current user api does not support asynchronous calls`，
        表示該部署僅支援同步調用，請將請求頭改為 `X-DashScope-Async: disable`。

    ⚠ 这句报错容易被一律当成「模型名填错」，但官方给的是**部署形态**的解释：
      端点形如 https://<自定义>.maas.aliyuncs.com/api/v1 或 qianwenaiapi.com
      的都是**独享部署**（公共端点是 dashscope.aliyuncs.com）。
      也就是说：模型名填对了，这句照样可能出现 —— 那是账号侧的部署配置问题，
      程序改不了，只能去控制台看该部署是否开通了异步能力。
    """
    base = _extract_error(resp, url)
    try:
        blob = json.dumps(resp.json(), ensure_ascii=False).lower()
    except Exception:
        blob = (resp.text or "").lower()
    if "asynchronous" in blob or "异步" in blob:
        raise RuntimeError(
            base
            + "\n\n⚠ 这句 403 绝大多数情况是**模型名填成了同步模型**。\n"
              "  实测（同一个独享部署端点，同一把 Key）：\n"
              "    qwen-audio-3.0-asr-flash-filetrans → 200，正常拿到 task_id ✅\n"
              "    qwen3-asr-flash-filetrans          → 200，正常拿到 task_id ✅\n"
              "    qwen3-asr-flash（同步）              → 403，就是这句 ❌\n"
            f"  你当前填的是：**{model}**\n\n"
              "  → 改成以 -filetrans 结尾的异步模型名再试：\n"
              "      qwen-audio-3.0-asr-flash-filetrans\n"
              "      qwen3-asr-flash-filetrans\n"
              "      fun-asr\n"
              "      paraformer-v2（仅北京地域）\n\n"
              "  官方文档还提到另一种可能：调用**独享部署**时若报这句，说明该部署\n"
              "  只支持同步。独享部署的端点形如 <自定义>.maas.aliyuncs.com /\n"
              "  qianwenaiapi.com（公共端点是 dashscope.aliyuncs.com）。\n"
              "  但按上面实测，换对模型名后你的部署是支持异步的；\n"
              "  若换完仍报同样这句，才需要去控制台确认部署形态。"
        )
    raise RuntimeError(base)


def _dashscope_submit(
    base_url: str,
    api_key: str,
    model: str,
    audio_url: str,
    language: str,
    extra_headers: Optional[Dict[str, str]] = None,
    role_separation: bool = False,
    speaker_count: int = 2,
    colloquial_proc: bool = False,
) -> str:
    """提交异步转写任务，返回 task_id。

    parameters 字段全部来自官方「任务提交接口」文档（2026-10-04 查证）：
      channel_id                  音轨索引（多声道会**分别计费**，默认只取 [0]）
      language_hints              仅 paraformer-v2 支持；其他模型传了可能报错
      diarization_enabled         自动说话人分离；**仅单声道**，多声道不支持
      speaker_count               说话人数参考值（2-100），只是提示不保证
      disfluency_removal_enabled  过滤语气词（嗯/啊/呃）——等价于 xf_asr 的口语规整
    """
    requests = import_required("requests", "requests")
    session = _get_session()
    url = build_stt_endpoint(base_url, "dashscope_async")
    if not model:
        raise RuntimeError(
            "DashScope 模式必须填模型名（如 paraformer-v2 / qwen3-asr-flash-filetrans / "
            "qwen-audio-3.0-asr-flash-filetrans）。"
        )
    _reject_sync_only_model(model)
    # 官方明确警告：URL 含空格/中文必须先 percent-encode，
    # 否则报 InvalidFile.DownloadFailed。用户的课件文件名几乎都是中文。
    #
    # ⚠⚠ `%` **必须**在 safe 里：URL 可能带 S3/R2 的 presigned 签名，
    #   里面 X-Amz-Credential 的值含 `AKID%2F2026...%2Faws4_request`。
    #   若 `%` 不 safe，quote 会把 `%2F` 二次编码成 `%252F`，
    #   R2/S3 收到后解码一次只剩 `AKID%2F...`，再解析 scope 时
    #   「slash-separated parts」= 1，签名直接失效：
    #     InvalidArgument: Credential sigv4 header should have at least
    #     5 slash-separated parts, not 1
    #   实测：同一个 key，原始 presigned URL 能下（200/5.2MB），
    #   经二次编码后 400，且表现为任务 FILE_DOWNLOAD_FAILED。
    #   `%` 放进 safe 不影响中文/空格的编码（它们本来就不在 safe 里）。
    safe_url = urllib.parse.quote(audio_url, safe=":%/?#[]@!$&'()*+,;=")

    params: Dict[str, object] = {"channel_id": [0]}
    if language and "paraformer" in str(model).lower():
        # language_hints 只对 paraformer-v2 系列有效，其他模型传了会 400
        params["language_hints"] = [language]
    if colloquial_proc:
        params["disfluency_removal_enabled"] = True
    if role_separation:
        params["diarization_enabled"] = True
        if 2 <= int(speaker_count or 2) <= 100:
            params["speaker_count"] = int(speaker_count)

    body = {
        "model": model,
        "input": {"file_urls": [safe_url]},
        "parameters": params,
    }
    logger.info(
        f"提交 DashScope 异步任务: {url}\n"
        f"  model={model}  分离={bool(role_separation)}  人数={speaker_count}\n"
        f"  音频 URL={safe_url[:110]}"
    )
    resp = session.post(
        url, headers=_dashscope_headers(api_key, extra_headers), json=body, timeout=60
    )
    if resp.status_code != 200:
        _raise_dashscope_submit_error(resp, url, model)
    payload = resp.json()
    task_id = str(((payload or {}).get("output") or {}).get("task_id") or "")
    if not task_id:
        raise RuntimeError(f"DashScope 没返回 task_id: {str(payload)[:300]}")
    logger.info(f"任务已提交: {task_id}")
    return task_id


def _raise_dashscope_poll_error(resp, url: str, task_id: str) -> None:
    """轮询失败时抛错。任务**已提交成功**，别再把锅甩给模型名。

    ⚠ 实测（2026-10-04）：带 `X-DashScope-Async: enable` 轮询同一个
    已成功提交的 task_id → 403 current user api does not support
    asynchronous calls；去掉该头 → 200 正常返回 task_status。
    也就是说这句报错在**轮询**阶段出现时，含义与**提交**阶段完全不同，
    后者才是「模型名填成同步模型」。不说清的话，用户会去反复改模型名。
    """
    base = _extract_error(resp, url)
    try:
        blob = json.dumps(resp.json(), ensure_ascii=False).lower()
    except Exception:
        blob = (resp.text or "").lower()
    if "asynchronous" in blob or "异步" in blob:
        raise RuntimeError(
            base
            + "\n\n⚠ 注意：任务**已经提交成功**（task_id="
            + task_id
            + "），这一步是**轮询**被拒，不是模型名的问题。\n"
              "  官方文档的轮询接口只要求两个头（Authorization + Content-Type），\n"
              "  **不能带** X-DashScope-Async —— 带了就会得到这句 403。\n"
              "  （这与提交阶段的同名报错是两回事：提交阶段才是模型名填成同步模型。）\n"
              "  如果你是手动调这个接口，去掉 X-DashScope-Async 头即可。"
        )
    raise RuntimeError(base)


def _dashscope_poll(
    base_url: str,
    api_key: str,
    task_id: str,
    max_wait: int,
    extra_headers: Optional[Dict[str, str]] = None,
) -> str:
    """轮询任务直到完成，返回 transcription_url。"""
    requests = import_required("requests", "requests")
    session = _get_session()
    url = build_task_query_url(base_url, task_id)
    deadline = time.time() + max_wait
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        resp = session.get(
            url, headers=_dashscope_poll_headers(api_key, extra_headers), timeout=30
        )
        if resp.status_code != 200:
            _raise_dashscope_poll_error(resp, url, task_id)
        payload = resp.json() or {}
        out = payload.get("output") or {}
        status = str(out.get("task_status") or "").upper()
        if status in DASHSCOPE_DONE:
            # ⚠ 官方明确：整体 SUCCEEDED 时**子任务仍可能 FAILED**，
            #    必须看 results[].subtask_status，否则会拿着一个不存在的 URL 去下载。
            results = out.get("results") or []
            for item in results:
                if not isinstance(item, dict):
                    continue
                sub = str(item.get("subtask_status") or "SUCCEEDED").upper()
                if sub != "SUCCEEDED":
                    code = item.get("code") or ""
                    msg = item.get("message") or ""
                    hint = ""
                    if code == "InvalidFile.DownloadFailed":
                        hint = (
                            "\n常见原因：音频 URL 里有中文/空格没编码好，或图床不允许外网访问。"
                        )
                    raise RuntimeError(
                        f"DashScope 子任务失败（{sub}，code={code}）: {msg}{hint}"
                    )
                link = str(item.get("transcription_url") or "").strip()
                if link:
                    logger.info(f"任务完成（第 {attempt} 次轮询），结果链接已拿到")
                    return link
            inline = out.get("transcription")
            if inline:
                logger.info("任务完成且结果内联，无需二次下载")
                return f"inline:{json.dumps(inline, ensure_ascii=False)}"
            raise RuntimeError(
                f"任务状态是 {status}，但结果里既没有 transcription_url 也没有内联内容: "
                f"{str(payload)[:300]}"
            )
        if status in DASHSCOPE_FAILED:
            raise RuntimeError(
                f"DashScope 任务失败（status={status}）: {str(payload)[:300]}"
            )
        logger.info(f"轮询 #{attempt}: status={status or 'PENDING'}")
        time.sleep(DASHSCOPE_POLL_INTERVAL)
    raise RuntimeError(
        f"DashScope 任务轮询超时（>{max_wait}s, {attempt} 次），task_id={task_id}"
    )


def _dashscope_download(link: str, timeout: int = 120) -> dict:
    """下载 transcription_url 指向的结果 JSON（24 小时有效，务必及时取）。"""
    requests = import_required("requests", "requests")
    session = _get_session()
    if link.startswith("inline:"):
        return json.loads(link[len("inline:"):])
    resp = session.get(link, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(
            f"下载转写结果失败 HTTP {resp.status_code}: {link[:120]}"
        )
    try:
        return resp.json()
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"转写结果不是 JSON（链接可能已过期 24h）: {link[:120]}"
        ) from exc


def _dashscope_transcribe(
    audio_path: Path,
    base_url: str,
    api_key: str,
    model: str,
    language: str,
    max_wait: int,
    audio_url: str = "",
    image_host_config: Optional[dict] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    role_separation: bool = False,
    speaker_count: int = 2,
    colloquial_proc: bool = False,
) -> dict:
    """DashScope 异步转写全流程：拿 URL → 提交 → 轮询 → 下载。

    Returns:
        归一化成 {text, segments(秒), n_speakers} 的 dict，交给 _format_segments 用。
    """
    url = audio_url or upload_audio_for_public_url(audio_path, image_host_config or {})
    task_id = _dashscope_submit(
        base_url, api_key, model, url, language, extra_headers,
        role_separation=role_separation,
        speaker_count=speaker_count,
        colloquial_proc=colloquial_proc,
    )
    link = _dashscope_poll(base_url, api_key, task_id, max_wait, extra_headers)
    payload = _dashscope_download(link)
    return _normalize_dashscope_result(payload)


def _normalize_dashscope_result(payload) -> dict:
    """把 DashScope 结果 JSON 归一成 {text, segments, n_speakers}（**秒**）。

    schema 依据官方「识别结果说明」（2026-10-04 查证原文）：

        {"file_url": ..., "properties": {...},
         "transcripts": [{
            "channel_id": 0,
            "content_duration_in_milliseconds": 3720,
            "text": "整段文本",
            "sentences": [
               {"begin_time": 100, "end_time": 3820,   <-- 毫秒
                "text": "...", "sentence_id": 1,
                "speaker_id": 0,                     <-- 仅开启说话人分离时才有
                "words": [...]}
            ]}]}

    三处最容易踩的：
      1. 顶层是 transcripts[]，sentences 在里面一层
      2. **时间戳单位是毫秒**（官方原文 "Timestamps are in milliseconds"），
         归一时统一除 1000 转成秒。否则 3.2 秒会被显示成 53 分 20 秒。
      3. speaker_id **只在开了 diarization_enabled 时返回**
    """
    if isinstance(payload, str):
        return {"text": payload, "segments": [], "n_speakers": 0}
    if not isinstance(payload, dict):
        return {"text": str(payload), "segments": [], "n_speakers": 0}

    sentences: list = []
    plain = ""
    for tr in (payload.get("transcripts") or []):
        if not isinstance(tr, dict):
            continue
        if not plain:
            plain = str(tr.get("text") or "").strip()
        for sent in (tr.get("sentences") or []):
            if isinstance(sent, dict):
                sentences.append(sent)

    # 兜底：某些部署可能直接给 sentences / 顶层 text
    if not sentences:
        for key in ("sentences", "sentences_list", "segments", "sentence_info"):
            v = payload.get(key)
            if isinstance(v, list) and v:
                sentences = [x for x in v if isinstance(x, dict)]
                break
    if not plain:
        for src in (payload, payload.get("output") or {}):
            for key in ("text", "transcription", "content"):
                v = src.get(key) if isinstance(src, dict) else None
                if isinstance(v, str) and v.strip():
                    plain = v.strip()
                    break
            if plain:
                break

    def _sec(item, *keys) -> float:
        """取毫秒字段并转成秒。"""
        for k in keys:
            if item.get(k) is not None:
                try:
                    return float(item[k]) / 1000.0
                except (TypeError, ValueError):
                    continue
        return 0.0

    segments = []
    for sent in sentences:
        text = str(
            sent.get("text") or sent.get("sentence") or sent.get("content") or ""
        ).strip()
        if not text:
            continue
        spk = sent.get("speaker_id")
        if spk in (None, ""):
            for k in ("speaker", "spk", "speakerId", "role_id"):
                if sent.get(k) not in (None, ""):
                    spk = sent.get(k)
                    break
        segments.append({
            "text": text,
            "start": _sec(sent, "begin_time", "start_time", "begin", "start", "offset"),
            "end": _sec(sent, "end_time", "end"),
            "speaker": (f"spk{spk}" if spk not in (None, "") else ""),
        })

    if not plain and segments:
        plain = "".join(s["text"] for s in segments)
    if not plain and not segments:
        raise RuntimeError(
            "认不出 DashScope 返回结果的结构（可能该部署用别的字段名）。"
            f"顶层键: {sorted(payload)[:12]}"
        )
    speakers = {s["speaker"] for s in segments if s["speaker"]}
    # duration 取服务端给的**真实音频时长**（毫秒 → 秒）。
    # 比本地 ffprobe 探测更准（探测的是切段/转码后的文件），
    # 供 _assemble 的时间轴累加用，省掉一次 ffprobe 调用。
    duration = 0.0
    props = payload.get("properties")
    for src in (props if isinstance(props, dict) else {}, payload):
        if isinstance(src, dict):
            duration = _sec(src, "original_duration_in_milliseconds",
                            "duration_in_milliseconds", "duration")
            if duration:
                break
    if not duration and segments:
        duration = max((s["end"] for s in segments), default=0.0)
    return {
        "text": plain,
        "segments": segments,
        "n_speakers": len(speakers),
        "duration": duration,
    }



# ============ 响应解析 ============

def _format_ts(sec: float) -> str:
    """秒 → [MM:SS]（超 1 小时带小时位）。与 xf_asr / funasr 格式一致。"""
    if sec is None or sec < 0:
        return "00:00"
    total = int(sec)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _format_segments(
    result: dict,
    role_separation: bool,
    timestamps: bool,
    time_offset: float = 0.0,
    speaker_offset: Optional[dict] = None,
) -> List[str]:
    """把 segments[]（或 DashScope 归一后的同构 dict）格式化成行。

    DashScope 那条路经 `_normalize_dashscope_result` 归一后，
    segments 项的键是 text / start / end / speaker（**秒**），
    与这里的 OpenAI 兼容形态不同，故先按是否已有 start 判定。
    两种形态字段名基本同名（speaker vs 无），见下方取值优先级。

    Args:
        time_offset:     段起始时间偏移（秒）—— 切段后每段 start 都从 0 重新计
        speaker_offset:  跨段说话人重映射表 {本段 S1: 全局 S1, ...}。
                         MiniMax 每个独立请求都会重新编号 S1/S2，
                         切段拼接时必须映射，否则第二段又冒出个"说话人1"。
                         传入 None 表示不重映射（单文件时用不到）。
    """
    segments = result.get("segments")
    if not isinstance(segments, list) or not segments:
        return []

    lines: List[str] = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        text = str(seg.get("text") or "").strip()
        if not text:
            continue
        prefix = ""
        if timestamps:
            try:
                start = float(seg.get("start") or 0.0) + time_offset
            except (TypeError, ValueError):
                start = time_offset
            if start > 0:
                prefix += f"[{_format_ts(start)}] "
        if role_separation:
            spk = str(seg.get("speaker") or "")
            if spk:
                if speaker_offset is not None:
                    spk = speaker_offset.get(spk, spk)
                prefix += f"【{spk}】"
        lines.append(f"{prefix}{text}" if prefix else text)
    return lines


def _build_speaker_map(
    prev_map: dict,
    chunk_speakers: List[str],
) -> Tuple[dict, dict]:
    """跨段说话人编号映射（诚实版：多数情况下是恒等映射）。

    **必须先说清楚这个能力的天花板**：MiniMax 每个独立请求都从 S1 重新编号，
    响应里**没有任何跨段身份信息**。所以：

      - 段间人数相同时 → 恒等映射（S1→S1、S2→S2），程序做不了更多
      - 某段人数变多时 → 新出现的编号分配下一个未占用的全局编号（如 S3）
      - 段间人数变少时 → 复用已有编号，不回收

    **不能保证的事**：如果两个人音色接近、在不同段落里的编号互换了
    （第一段 S1 是老师、第二段 S1 是学生），程序无从判断，会认错。
    想要严格一致的跨段说话人身份，只能用单文件长音频的引擎
    （讯飞 lfasr 支持 5 小时单文件）或自己接 pyannote 之类的带全局聚类的方案。

    保留这个函数的价值：让「后一段出现第 3 个人」时不至于撞号，
    并且提供一个将来换成真正全局聚类时的挂载点。
    """
    global_map = dict(prev_map)
    used = set(global_map.values())
    for spk in chunk_speakers:
        if spk in global_map:
            continue
        idx = 1
        while f"S{idx}" in used:
            idx += 1
        global_map[spk] = f"S{idx}"
        used.add(f"S{idx}")
    return global_map, global_map


# ============ 主入口 ============

def transcribe_audio_with_custom_post(audio_path: str, config: dict) -> str:
    """自定义 POST 云端 ASR 主入口（默认端点 = MiniMax 官方，可改成任意兼容服务）。

    Args:
        audio_path: 音频文件路径
        config:     config.json 中 transcribe.custom_post 块

    Returns:
        转写文本

    config 字段：
      - api_key         API Key（写入 `Authorization: Bearer <key>`）
      - endpoint        接口地址，默认 MiniMax 官方
      - headers_file    请求头文件路径；留空 = 程序内 custom_post_headers.txt
      - language        BCP-47；留空 = 自动检测
      - role_separation / timestamps / max_wait_seconds / mock

    Raises:
        RuntimeError: 凭证缺失 / API 调用失败 / 切段失败

    Mock 行为（与 xf_asr 一致的三分支日志）：
        1. mock=true + 凭证齐全 → WARNING（可能忘了取消勾选）
        2. mock=true + 凭证缺失 → INFO
        3. mock=false + 凭证缺失 → WARNING + 自动降级 mock

    切段行为（与 xf_asr **相反**）：
        默认单请求 ≤500 秒 / 50 MB，超了必须切；讯飞是 5 小时不用切。
        换私有部署时若限制不同，改模块顶部 MAX_DURATION_SEC / MAX_FILE_BYTES。
    """
    cfg = config or {}
    mock = bool(cfg.get("mock", False))
    api_key = (cfg.get("api_key") or "").strip()
    endpoint = (cfg.get("endpoint") or DEFAULT_ENDPOINT).strip() or DEFAULT_ENDPOINT
    language = (cfg.get("language") or "").strip()
    role_separation = bool(cfg.get("role_separation", False))
    timestamps = bool(cfg.get("timestamps", True))
    max_wait = int(cfg.get("max_wait_seconds") or 300)
    # 2026-10-04 新增：API 风格 + DashScope 专用字段
    api_style = str(cfg.get("api_style") or "openai_compat").strip()
    if api_style not in API_STYLES:
        logger.warning(
            f"未知的 API 风格 {api_style!r}，回落到 openai_compat"
            f"（可选：{list(API_STYLES)}）"
        )
        api_style = "openai_compat"
    model = str(cfg.get("model") or "").strip()
    audio_url = str(cfg.get("audio_url") or "").strip()
    # DashScope 要公网 URL：优先用户直填，其次借图床中转
    image_host_config: Optional[dict] = None
    if api_style == "dashscope_async" and not audio_url:
        try:
            # 复用本模块的配置桥接：它读的正是同一份一览成文 config.json，
            # 里面的 image_host 段就是图床配置。
            from .config_bridge import load_config

            image_host_config = (load_config() or {}).get("image_host") or {}
        except Exception as e:
            logger.warning(f"读图床配置失败（DashScope 模式需要）: {e}")

    # 请求头文件：留空用程序内默认文件
    headers_path_raw = (cfg.get("headers_file") or "").strip()
    headers_path = Path(headers_path_raw) if headers_path_raw else default_headers_file()
    # 不存在会自动生成模板
    extra_headers = load_headers_file(headers_path)
    if extra_headers and api_style == "dashscope_async":
        logger.info("DashScope 模式也会带上请求头文件里的自定义头")

    # 凭证可以来自两处：API Key 输入框，或请求头文件里的 Authorization。
    # 后者支持非 Bearer 鉴权（私有部署常见），所以空 API Key 不等于没凭证。
    def _has_auth(h: Dict[str, str]) -> bool:
        return any(k.lower() == "authorization" and v.strip() for k, v in h.items())

    has_creds = bool(api_key) or _has_auth(extra_headers)

    if mock or not has_creds:
        if mock and has_creds:
            logger.warning(
                "custom_post 凭证已配置（API Key 或请求头文件里的 Authorization）"
                "但 Mock 模式仍勾选——将走 Mock 不调真实 API。"
                "如需真实 API，请取消 Mock 勾选。"
            )
        elif mock:
            logger.info("custom_post Mock 模式（显式勾选）—— 不调真实 API。")
        else:
            logger.warning(
                "custom_post 凭证缺失，自动降级 Mock 模式（不调网络、不需要 key）。\n"
                f"  填「API Key」输入框，或在请求头文件 {headers_path} 里写 "
                "Authorization: <你的凭据>，然后取消 mock 勾选。"
            )
        return _mock_transcribe(audio_path)

    _validate_credentials(has_creds)
    audio_path_obj = Path(audio_path)

    # 端点格式预检：早点报，比服务端 404 好查
    if not endpoint.lower().startswith(("http://", "https://")):
        hint = (
            f"DashScope 示例: https://dashscope.aliyuncs.com/api/v1"
            if api_style == "dashscope_async"
            else f"MiniMax 官方示例: {DEFAULT_ENDPOINT}"
        )
        raise RuntimeError(
            f"接口地址必须以 http:// 或 https:// 开头，实得: {endpoint!r}。\n"
            f"⚠ 只填到「/api/v1」这一层，后面的路径程序自己拼。{hint}"
        )

    # 格式预检：裸 PCM 会被 400 拒，MP4/MKV 这类容器也不支持
    ext = audio_path_obj.suffix.lower()
    if ext not in SUPPORTED_EXTS:
        logger.warning(
            f"文件后缀 {ext} 不在支持列表（{'/'.join(sorted(SUPPORTED_EXTS))}）——"
            "若报 400 请先转成 mp3/wav（程序下载音轨时默认就是 mp3）"
        )

    # ===== DashScope 异步：单独一条路（音频必须先变成公网 URL）=====
    if api_style == "dashscope_async":
        if not model:
            raise RuntimeError(
                "DashScope 模式必须填「模型名」（如 qwen3-asr-flash-filetrans / "
                "qwen-audio-3.0-asr-flash-filetrans / paraformer-v2）。\n"
                "在「设置 → 转写 → 自定义（POST）高级 → 模型名」填写。"
            )
        t0 = time.time()
        result = _dashscope_transcribe(
            audio_path_obj,
            base_url=endpoint,
            api_key=api_key,
            model=model,
            language=language,
            max_wait=max_wait,
            audio_url=audio_url,
            image_host_config=image_host_config,
            extra_headers=extra_headers,
            role_separation=role_separation,
            speaker_count=int(cfg.get("speaker_count") or 2),
            colloquial_proc=bool(cfg.get("colloquial_proc", False)),
        )
        # ⚠ `duration` 在下面（openai 兼容路径）才探测，本分支在 1520 行之前就 return，
        #   直接用会 UnboundLocalError。这里优先取 DashScope 返回的真实时长
        #   （properties.original_duration_in_milliseconds，比本地探测更准），
        #   拿不到再回落 ffprobe。
        dur = float(result.get("duration") or 0.0)
        if dur <= 0:
            dur = _probe_audio_duration(audio_path_obj)
        text = _assemble(
            [result], role_separation, timestamps,
            chunk_durations=[dur if dur > 0 else 0.0],
        )
        logger.info(
            f"DashScope 转写完成 (耗时: {format_time(time.time() - t0)}，"
            f"共 {len(text)} 字符)"
        )
        _release_cached_session()
        return text

    start = time.time()
    duration = _probe_audio_duration(audio_path_obj)
    chunk_paths, is_chunked = _split_audio_for_long(audio_path, duration)
    temp_dir = chunk_paths[0].parent if is_chunked else None

    try:
        if not is_chunked:
            result = _transcribe_one(
                audio_path_obj, api_key, language, role_separation, timestamps,
                max_wait, endpoint=endpoint, extra_headers=extra_headers,
            )
            text = _assemble(
                [result], role_separation, timestamps,
                chunk_durations=[duration if duration > 0 else 0.0],
            )
            logger.info(
                f"自定义 POST ASR 转写完成 (耗时: {format_time(time.time() - start)}，"
                f"共 {len(text)} 字符)"
            )
            return text

        # === 切段路径：需要累加时间偏移 + 说话人编号对齐 ===
        logger.info(f"长音频转写: {audio_path}（已切 {len(chunk_paths)} 段）")
        results: List[dict] = []
        chunk_durations: List[float] = []
        for i, chunk in enumerate(chunk_paths, 1):
            logger.info(f"转写第 {i}/{len(chunk_paths)} 段: {chunk.name}")
            results.append(
                _transcribe_one(chunk, api_key, language, role_separation, timestamps,
                                max_wait, endpoint=endpoint, extra_headers=extra_headers)
            )
            chunk_durations.append(_probe_audio_duration(chunk))

        text = _assemble(
            results, role_separation, timestamps, chunk_durations=chunk_durations
        )
        if role_separation and is_chunked:
            logger.info(
                "注意：切段后服务端每段独立重编 S1/S2，响应里没有跨段身份信息——"
                "人数相同时程序只能按编号对齐（可能认错人）。"
                "需要严格的跨段说话人身份请用讯飞（5 小时单文件，不用切段）"
            )
        logger.info(
            f"转写完成（{len(chunk_paths)} 段拼接, "
            f"耗时 {format_time(time.time() - start)}，共 {len(text)} 字符）"
        )
        return text
    finally:
        if temp_dir is not None:
            _cleanup_dir(temp_dir)
        _release_cached_session()


def _collect_chunk_speakers(result: dict) -> List[str]:
    """按出现顺序收集本段出现过的说话人编号（S1 / S2 …）。"""
    speakers: List[str] = []
    for seg in result.get("segments") or []:
        if isinstance(seg, dict):
            spk = str(seg.get("speaker") or "")
            if spk and spk not in speakers:
                speakers.append(spk)
    return speakers


def _assemble(
    results: List[dict],
    role_separation: bool,
    timestamps: bool,
    chunk_durations: List[float],
) -> str:
    """把多个请求的结果拼成一份文本。

    - timestamps=True 时累加段偏移，得到全局时间轴
    - role_separation=True 时做跨段说话人重映射
    - 拿不到 segments（用 json 格式 / 老响应）→ 回落到各段 text 直接拼
    """
    lines: List[str] = []
    global_speaker_map: dict = {}
    time_offset = 0.0

    for idx, result in enumerate(results):
        if role_separation:
            # 先扩映射表，再格式化——这样本段新出现的说话人也能拿到全局编号
            _, global_speaker_map = _build_speaker_map(
                global_speaker_map, _collect_chunk_speakers(result)
            )

        seg_lines = _format_segments(
            result,
            role_separation=role_separation,
            timestamps=timestamps,
            time_offset=time_offset,
            speaker_offset=global_speaker_map if (role_separation and global_speaker_map) else None,
        )
        if seg_lines:
            lines.extend(seg_lines)
        else:
            # 没 segments（response_format=json 或服务端没给）→ 整段 text 兜底
            plain = str(result.get("text") or "").strip()
            if plain:
                lines.append(plain)

        if idx < len(chunk_durations):
            time_offset += max(0.0, float(chunk_durations[idx] or 0.0))

    return "\n".join(lines)


def _mock_transcribe(audio_path: str) -> str:
    """Mock 转写：返回 fake 中文文本（基于音频时长）。"""
    duration = _probe_audio_duration(Path(audio_path))
    n = max(5, int(duration // 30) + 1) if duration > 0 else 5
    sentence = (
        "（这是自定义 POST ASR 的 mock 转写文本。"
        "在「设置 → 转写 → 自定义（POST）高级」填入 API Key（或在请求头文件里写 "
        "Authorization）并取消 mock 勾选即可调用真实接口。）"
    )
    return "\n".join([sentence] * min(n, 50))


# ============ 旧引擎名兼容 ============

def transcribe_audio_with_minimax_asr(audio_path: str, config: dict) -> str:
    """**已弃用**的旧入口，转发到 transcribe_audio_with_custom_post()。

    2026-10-02：本引擎从「MiniMax 专用」泛化成「自定义 POST」后保留此别名，
    避免用户 config.json 里已保存的 "asr_engine": "minimax_asr" 静默失效
    （那会回落到 funasr，用户以为在用云端其实在本地跑，很难查）。
    """
    logger.warning(
        'asr_engine="minimax_asr" 已改名为 "custom_post"（本引擎已泛化为自定义 POST）。'
        '本次仍按旧名执行，请把 config.json 里的 asr_engine 改成 "custom_post"，'
        "并把 transcribe.minimax_asr 块改名为 transcribe.custom_post。"
    )
    return transcribe_audio_with_custom_post(audio_path, config)
