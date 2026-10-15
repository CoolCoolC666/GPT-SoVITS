"""讯飞听见云端 ASR（长语音转写 REST API）。

=========================== 移植说明 ===========================

本文件由 CoolCoolC666/video-to-article（MIT）的
src/video_to_article/media/xf_asr.py 移植而来，只改了四处：

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
import base64
import glob
import hashlib
import hmac
import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.parse
from pathlib import Path
from typing import List, Optional

from .logutil import configure_logging
from .text_utils import format_time

logger = configure_logging()


# 讯飞长语音转写 v2 API 端点
_UPLOAD_URL = "https://raasr.xfyun.cn/v2/api/upload"
_RESULT_URL = "https://raasr.xfyun.cn/v2/api/getResult"

# 长音频自动切段阈值
CHUNK_THRESHOLD_SEC = 450  # > 7.5 分钟自动切段
CHUNK_SEGMENT_SEC = 300    # 每段 5 分钟
MAX_FILE_BYTES = 500 * 1024 * 1024  # 讯飞单文件 ≤500MB

# ffmpeg/ffprobe 路径（项目内副本 + 已知 skill 路径 + PATH fallback）

# 讯飞任务状态码
# 讯飞 raasr v2 API 任务状态码（参考 caitongbo/3drx.top/szfx.top demo 一致）：
#   - status = 3 → 仍在处理中（demo while status == 3: ... time.sleep(5)）
#   - status = 4 → 处理完成（demo if status == 4: break）
# 2026-09-30 订正：之前误写 0-4/9/5，全部错。
_STATUS_PROCESSING = frozenset({3})
_STATUS_SUCCESS = 4

# 2026-10-02 新增：可配置的转写增强参数（走 upload 的 URL query string）
#
# 垂直领域（pd）—— 讯飞为特定领域准备了定制声学+语言模型：
#   court 法律 / edu 教育 / finance 金融 / medical 医疗 / tech 科技
#   sport 体育 / gov 政府 / game 游戏 / ecom 电商 / car 汽车
#   空字符串 = 通用模型（讯飞默认）
PD_DOMAINS = {
    "": "通用（不设领域）",
    "edu": "教育（课程/课堂实录）",
    "court": "法律",
    "finance": "金融",
    "medical": "医疗",
    "tech": "科技",
    "sport": "体育",
    "gov": "政府",
    "game": "游戏",
    "ecom": "电商",
    "car": "汽车",
}

# 说话人分离的失败降级错误码：
#   26600 转写业务通用错误 —— 常见于「参数不支持 / 权限未开通」
#   26610 请求参数错误       —— 常见于「参数名或取值不被该账号接受」
# 官方 Java SDK 文档明确写：role_type「只有在开通了角色分离功能的前提下才会生效」，
# 且「发音人分离目前还是测试效果达不到商用标准」。所以这里不硬失败——
# 带上增强参数失败就自动退回最小参数集重试，保证转写本身不中断。
_DEGRADE_CODES = frozenset({"26600", "26610"})

# 模块级 cache：HTTP Session 复用（多段上传共用连接池）
_cached_session: Optional[object] = None


# ============ 工具：ffmpeg / ffprobe / 切段 ============

def _find_ffmpeg() -> Optional[str]:
    """复用 media.py 的定位逻辑（GPT-SoVITS 自带 runtime/ffmpeg.exe 优先）。"""
    from .media import find_ffmpeg

    return find_ffmpeg()


def _find_ffprobe() -> Optional[str]:
    """复用 media.py 的定位逻辑。"""
    from .media import find_ffprobe

    return find_ffprobe()


def _probe_audio_duration(audio_path: str) -> float:
    """用 ffprobe 探测音频时长（秒）。失败返回 -1。"""
    ffprobe = _find_ffprobe()
    if not ffprobe:
        logger.warning("ffprobe 找不到，跳过时长探测（不切段）")
        return -1.0
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "json", audio_path],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            logger.warning(f"ffprobe 失败 (code={result.returncode})，不切段")
            return -1.0
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])
    except (subprocess.TimeoutExpired, json.JSONDecodeError, ValueError, OSError) as e:
        logger.warning(f"ffprobe 异常: {e}，不切段")
        return -1.0


def _do_split(audio_path: str, segment_sec: int, total_duration: float) -> List[Path]:
    """ffmpeg 切段到 temp 目录，返回所有段路径。"""
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        logger.warning("ffmpeg 找不到，跳过切段（直接转写原文件）")
        return [Path(audio_path)]
    temp_dir = Path(tempfile.mkdtemp(prefix="xf_asr_chunks_"))
    chunk_paths = []
    i = 0
    while i < int(total_duration):
        out_path = temp_dir / f"chunk_{i:06d}.wav"
        cmd = [
            ffmpeg, "-y", "-loglevel", "error",
            "-i", audio_path,
            "-ss", str(i), "-t", str(segment_sec),
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
            str(out_path),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                logger.warning(f"ffmpeg 切段失败 (chunk {i}): {result.stderr[:200]}")
                break
        except subprocess.TimeoutExpired:
            logger.warning(f"ffmpeg 切段超时 (chunk {i})")
            break
        if not out_path.exists() or out_path.stat().st_size < 1000:
            # 最后一段可能 < segment_sec，文件太小说明切完了
            break
        chunk_paths.append(out_path)
        i += segment_sec
    if not chunk_paths:
        logger.warning("切段失败，回退到原文件")
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except OSError:
            pass
        return [Path(audio_path)]
    logger.info(f"已切 {len(chunk_paths)} 段到 {temp_dir}")
    return chunk_paths


def _split_audio_for_long(
    audio_path: str,
    segment_sec: int = CHUNK_SEGMENT_SEC,
) -> List[Path]:
    """长音频处理策略。

    2026-09-30 简化：客户端不再切段。
    讯飞 raasr.xfyun.cn/v2/api 服务端自己会处理长音频（官方文档明示 ≤500MB
    单文件直传即可），客户端切段反而引入风险——任何一段失败都整体失败，
    切段边界可能引入静音或 wav 头丢失，且增加 HTTP 请求次数。

    保留函数签名 + threshold 常量以便未来重新启用（如果 server 端改了限制）。

    Returns:
        永远 [Path(audio_path)]——不切段
    """
    audio_path_obj = Path(audio_path)
    duration = _probe_audio_duration(audio_path_obj)
    if duration > 0:
        logger.info(
            f"音频时长 {duration:.1f}s——讯飞 raasr v2 API 服务端自动处理长音频，"
            f"客户端不再切段（{audio_path_obj.name}）"
        )
    else:
        logger.info(f"讯飞 raasr v2 API 直接上传：{audio_path_obj.name}")
    return [audio_path_obj]


def _cleanup_chunks(chunk_paths: List[Path]) -> None:
    """清理切段产生的 temp 目录和文件。"""
    if not chunk_paths:
        return
    temp_dirs = {p.parent for p in chunk_paths if p.parent.name.startswith("xf_asr_chunks_")}
    for d in temp_dirs:
        try:
            shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass


def _cleanup_orphaned_chunks_on_exit() -> None:
    """2026-09-27: 进程退出时清残留 xf_asr_chunks_* tempdir。

    正常 finally 块会清，但 GUI 被强杀/任务管理器结束/断电时 finally 不跑，
    tempdir 永久残留 %TEMP%。atexit 在 Python 进程正常退出时一定会跑。
    """
    cleaned = 0
    for d in glob.glob(os.path.join(tempfile.gettempdir(), "xf_asr_chunks_*")):
        try:
            shutil.rmtree(d, ignore_errors=True)
            cleaned += 1
        except Exception:
            pass
    if cleaned:
        logger.info(f"[atexit] 清理残留切段目录: {cleaned} 个")


atexit.register(_cleanup_orphaned_chunks_on_exit)


# ============ 签名 + API 调用 ============

def _sign_request(app_id: str, secret_key: str, ts: int) -> str:
    """生成讯飞长语音 API 签名（MD5 + HMAC-SHA1 + base64）。

    官方算法（参考 caitongbo/Speech-to-Text Ifasr_new.py，URL 是 raasr.xfyun.cn/v2/api）：
        1. md5_hex  = MD5(app_id + ts).hexdigest()           # 32-hex msg
        2. raw     = HMAC-SHA1(secret_key, md5_hex.bytes)    # 用 MD5 hex 当 msg
        3. signa   = base64(raw).decode()

    2026-09-27 订正：之前误写成 base64(HMAC-SHA1(secret, f"{app_id}{ts}"))
    漏了 MD5 一步——讯飞服务端返回 26601 signa verify fail。已按官方 demo 修复。

    注意：baseString 不是 app_id+ts 原串，而是 app_id+ts 先 MD5 一次。
    与「实时语音转写 WebSocket」的 'api_key={key}&timestamp={ts}' 完全不同。
    """
    md5 = hashlib.md5()
    md5.update(f"{app_id}{ts}".encode("utf-8"))
    md5_hex = md5.hexdigest().encode("utf-8")  # 32-hex msg
    sig = hmac.new(secret_key.encode("utf-8"), md5_hex, hashlib.sha1).digest()
    return base64.b64encode(sig).decode("utf-8")


def _get_session():
    """获取 / 创建 requests.Session（连接池复用）。"""
    global _cached_session
    if _cached_session is None:
        import requests  # 延迟 import：避免未装 requests 时启动报错
        _cached_session = requests.Session()
    return _cached_session


def _release_cached_session() -> None:
    """释放 Session（多段转写完成后调用，避免连接占资源）。"""
    global _cached_session
    if _cached_session is not None:
        try:
            _cached_session.close()
        except Exception:
            pass
        _cached_session = None


def _upload_audio(
    audio_path: str,
    app_id: str,
    secret_key: str,
    language: str = "cn",
    role_separation: bool = False,
    role_num: int = 2,
    pd_domain: str = "",
    colloquial_proc: bool = False,
) -> str:
    """上传音频到讯飞 API，返回 task_id（= orderId）。

    讯飞 raasr.xfyun.cn/v2/api/upload 的真实协议 —— 与 caitongbo/Speech-to-Text
    Ifasr_new.py 等官方/民间 Python demo 一致：

      1. 所有鉴权/元参数 (appId / signa / ts / fileSize / fileName /
         duration / language / sliceSize / ...) 走 URL **query string**，
         拼成 `…/upload?appId=xxx&signa=yyy&ts=zzz&...`
      2. 音频二进制 bytes 直接走 HTTP body（`requests.post(data=...)`）
      3. Content-Type header = `application/json`

    **不要**用 `requests.post(files={"data": ...})`！requests 会自动生成
    `multipart/form-data; boundary=xxx` 格式，与讯飞期望的「query string +
    raw body」完全不一致 → 服务端返回 code=26600
    「转写业务通用错误 | 音频需上传二进制音频流数据 | 音频使用表单方式上传」。

    与实时语音识别（WebSocket）鉴权不同，HTTP 长语音 REST API 走 query string。

    历史：2026-09-30 修正。之前用 multipart files，被 26600 挡住。
    参照多 demo（caitongbo/Speech-to-Text Ifasr_new.py / richice/-/xf_lfasr.py /
    JianyaoChen/video-carrier/xf_lfasr.py / lanbinleo/bili2text/xunfei.py）
    一致方案。

    2026-10-02 新增可选增强参数（全部走 query string）：
      - role_separation + role_num → roleType=1 / roleNum=N（说话人分离）
      - pd_domain                 → pd=edu（教育领域定制模型）
      - colloquial_proc           → eng_colloqproc=true（口语规整：去「嗯啊呃」+ 口癖）
    任一增强参数被账号拒绝（26600/26610）→ 自动退回最小参数集重试一次，
    不让可选能力变成硬失败（官方明确 role_type 需开通权限才生效）。

    Raises:
        RuntimeError: 上传失败（凭证错 / 文件过大 / 网络问题）
    """
    session = _get_session()
    ts = int(time.time())
    signa = _sign_request(app_id, secret_key, ts)

    audio_size = Path(audio_path).stat().st_size
    if audio_size > MAX_FILE_BYTES:
        raise RuntimeError(
            f"音频文件过大 ({audio_size / 1024 / 1024:.0f}MB)，"
            f"讯飞长语音 API 单文件上限 500MB（应自动切段却失败？检查 ffmpeg）"
        )

    # === 探测时长（用于 form_data 的 duration 字段）===
    # demo 一致必传 duration 参数（单位秒，估算转写耗时用）
    # ⚠️ duration 不准可能让 server 误判超时
    audio_duration = _probe_audio_duration(Path(audio_path))

    # === 最小参数集（鉴权 + 文件元信息，demo 一致必传，不可省）===
    base_params = {
        "appId": app_id,
        "signa": signa,
        "ts": str(ts),
        "fileSize": str(audio_size),
        "fileName": Path(audio_path).name,
        "sliceSize": str(audio_size),
        "language": language,
        "duration": str(int(audio_duration)) if audio_duration > 0 else "60",
    }

    # === 增强参数（可选，被拒也不致命）===
    enhanced_params: dict = {}
    if role_separation:
        # 标准版 lfasr 只支持 roleType=1（通用角色分离）
        # roleNum=0 表示自动盲分，1..10 表示指定人数
        enhanced_params["roleType"] = "1"
        enhanced_params["roleNum"] = str(int(role_num))
    if pd_domain:
        enhanced_params["pd"] = pd_domain
    if colloquial_proc:
        enhanced_params["eng_colloqproc"] = "true"

    logger.info(
        f"上传音频到讯飞: {audio_path} "
        f"({audio_size / 1024 / 1024:.2f}MB, language={language})"
        + (f" 增强参数={enhanced_params}" if enhanced_params else "")
    )

    # === 音频二进制走 raw body（不是 multipart files）===
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    def _do_upload(params: dict):
        url = f"{_UPLOAD_URL}?{urllib.parse.urlencode(params)}"
        resp = session.post(
            url,
            headers={"Content-Type": "application/json"},
            data=audio_bytes,
            timeout=180,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"讯飞 upload HTTP {resp.status_code}: {resp.text[:300]}"
            )
        try:
            return resp.json()
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"讯飞 upload 返回非 JSON (HTTP {resp.status_code}): {resp.text[:300]}"
            ) from exc

    result = _do_upload({**base_params, **enhanced_params})

    # === 失败降级：增强参数被账号拒绝 → 退回最小参数集重试 ===
    code = str(result.get("code", ""))
    if code in _DEGRADE_CODES and enhanced_params:
        logger.warning(
            f"讯飞 upload 拒绝增强参数 {sorted(enhanced_params)} (code={code}: "
            f"{result.get('descInfo', '')})，自动退回最小参数集重试"
            "（说话人分离/领域优化需账号开通对应能力）"
        )
        result = _do_upload(base_params)

    if result.get("code") != "000000":
        raise RuntimeError(
            f"讯飞 upload 失败 (code={result.get('code')}): "
            f"{result.get('descInfo', '')}"
        )
    # 2026-09-30 订正：JSON key 是 orderId，不是 taskId（讯飞文档拼写）
    # user 实测报 KeyError: 'taskId'（2026-09-30 22:06 截图）——所有 demo 一致用 orderId
    task_id = result["content"]["orderId"]
    logger.info(f"上传成功，task_id={task_id}")
    return task_id


def _iter_segments(data1: dict):
    """把 orderResult 里每段话拆出来，产出 (角色编号, 起止毫秒, 文本)。

    2026-10-02：新增读取 `st.rl`（角色编号）与 `st.bg`/`st.ed`（句段起止毫秒）。
    之前只抽 `ws[].cw[].w` 拼纯文本，把这两个字段全丢了 —— 白捡的东西。

    真实结构（caitongbo 字符串版 / csdn dict 版都兼容）：
        data1 = {"lattice": [...], "lattice2": [...], "label": {...}}
        segment = {
            "begin": "50", "end": "1840", "spk": "段落-0",   # lattice2 顶层
            "json_1best": "…" 或 {...},                        # 两种形式 demo 都有
        }
        json_1best = {"st": {
            "bg": "50", "ed": "1840", "rl": "0",              # ← 角色 + 时间
            "rt": [{"ws": [{"wb": 1, "we": 16, "cw": [{"w": "这"}]}]}],
        }}

    角色编号取值优先级：segment.spk（lattice2 顶层）> st.rl。
    时间取值优先级：segment.begin/end > st.bg/ed > 全 0。
    """
    segments = data1.get("lattice2") or data1.get("lattice") or []
    for seg in segments:
        json_1best_raw = seg.get("json_1best")
        if json_1best_raw is None:
            continue
        # 兼容 string（caitongbo 旧 demo）和 dict（csdn 新 demo）
        if isinstance(json_1best_raw, str):
            try:
                json_1best = json.loads(json_1best_raw)
            except json.JSONDecodeError:
                continue
        else:
            json_1best = json_1best_raw
        st = json_1best.get("st", {})

        chars: List[str] = []
        for rt in st.get("rt", []):
            for ws in rt.get("ws", []):
                for cw in ws.get("cw", []):
                    if isinstance(cw, dict):
                        w = cw.get("w")
                        if w:
                            chars.append(w)
        text = "".join(chars)
        if not text:
            continue

        # 角色编号：lattice2 顶层 spk（"段落-0"）优先，回落 st.rl（"0"/"1"）
        speaker = seg.get("spk")
        if speaker in (None, ""):
            rl = st.get("rl")
            speaker = f"spk{rl}" if rl not in (None, "") else None

        # 起止时间（毫秒）：lattice2 顶层 begin/end 优先，回落 st.bg/ed
        def _ms(top_key: str, st_key: str) -> int:
            raw = seg.get(top_key, st.get(st_key))
            try:
                return int(raw)
            except (TypeError, ValueError):
                return 0

        yield speaker, _ms("begin", "bg"), _ms("end", "ed"), text


def _format_ms(ms: int) -> str:
    """毫秒 → HH:MM:SS（超过 1 小时才带小时位）。"""
    if ms <= 0:
        return "00:00:00"
    total = ms // 1000
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _parse_result_content(
    content: dict,
    role_separation: bool = False,
    timestamps: bool = True,
) -> str:
    """解析 getResult 返回的 content，按时间顺序拼成可读文本。

    2026-09-30 订正：response.content.orderResult 是 **JSON 字符串**（不是 dict）。

    真实结构（caitongbo/Speech-to-Text Ifasr_new.py + csdn 多 demo 一致）：

        orderResult_str (string)
        └─ json.loads(orderResult_str)
           └─ { "lattice": [...], "lattice2": [...], "label": {...} }
              └─ lattice[] / lattice2[] (list of segments)
                 └─ segment.json_1best (string OR dict，demo 两种写法都有)
                    └─ json.loads(...) → {"st": {"rt": [{"ws": [{"cw": [{"w": "字"}]}]}]}}

    2026-10-02：输出格式随配置变化（都保留 LLM 可读的纯文本形态）：

        都关：      今天我们讲印象派。
        开时间戳：  [00:12] 今天我们讲印象派。
        开说话人：  [00:12] 【说话人1】今天我们讲印象派。
        都开：      [00:12] 【说话人1】今天我们讲印象派。

    兼容策略：
    - lattice2 优先（带 begin/end/spk），fallback 到 lattice（caitongbo）
    - json_1best 兼容 string（caitongbo）和 dict（csdn）
    - 一个角色都没解析出来时自动退回纯文本模式（不输出空的【说话人】标签）
    """
    order_result_str = content.get("orderResult")
    if not order_result_str:
        return ""
    try:
        data1 = json.loads(order_result_str)
    except (json.JSONDecodeError, TypeError) as exc:
        # 防御性日志：HTTP layer 的 JSONDecodeError 已经先抓了一层，
        # 这里只是兜底（空字符串 / 异常 payload），用 INFO 避免污染正常日志
        logger.info(f"_parse_result_content 拿到非 JSON orderResult（防御性 fallback）: {exc}")
        return ""

    rows = list(_iter_segments(data1))
    if not rows:
        return ""

    # 说话人开关开了但一个角色号都没解析到 → 退回不带标签（避免满屏空标签）
    has_speaker = any(speaker for speaker, _, _, _ in rows)
    use_speaker = role_separation and has_speaker
    if role_separation and not has_speaker:
        logger.info(
            "已开启说话人分离，但响应里没有 st.rl / spk 字段"
            "（可能账号未开通角色分离能力），本次输出不带说话人标签"
        )

    # 角色编号 → 连续编号（rl=0/1 映射成 说话人1/说话人2，读起来更顺）
    speaker_map: dict = {}
    if use_speaker:
        order = [s for s, _, _, _ in rows if s and s not in speaker_map]
        speaker_map = {s: f"说话人{i + 1}" for i, s in enumerate(order)}

    lines: List[str] = []
    for speaker, begin_ms, _end_ms, text in rows:
        prefix = ""
        if timestamps and begin_ms > 0:
            prefix += f"[{_format_ms(begin_ms)}] "
        if use_speaker and speaker:
            prefix += f"【{speaker_map.get(speaker, speaker)}】"
        lines.append(f"{prefix}{text}" if prefix else text)
    return "\n".join(lines)


def _poll_result(
    task_id: str,
    app_id: str,
    secret_key: str,
    max_wait: int = 600,
    poll_interval: float = 5.0,
    role_separation: bool = False,
    timestamps: bool = True,
) -> str:
    """轮询讯飞结果直到完成，返回转写文本。

    任务状态码（`content.orderInfo.status`）：
    - 3: 仍在处理中
    - 4: 处理完成
    其他值 → 视为失败并抛错。

    Args:
        role_separation: 解析时给每段加【说话人N】标签
        timestamps:      解析时给每段加 [MM:SS] 前缀

    Raises:
        RuntimeError: 超时 / 任务失败
    """
    session = _get_session()
    deadline = time.time() + max_wait
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        ts = int(time.time())
        signa = _sign_request(app_id, secret_key, ts)
        form_data = {
            "appId": app_id,
            "signa": signa,
            "ts": str(ts),
            "orderId": task_id,  # 2026-09-30 订正：JSON 字段是 orderId
        }
        try:
            resp = session.post(_RESULT_URL, data=form_data, timeout=30)
        except Exception as exc:
            # 网络抖动：WARNING + 重试，不立即抛
            logger.warning(f"poll #{attempt} 网络异常: {exc}，重试")
            time.sleep(poll_interval)
            continue
        if resp.status_code != 200:
            logger.warning(f"poll #{attempt} HTTP {resp.status_code}，重试")
            time.sleep(poll_interval)
            continue
        try:
            result = resp.json()
        except json.JSONDecodeError:
            logger.warning(f"poll #{attempt} 非 JSON 响应，重试")
            time.sleep(poll_interval)
            continue
        if result.get("code") != "000000":
            raise RuntimeError(
                f"讯飞 getResult 失败 (code={result.get('code')}): "
                f"{result.get('descInfo', '')}"
            )
        # 2026-09-30 订正：响应结构是 content.orderInfo.status，不是 content.taskStatus
        # demo 一致：caicongbo / 3drx.top / szfx.top 都是 result['content']['orderInfo']['status']
        # user 实测 KeyError: 'taskStatus'（2026-09-30 22:08）
        status = int(result["content"]["orderInfo"]["status"])
        logger.info(f"poll #{attempt}: status={status}")
        if status in _STATUS_PROCESSING:
            time.sleep(poll_interval)
            continue
        if status == _STATUS_SUCCESS:
            return _parse_result_content(
                result["content"],
                role_separation=role_separation,
                timestamps=timestamps,
            )
        raise RuntimeError(f"未知任务状态: status={status}")
    raise RuntimeError(
        f"讯飞轮询超时 ({max_wait}s, {attempt} 次尝试)，"
        f"task_id={task_id}"
    )


def _validate_credentials(app_id: str, secret_key: str) -> None:
    """校验讯飞凭证非空。

    长语音转写鉴权只需 APPID + SecretKey 两件套（短音频才要 APIKey）。
    """
    missing = []
    if not app_id:
        missing.append("app_id")
    if not secret_key:
        missing.append("secret_key")
    if missing:
        raise RuntimeError(
            f"讯飞长语音鉴权需要 2 项凭证（APPID + SecretKey），缺少: {', '.join(missing)}。"
            f"请到 https://www.xfyun.cn/ 注册 → 控制台 → 「语音转写」服务创建应用获取。"
            f"或勾「Mock 模式」跳过真实 API（用于本地调试）。"
        )


def _mock_transcribe(audio_path: str) -> str:
    """Mock 转写：返回 fake 中文文本（基于音频时长生成）。

    仅用于本地开发 / UI 验证：
    - 不调网络、不需要凭证
    - 文本内容固定，每 30 秒重复一行
    - 真实接入凭证后，config 设 mock=false 即可切换真实 API
    """
    duration = _probe_audio_duration(audio_path)
    if duration > 0:
        n = max(1, int(duration / 30))
    else:
        # 探测失败（如没有 ffprobe）：默认 5 行
        n = 5
    sentence = (
        "（这是讯飞听见 mock 模式的 fake 转写文本。"
        "真实接入凭证后关闭 mock 即可调用云端 API。）"
    )
    return "\n".join([sentence] * min(n, 50))  # 上限 50 行


# ============ 主入口 ============

def _transcribe_one(
    audio_path: str,
    app_id: str,
    secret_key: str,
    language: str,
    max_wait: int,
    role_separation: bool = False,
    role_num: int = 2,
    pd_domain: str = "",
    colloquial_proc: bool = False,
    timestamps: bool = True,
) -> str:
    """单文件真实 API：upload + poll。"""
    task_id = _upload_audio(
        audio_path,
        app_id,
        secret_key,
        language=language,
        role_separation=role_separation,
        role_num=role_num,
        pd_domain=pd_domain,
        colloquial_proc=colloquial_proc,
    )
    return _poll_result(
        task_id,
        app_id,
        secret_key,
        max_wait=max_wait,
        role_separation=role_separation,
        timestamps=timestamps,
    )


def transcribe_audio_with_xf_asr(audio_path: str, config: dict) -> str:
    """讯飞听见云端 ASR 主入口。

    Args:
        audio_path: 音频文件路径（wav/mp3/m4a/flac 等）
        config: config.json 中 transcribe.xf_asr 块

    Returns:
        转写文本

    Raises:
        RuntimeError: 凭证缺失 / API 调用失败 / 文件过大

    Mock 行为：
        - mock=true 或凭证缺失 → _mock_transcribe(audio_path)
        - 真实凭证 + mock=false → 调讯飞 API

    可选增强（2026-10-02 新增，全部走 upload query string）：
        - role_separation + role_num  → 说话人分离（【说话人1】标签）
        - pd_domain="edu"             → 教育领域定制模型
        - colloquial_proc=true        → 口语规整（去「嗯啊呃」+ 口癖）
        - timestamps                  → 每段加 [MM:SS] 前缀
    增强参数被账号拒绝时自动退回最小参数集重试，不让可选能力变成硬失败。

    长音频处理：
        2026-09-30 起**不切段**——讯飞 server 自己处理长音频（≤500MB / 5h），
        客户端切段反而引入 wav 头丢失 + 边界静音风险。
    """
    cfg = config or {}
    mock = bool(cfg.get("mock", False))
    app_id = (cfg.get("app_id") or "").strip()
    secret_key = (cfg.get("secret_key") or "").strip()
    language = (cfg.get("language") or "cn").strip()
    max_wait = int(cfg.get("max_wait_seconds") or 600)
    role_separation = bool(cfg.get("role_separation", False))
    try:
        role_num = int(cfg.get("role_num", 2))
    except (TypeError, ValueError):
        role_num = 2
    role_num = max(0, min(10, role_num))  # 讯飞上限 0..10（0 = 自动盲分）
    pd_domain = (cfg.get("pd_domain") or "").strip()
    colloquial_proc = bool(cfg.get("colloquial_proc", False))
    timestamps = bool(cfg.get("timestamps", True))

    # Mock 模式：分三种情况打不同日志（让用户一眼看出走了哪条分支）
    #   1. mock=true + 凭证齐全 → 用户显式勾选 mock（可能忘了取消）→ WARNING
    #   2. mock=true + 凭证缺失 → 用户显式勾选 mock → INFO
    #   3. mock=false + 凭证缺失 → 自动降级 mock（凭证缺失自动 fallback）→ WARNING
    if mock or not (app_id and secret_key):
        if mock and app_id and secret_key:
            # 情况 1：凭证齐全但勾了 mock（用户可能忘了取消勾选）
            logger.warning(
                "xf_asr 凭证已配置（APPID + SecretKey）但 Mock 模式仍勾选——"
                "将走 Mock 不调真实 API。如需真实 API，请取消 Mock 勾选。"
            )
        elif mock:
            # 情况 2：用户显式勾选 mock（无论凭证）
            logger.info(
                "xf_asr Mock 模式（显式勾选）—— 不调真实 API，返回 fake 中文文本。"
            )
        else:
            # 情况 3：凭证缺失自动降级
            logger.warning(
                "xf_asr 凭证缺失，自动降级 Mock 模式（不调网络、不需要 key）。"
                "如需真实 API，请配置 app_id + secret_key 并取消 mock 勾选。"
            )
        return _mock_transcribe(audio_path)

    # 真实 API 模式
    _validate_credentials(app_id, secret_key)
    audio_path_obj = Path(audio_path)
    chunk_paths = _split_audio_for_long(audio_path)
    is_chunked = len(chunk_paths) > 1 or chunk_paths[0] != audio_path_obj
    start = time.time()
    try:
        if is_chunked:
            logger.info(
                f"讯飞长音频转写: {audio_path}（已切 {len(chunk_paths)} 段）"
            )
            texts = []
            for i, chunk in enumerate(chunk_paths, 1):
                logger.info(
                    f"转写第 {i}/{len(chunk_paths)} 段: {chunk.name}"
                )
                texts.append(
                    _transcribe_one(
                        str(chunk),
                        app_id,
                        secret_key,
                        language,
                        max_wait,
                        role_separation=role_separation,
                        role_num=role_num,
                        pd_domain=pd_domain,
                        colloquial_proc=colloquial_proc,
                        timestamps=timestamps,
                    )
                )
            result = "\n".join(texts)
            logger.info(
                f"讯飞转写完成（{len(chunk_paths)} 段拼接, "
                f"耗时 {format_time(time.time() - start)}，共 {len(result)} 字符）"
            )
            return result
        result = _transcribe_one(
            audio_path,
            app_id,
            secret_key,
            language,
            max_wait,
            role_separation=role_separation,
            role_num=role_num,
            pd_domain=pd_domain,
            colloquial_proc=colloquial_proc,
            timestamps=timestamps,
        )
        logger.info(
            f"讯飞转写完成 (耗时 {format_time(time.time() - start)}，"
            f"共 {len(result)} 字符)"
        )
        return result
    finally:
        if is_chunked:
            _cleanup_chunks(chunk_paths)
        # 短音频不释放 session（单次任务复用即可）；长音频末尾释放更稳
        if is_chunked:
            _release_cached_session()