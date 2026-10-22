"""云端 ASR 引擎在「0c-语音识别工具」里的共用执行层。

## 这个文件解决三件事

### 1. 输出格式必须和本地引擎完全一致

上游 `tools/asr/funasr_asr.py` 产出的标注文件是：

    {输出目录}/{输入目录名}.list      每行：文件路径|说话人|语言|文本

下游 `GPT_SoVITS/prepare_datasets/1-get-text.py` 按 `|` 切成四段，
`tools/subfix_webui.py`（0d 标注）也按行读。**格式不一致 = 下游静默丢数据**，
所以云端引擎必须逐字对齐这个格式。

### 2. 文本里的时间戳/说话人标签要洗掉

云端引擎返回的是带标记的行，例如：

    [00:10] 日常生活呢其实这是我们非常
    [00:13] 【说话人1】普及的，我们可以说……

这些标记对「看稿」有用，但对**训练语料**是噪声——训练时模型要学的是文字本身。
所以这里剥掉前缀，只留纯文本。

### 3. 语言标记要映射成训练侧认的 token

`1-get-text.py` 只认 `ZH/EN/JA/KO/yue`（大小写都收），**别的会被打印警告然后跳过**。
讯飞用的是自己的 `cn`，custom_post 可能给 `auto`，都得映射过去，
否则整份 .list 会被下游丢弃，而错误只出现在远处的一行警告里。
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, List, Optional, Tuple

ASR_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ASR_DIR.parent.parent

# 工具以子进程方式被拉起（python tools/asr/xxx.py），此时 sys.path[0] 是 tools/asr，
# 拿不到项目根，所以显式补上——要 import tools.video2article 里的引擎实现。
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 训练侧接受的语言 token（见 1-get-text.py 的 language_v1_to_language_v2）
TRAIN_LANGS = {"ZH", "EN", "JA", "KO", "YUE"}

# 各引擎的语言写法 → 训练 token
LANG_MAP = {
    # 讯飞
    "cn": "ZH", "zh": "ZH", "zh-cn": "ZH", "mandarin": "ZH",
    "en": "EN", "en-us": "EN", "en-gb": "EN",
    "ja": "JA", "jp": "JA",
    "ko": "KO", "kr": "KO",
    "yue": "YUE", "zh-hk": "YUE",
    # custom_post / 通用
    "chinese": "ZH", "english": "EN", "japanese": "JA", "korean": "KO",
    "auto": "ZH",
}

# 行内标记：[00:10] / [00:01:23] / 【说话人1】
#
# ⚠ 重复组之间必须允许空白：讯飞的实际输出是 "[00:13] 【说话人1】普及的…"，
#   时间戳和说话人标签之间**有一个空格**。早期版本写成
#   `(?:\[\d+:\d+\]|【[^】]+】)+` 时，匹配到 "[00:13]" 就停了，
#   后面那个空格让重复中断，【说话人1】 整段留在了正文里——训练语料里
#   混进这种标签，模型会学着念出来。
_TAG_RE = re.compile(r"^\s*(?:\s*(?:\[\d{1,2}:\d{2}(?::\d{2})?\]|【[^】]{1,20}】))+\s*")

# 说话人标签可能出现在行中间（不同引擎的格式不一），一律剥掉
_SPEAKER_TAG_RE = re.compile(r"【\s*说话人\s*\d*\s*】")

_BRACKET_ANY_RE = re.compile(r"[\[\【][^\]】]{0,20}[\]\】]")


def map_lang(lang: str, text: str = "") -> str:
    """把引擎的语言写法映射成训练侧 token。映射不了就按文本内容猜。"""
    key = str(lang or "").strip().lower()
    if key in LANG_MAP:
        mapped = LANG_MAP[key]
        if mapped != "ZH" or key != "auto":
            return mapped
    # auto 或未知：按有没有汉字猜
    if any("\u4e00" <= ch <= "\u9fff" for ch in text or ""):
        return "ZH"
    if text and all(ord(ch) < 128 for ch in text):
        return "EN"
    return "ZH"


def clean_text(text: str, drop_all_brackets: bool = False) -> str:
    """剥掉时间戳/说话人标签，归一空白。

    `drop_all_brackets=True` 时连正文里残留的方括号内容也一并去掉
    （用于自动检测语言时的粗判）。
    """
    if not text:
        return ""
    lines = []
    for raw in str(text).splitlines():
        line = _TAG_RE.sub("", raw).strip()
        # 说话人标签在哪都剥——它不含任何语言信息，留在训练语料里会被模型学走
        line = _SPEAKER_TAG_RE.sub("", line).strip()
        if not line:
            continue
        lines.append(line)
    joined = "".join(lines) if _is_cjk("".join(lines)) else " ".join(lines)
    joined = re.sub(r"[ \t]+", " ", joined).strip()
    if drop_all_brackets:
        joined = _BRACKET_ANY_RE.sub("", joined)
    return joined


def _is_cjk(text: str) -> bool:
    """文本主体是不是中文——中文段落之间不加空格，英文要加。"""
    if not text:
        return False
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return cjk * 2 >= len(text)


def ffmpeg_path() -> Optional[str]:
    for candidate in (
        PROJECT_ROOT / "runtime" / "ffmpeg.exe",
        PROJECT_ROOT / "runtime" / "ffmpeg",
        PROJECT_ROOT / "ffmpeg.exe",
    ):
        if candidate.is_file():
            return str(candidate)
    from shutil import which

    return which("ffmpeg")


def ffprobe_path() -> Optional[str]:
    for candidate in (
        PROJECT_ROOT / "runtime" / "ffprobe.exe",
        PROJECT_ROOT / "runtime" / "ffprobe",
        PROJECT_ROOT / "ffprobe.exe",
    ):
        if candidate.is_file():
            return str(candidate)
    from shutil import which

    return which("ffprobe")


def _probe(path: Path) -> Tuple[int, int]:
    """返回 (采样率, 声道数)；探测不到返回 (0, 0)。"""
    probe = ffprobe_path()
    if not probe:
        return 0, 0
    try:
        out = subprocess.run(
            [probe, "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=sample_rate,channels",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        if out.returncode != 0:
            return 0, 0
        parts = out.stdout.strip().split(",")
        if len(parts) < 2:
            return 0, 0
        return int(parts[0]), int(parts[1])
    except (ValueError, OSError, subprocess.TimeoutExpired):
        return 0, 0


def normalize_audio(path: Path, work_dir: Path) -> Path:
    """归一化到 16kHz 单声道 WAV。

    为什么必须做：
    - 0b 切片保留源采样率，44.1k 立体声的 wav 直接上传，体积是 16k 单声道的
      近 5 倍 → 上传时间成倍增加（这正是「加速」的一部分）
    - 讯飞对采样率有偏好，16k 是通用最佳采样率

    已经合规的文件**原样返回**，不做无谓的转码。
    """
    rate, channels = _probe(path)
    if rate == 16000 and channels == 1:
        return path

    ffmpeg = ffmpeg_path()
    if not ffmpeg:
        # 没有 ffmpeg 就只能原样上传，让云端自己处理
        return path

    work_dir.mkdir(parents=True, exist_ok=True)
    out_path = work_dir / f"{path.stem}_16k_mono.wav"
    try:
        result = subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", str(path),
             "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out_path)],
            capture_output=True, text=True, timeout=300,
        )
    except (OSError, subprocess.TimeoutExpired):
        return path
    if result.returncode != 0 or not out_path.is_file() or out_path.stat().st_size < 512:
        return path
    return out_path


def list_audio_files(input_folder: str) -> List[Path]:
    """列出输入目录里的音频文件（排序，保证输出稳定）。"""
    folder = Path(input_folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"输入目录不存在: {folder}")
    exts = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".wma", ".aac", ".opus"}
    files = [p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in exts]
    return files


def transcribe_folder(
    input_folder: str,
    output_folder: str,
    transcribe_one: Callable[[Path], str],
    language: str = "zh",
    concurrency: int = 4,
    speaker_name: str = "",
) -> str:
    """跑完整文件夹的转写，产出标注文件，返回它的路径。

    `transcribe_one(path) -> str` 由具体引擎实现。

    ## 为什么用线程池

    云端引擎的耗时全在网络等待上（上传 + 轮询），CPU 基本闲着，
    线程池在这里是真并行（网络 IO 会释放 GIL）。
    本地引擎（funasr）反而**不能**这么干——它是 GPU 推理，
    并发只会抢显存甚至 OOM，所以本地引擎维持串行。

    并发数默认 4：讯飞/自建网关都有限流，并发太高会触发 429。
    """
    files = list_audio_files(input_folder)
    output_folder = output_folder or "output/asr_opt"
    os.makedirs(output_folder, exist_ok=True)

    folder_name = os.path.basename(os.path.normpath(input_folder))
    speaker = speaker_name or folder_name
    out_path = os.path.abspath(os.path.join(output_folder, f"{folder_name}.list"))

    if not files:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("")
        print(f"输入目录里没有音频文件，已写出空标注文件: {out_path}")
        return out_path

    # 临时目录放在项目内 TEMP，避免占用系统盘（C 盘）
    with tempfile.TemporaryDirectory(prefix="asr_cloud_", dir=str(_project_temp_dir())) as tmp:
        work_dir = Path(tmp)
        results = []

        workers = max(1, min(int(concurrency or 1), 16, len(files)))
        print(f"共 {len(files)} 个文件，云端转写并发 {workers}")

        def _task(path: Path):
            try:
                audio = normalize_audio(path, work_dir)
                text = transcribe_one(audio)
                return path, text, ""
            except Exception:
                return path, "", traceback.format_exc()

        if workers == 1:
            for path in files:
                results.append(_task(path))
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_task, p) for p in files]
                for i, future in enumerate(as_completed(futures), 1):
                    results.append(future.result())
                    print(f"  进度 {i}/{len(files)}")

        # 按文件名排序回写，保证 .list 顺序稳定（多说话人训练靠这个对齐）
        results.sort(key=lambda item: item[0].name)

        lines = []
        for path, text, error in results:
            if error:
                print(f"[失败] {path.name}\n{error}")
                continue
            clean = clean_text(text)
            if not clean:
                print(f"[空结果] {path.name}（该片段可能没有人声）")
                continue
            lang_token = map_lang(language, clean)
            lines.append(f"{path}|{speaker}|{lang_token}|{clean}")

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"ASR 任务完成->标注文件路径: {out_path}（{len(lines)}/{len(files)} 条）")
    return out_path


def _project_temp_dir() -> Path:
    temp_dir = PROJECT_ROOT / "TEMP" / "asr_cloud"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir
