"""运行期路径。

移植说明：一览成文的 paths.py 把 APP_ROOT 当作「exe 旁边的目录」（PyPI 包 +
绿色包双形态）。本模块是装在 GPT-SoVITS 目录里的库，没有 exe，所以：

- APP_ROOT  = GPT-SoVITS 项目根（本文件的 .../tools/video2article/paths.py 往上两级）
- PROMPTS_DIR = **模块自带**的 prompts/（模板随仓库走，不依赖外部安装）
- OUTPUT_DIR  = <GPT-SoVITS>/output/video2article（output/ 本来就在 .gitignore 里）

关于 PROMPTS_DIR 为什么不给外部覆盖能力：一览成文的模板可以按视频类型自由新增，
而本仓库把模板当**受版本控制的资产**，用户要加模板应直接改仓库里的 .md，
这样别人 clone 下来拿到的是同一套模板。外部覆盖留给配置文件的 llm 部分即可。
"""

from __future__ import annotations

import contextlib
import sys
import tempfile
from pathlib import Path

_MODULE_DIR = Path(__file__).resolve().parent


def get_app_root() -> Path:
    """GPT-SoVITS 项目根目录。"""
    # .../GPT-SoVITS/tools/video2article/paths.py → parents[2] = GPT-SoVITS
    if _MODULE_DIR.parent.name == "tools":
        return _MODULE_DIR.parent.parent
    # 兜底：被拷到别处时退回 CWD
    return Path.cwd()


APP_ROOT = get_app_root()

MODULE_DIR = _MODULE_DIR
PROMPTS_DIR = MODULE_DIR / "prompts"
OUTPUT_DIR = APP_ROOT / "output" / "video2article"
LOGS_DIR = APP_ROOT / "logs"
RUNTIME_DIR = APP_ROOT / "runtime"

# 切片等临时产物的落点。
#
# ⚠ 为什么不用 tempfile 默认目录：`tempfile.mkdtemp()` 默认写到
#   C:\Users\<用户>\AppData\Local\Temp。两个云端引擎对长音频会切段
#   （custom_post 450 秒一段、讯飞历史上也切），一个 1 小时的视频能堆出
#   几百 MB 到 **1GB** 的切片——全记在 C 盘账上。本机 C 盘常年偏紧，
#   所以强制改到项目内的 TEMP（E 盘）。
MODULE_TEMP_DIR = APP_ROOT / "TEMP" / "video2article"

# 一览成文的自定义请求头文件（可能含密钥）——放在模块目录，仓库忽略它
CUSTOM_POST_HEADERS_FILE = MODULE_DIR / "custom_post_headers.txt"

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv", ".webm", ".m4v", ".rm", ".rmvb"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma"}
LOCAL_MEDIA_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS


def ensure_runtime_dirs() -> None:
    """建好本模块要写的目录。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    MODULE_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    (PROMPTS_DIR / "system").mkdir(parents=True, exist_ok=True)
    (PROMPTS_DIR / "articles").mkdir(parents=True, exist_ok=True)


@contextlib.contextmanager
def use_local_temp():
    """把 tempfile 的落点临时改到项目内的 TEMP（E 盘）。

    ## 为什么要这么做

    两个云端引擎都对长音频切段（custom_post 每 450 秒一段）。切片默认由
    `tempfile.mkdtemp()` 创建，落在系统 TEMP —— 也就是 **C 盘**。一次
    1 小时视频的转写切片能到几百 MB，反复跑几次就把 C 盘的账做大了。

    ## 为什么用上下文管理器而不是直接改全局

    `tempfile.tempdir` 是**进程级**全局量。在 WebUI 里它被改成什么，会影响
    同进程内所有别的库（Gradio、torch 数据集缓存……）。所以这里只在
    「跑一次转写」这个明确边界内改，跑完立刻还原，且用 try/finally 保证
    异常路径也还原。

    嵌套使用时，内层退出后恢复到外层设的值（存的是原始值，不是硬编码）。
    """
    previous = tempfile.tempdir
    MODULE_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(MODULE_TEMP_DIR)
    try:
        yield MODULE_TEMP_DIR
    finally:
        tempfile.tempdir = previous
